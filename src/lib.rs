use std::collections::HashMap;
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyPermissionError, PyValueError};
use pyo3::prelude::*;

pub mod engine;
pub mod mcp;
pub mod memory;
pub mod paged_attention;
pub mod scheduler;
pub mod telemetry;
pub mod tui;
pub mod validation;

use engine::NexusEngine;
use mcp::{McpClient, SecurityLevel, Transport};
use memory::NexusChannel;
use paged_attention::{KVCacheAllocator, LogicalMemory};
use scheduler::{AgentState, ContinuousScheduler};
use telemetry::NexusProfiler;
use validation::SemanticInterceptor;

/// Returns the Nexus-Core bare-metal engine initialization banner.
#[pyfunction]
fn engine_status() -> &'static str {
    "Nexus-Core Bare-Metal Engine Initialized"
}

/// Doubles every element of a 1-D float64 NumPy array in place, releasing the
/// GIL for the entire compute path.
///
/// The buffer is shared zero-copy via the NumPy/buffer protocol: Rust takes a
/// mutable view over Python-owned memory and mutates it directly. No
/// allocation, no copy, no return value. While the inner loop runs, Python's
/// GIL is released through `Python::allow_threads`, proving that the Rust
/// core can perform heavy work concurrently with other Python threads.
#[pyfunction]
fn mutate_tensor_in_place<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyArray1<f64>>,
) -> PyResult<()> {
    // SAFETY: the caller transfers exclusive mutable access to the underlying
    // buffer for the duration of this call; no other Python reference is
    // expected to alias `array` while the view is live.
    let mut view = unsafe { array.as_array_mut() };
    py.allow_threads(|| {
        for value in view.iter_mut() {
            *value *= 2.0;
        }
    });
    Ok(())
}

/// Python-facing handle to the Codata Substrate (`NexusChannel<String>`).
///
/// The handle is cheaply cloneable across Python references because the
/// substrate itself lives behind an `Arc`; multiple `PyNexusChannel` views
/// of the same channel are deliberately supported for future multi-agent
/// scenarios.
#[pyclass]
struct PyNexusChannel {
    inner: Arc<NexusChannel<String>>,
}

#[pymethods]
impl PyNexusChannel {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(NexusChannel::new(capacity)),
        }
    }

    /// Deposits a context token into the substrate. Surfaces capacity
    /// overflow and other Rust-side rejections as `ValueError` on the Python
    /// side, per the agentic contract (no silent drops).
    fn push(&self, item: String) -> PyResult<()> {
        self.inner.push(item).map_err(PyValueError::new_err)
    }

    /// Returns the most recent context as an active-state read. Non-
    /// destructive: the substrate retains the item.
    fn peek_context(&self) -> Option<String> {
        self.inner.peek_context()
    }
}

/// Python-facing handle to the PagedAttention KV-cache manager.
///
/// Owns the physical-block allocator together with the table of logical
/// sequences. CoW is exposed via `branch_sequence`: a child sequence is
/// just a new logical id pointing at the parent's `BlockId` vector with
/// every ref-count bumped.
#[pyclass]
struct PyKVCacheManager {
    allocator: KVCacheAllocator,
    sequences: HashMap<usize, LogicalMemory>,
    next_sequence_id: usize,
}

#[pymethods]
impl PyKVCacheManager {
    #[new]
    fn new(max_blocks: usize) -> Self {
        Self {
            allocator: KVCacheAllocator::new(max_blocks),
            sequences: HashMap::new(),
            next_sequence_id: 0,
        }
    }

    /// Allocates `num_blocks` fresh physical blocks and binds them to a new
    /// logical sequence. Rolls back partial allocations on failure to keep
    /// the allocator's ref-counts consistent.
    fn allocate_sequence(&mut self, num_blocks: usize) -> PyResult<usize> {
        let mut blocks: Vec<usize> = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match self.allocator.allocate_block() {
                Ok(id) => blocks.push(id),
                Err(message) => {
                    for id in blocks {
                        self.allocator.free_block(id);
                    }
                    return Err(PyValueError::new_err(message));
                }
            }
        }
        let sequence_id = self.next_sequence_id;
        self.next_sequence_id += 1;
        self.sequences
            .insert(sequence_id, LogicalMemory { blocks });
        Ok(sequence_id)
    }

    /// Branches a child sequence from `parent_sequence_id` via Copy-on-Write:
    /// the child reuses every parent `BlockId` and the allocator merely bumps
    /// each block's ref-count. No physical block is consumed.
    fn branch_sequence(&mut self, parent_sequence_id: usize) -> PyResult<usize> {
        let blocks = {
            let parent = self.sequences.get(&parent_sequence_id).ok_or_else(|| {
                PyValueError::new_err("parent sequence id does not exist")
            })?;
            parent.blocks.clone()
        };
        for &id in &blocks {
            self.allocator.duplicate_block(id);
        }
        let sequence_id = self.next_sequence_id;
        self.next_sequence_id += 1;
        self.sequences
            .insert(sequence_id, LogicalMemory { blocks });
        Ok(sequence_id)
    }

    /// Number of physical blocks currently free across the entire cache.
    fn get_available_blocks(&self) -> usize {
        self.allocator.available_blocks()
    }
}

/// Python-facing handle to the Continuous Batching scheduler.
///
/// Owns a `ContinuousScheduler` and exposes the minimum surface needed by
/// the Python orchestrator: register a task, run a tick, and signal halt
/// (tool round-trip) or finish (generation done). Every mutating method
/// takes `&mut self`; PyO3 enforces single-threaded mutation under the GIL.
#[pyclass]
struct PyContinuousScheduler {
    inner: ContinuousScheduler,
}

#[pymethods]
impl PyContinuousScheduler {
    #[new]
    fn new(max_batch_size: usize) -> Self {
        Self {
            inner: ContinuousScheduler::new(max_batch_size),
        }
    }

    /// Registers a new task in the `Waiting` state and returns its id.
    fn add_task(&mut self) -> usize {
        self.inner.add_task()
    }

    /// One engine tick. Returns the ids of every task currently in the
    /// active batch, after admitting waiting tasks into freed slots.
    fn step(&mut self) -> Vec<usize> {
        self.inner.step()
    }

    /// Signals that the task has finished generating. It will not re-enter
    /// the active batch on subsequent `step()` calls.
    fn mark_finished(&mut self, task_id: usize) -> PyResult<()> {
        self.inner
            .set_task_state(task_id, AgentState::Finished)
            .map_err(PyValueError::new_err)
    }

    /// Signals that the task has stalled on an external dependency
    /// (typically an MCP tool round-trip). It vacates its batch slot
    /// immediately, freeing the seat for a waiting task on the next tick.
    fn mark_halted(&mut self, task_id: usize) -> PyResult<()> {
        self.inner
            .set_task_state(task_id, AgentState::Halted)
            .map_err(PyValueError::new_err)
    }
}

/// Python-facing handle to the unified Nexus inference engine.
///
/// Wraps a `NexusEngine` that fuses the PagedAttention block allocator
/// with the Continuous Batching scheduler. The Python surface is
/// intentionally minimal — `add_agent` to enqueue work, `step` to drive
/// one tick — because the load-bearing behaviour the engine exists to
/// guarantee (recycling blocks across an oversubscribed workload) is
/// fully internal.
#[pyclass]
struct PyNexusEngine {
    inner: NexusEngine,
}

#[pymethods]
impl PyNexusEngine {
    #[new]
    fn new(max_blocks: usize, max_batch_size: usize) -> Self {
        Self {
            inner: NexusEngine::new(max_blocks, max_batch_size),
        }
    }

    /// Enqueues a new agent that will require `required_blocks` blocks of
    /// KV cache before it is considered done. Returns the agent id.
    fn add_agent(&mut self, required_blocks: usize) -> usize {
        self.inner.add_agent(required_blocks)
    }

    /// Drives one engine tick. Returns `True` while there is still work
    /// to do, `False` once the workload is fully drained.
    fn step(&mut self) -> bool {
        self.inner.step()
    }
}

/// Python-facing handle to the Cognitive Reliability Layer.
///
/// Wraps a `SemanticInterceptor` and exposes a single high-level
/// operation: `validate_with_retry` simulates the full auto-correction
/// loop the orchestrator runs against a real LLM. Each entry of
/// `simulated_outputs` stands in for one model attempt; the interceptor
/// rejects malformed or schema-violating attempts in Rust, prints the
/// correction prompt that the orchestrator would otherwise re-issue,
/// and only the first valid attempt is surfaced to Python.
#[pyclass]
struct PySemanticValidator {
    inner: SemanticInterceptor,
}

#[pymethods]
impl PySemanticValidator {
    #[new]
    fn new(required_keys: Vec<String>) -> Self {
        Self {
            inner: SemanticInterceptor::new(required_keys),
        }
    }

    /// Iterates over `simulated_outputs` until one of them clears the
    /// contract or the retry budget is exhausted. Correction prompts
    /// from rejected attempts are printed to stdout so a test harness
    /// can observe the loop without depending on log capture.
    fn validate_with_retry(
        &self,
        simulated_outputs: Vec<String>,
        max_retries: usize,
    ) -> PyResult<String> {
        let required_keys: Vec<String> = self.inner.required_keys().to_vec();
        let attempts = simulated_outputs.len().min(max_retries);

        for attempt in simulated_outputs.iter().take(attempts) {
            match self.inner.enforce_contract(attempt, required_keys.clone()) {
                Ok(valid_json) => return Ok(valid_json),
                Err(correction_prompt) => {
                    println!("{}", correction_prompt);
                }
            }
        }

        Err(PyValueError::new_err(
            "Quarantined: max retries exhausted without a contract-compliant response.",
        ))
    }
}

/// Python-facing handle to the Zero-Trust MCP client.
///
/// Wraps an `McpClient` with the mock transport wired in. The Python
/// surface is a single `invoke_tool` that mirrors the real MCP entry
/// point and routes every call through the Rust gatekeeper. Policy
/// denials surface as `PermissionError` (not `ValueError`) so callers
/// can branch deterministically on security violations.
#[pyclass]
struct PyMcpClient {
    inner: McpClient,
}

#[pymethods]
impl PyMcpClient {
    #[new]
    fn new(security_level_str: String) -> PyResult<Self> {
        let level = SecurityLevel::from_str(&security_level_str)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: McpClient::new(Transport::Mock, level),
        })
    }

    fn invoke_tool(&self, tool_name: String, payload: String) -> PyResult<String> {
        self.inner
            .invoke_tool(&tool_name, &payload)
            .map_err(PyPermissionError::new_err)
    }
}

/// Python-facing handle to the lock-free hardware profiler.
///
/// Wraps a `NexusProfiler` whose three counters are independent
/// `AtomicU64`s. The Python surface is intentionally
/// fire-and-forget — no return values from the recording methods —
/// because the cost of the FFI hop must stay close to the cost of the
/// underlying `lock xadd` for the profiler's "zero-overhead" claim to
/// survive being driven from Python.
#[pyclass]
struct PyNexusProfiler {
    inner: NexusProfiler,
}

#[pymethods]
impl PyNexusProfiler {
    #[new]
    fn new() -> Self {
        Self {
            inner: NexusProfiler::new(),
        }
    }

    fn record_ttft(&self, elapsed_micros: u64) {
        self.inner.record_ttft(elapsed_micros);
    }

    fn record_token_generated(&self) {
        self.inner.record_token_generated();
    }

    fn update_peak_vram(&self, current_blocks: u64) {
        self.inner.update_peak_vram(current_blocks);
    }

    /// Snapshots all three counters into a Python dictionary. Each
    /// value is coerced to `f64` so the dashboard side does not need
    /// to branch on numeric type — at metric scale the precision loss
    /// for `total_tokens` and `peak_vram_blocks` is irrelevant.
    fn export_metrics(&self) -> PyResult<HashMap<String, f64>> {
        let mut out = HashMap::with_capacity(3);
        out.insert("avg_ttft_ms".to_string(), self.inner.get_avg_ttft_ms());
        out.insert(
            "total_tokens".to_string(),
            self.inner.total_tokens() as f64,
        );
        out.insert(
            "peak_vram_blocks".to_string(),
            self.inner.peak_vram() as f64,
        );
        Ok(out)
    }
}

/// PyO3 entry point for the `nexus_core` extension module.
#[pymodule]
fn nexus_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_status, m)?)?;
    m.add_function(wrap_pyfunction!(mutate_tensor_in_place, m)?)?;
    m.add_class::<PyNexusChannel>()?;
    m.add_class::<PyKVCacheManager>()?;
    m.add_class::<PyContinuousScheduler>()?;
    m.add_class::<PyNexusEngine>()?;
    m.add_class::<PySemanticValidator>()?;
    m.add_class::<PyMcpClient>()?;
    m.add_class::<PyNexusProfiler>()?;
    Ok(())
}
