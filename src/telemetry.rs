//! Lock-free hardware profiler for the inference hot path.
//!
//! Every counter the profiler exposes is an `AtomicU64` updated with
//! `Ordering::Relaxed`. The choice is deliberate: a `Mutex` around
//! these fields would serialize every token-producing thread on a
//! single critical section, which is precisely the contention pattern
//! Nexus-Core's bare-metal engine is designed to avoid. `Relaxed`
//! ordering is sufficient because each counter is independent and we
//! never read two counters together expecting a consistent snapshot —
//! the export path tolerates stale reads of one field relative to
//! another, and that tradeoff is what buys us the zero-overhead claim.

use std::sync::atomic::{AtomicU64, Ordering};

/// Three independent counters that together describe the engine's
/// throughput, latency, and memory pressure. The struct is `Sync`
/// without further wrapping because every field is itself atomic, so
/// the same `&NexusProfiler` can be shared across the inference
/// thread, the TUI redraw task, and the Python exporter without any
/// `Arc<Mutex<...>>` plumbing.
pub struct NexusProfiler {
    total_tokens_generated: AtomicU64,
    cumulative_ttft_micros: AtomicU64,
    peak_vram_blocks: AtomicU64,
}

impl NexusProfiler {
    pub fn new() -> Self {
        Self {
            total_tokens_generated: AtomicU64::new(0),
            cumulative_ttft_micros: AtomicU64::new(0),
            peak_vram_blocks: AtomicU64::new(0),
        }
    }

    /// Adds one TTFT sample (in microseconds) to the running sum. The
    /// average is reconstructed at export time as
    /// `cumulative_ttft / total_tokens`. `#[inline(always)]` because
    /// this sits in the per-token critical path; we want the call to
    /// fold into a single `lock xadd` once monomorphized.
    #[inline(always)]
    pub fn record_ttft(&self, elapsed_micros: u64) {
        self.cumulative_ttft_micros
            .fetch_add(elapsed_micros, Ordering::Relaxed);
    }

    /// Increments the token counter. Same inlining rationale as
    /// `record_ttft`: this fires for every emitted token.
    #[inline(always)]
    pub fn record_token_generated(&self) {
        self.total_tokens_generated
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Maintains the high-water mark of allocated VRAM blocks via
    /// `fetch_max`. A subsequent call with a smaller value is a no-op,
    /// so the metric never lies about the worst case the engine has
    /// faced.
    pub fn update_peak_vram(&self, current_blocks: u64) {
        self.peak_vram_blocks
            .fetch_max(current_blocks, Ordering::Relaxed);
    }

    /// Average TTFT in milliseconds. Returns `0.0` when no token has
    /// been recorded yet so the caller can safely export the metric
    /// before the first inference completes — the alternative
    /// (panicking on division by zero) would be a footgun for any
    /// dashboard that polls during warm-up.
    pub fn get_avg_ttft_ms(&self) -> f64 {
        let tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        if tokens == 0 {
            return 0.0;
        }
        let cumulative_micros =
            self.cumulative_ttft_micros.load(Ordering::Relaxed) as f64;
        (cumulative_micros / tokens as f64) / 1_000.0
    }

    /// Snapshot accessors used by the PyO3 exporter. They live as
    /// methods rather than `pub` fields so the atomic load is the only
    /// way to reach the value — callers cannot accidentally take a
    /// reference into the AtomicU64 and reason about it as a plain
    /// integer.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens_generated.load(Ordering::Relaxed)
    }

    pub fn peak_vram(&self) -> u64 {
        self.peak_vram_blocks.load(Ordering::Relaxed)
    }
}

impl Default for NexusProfiler {
    fn default() -> Self {
        Self::new()
    }
}
