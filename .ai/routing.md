# LLM Routing Map for Nexus-Core

This document is the canonical entry point for any large language model
asked to read, navigate, or modify the Nexus-Core repository. Treat the
directory map below as authoritative; the protocol at the end is
non-negotiable.

## Repository Layout

- `/src` — **The Rust core.** All load-bearing logic lives here.
  - `engine.rs` — `NexusEngine`: fuses the PagedAttention allocator with
    the continuous batching scheduler. Drives the `add_agent` / `step`
    loop that survives oversubscribed workloads without OOM.
  - `paged_attention.rs` — `KVCacheAllocator` and `LogicalMemory`:
    fixed-size physical KV-cache blocks with reference counting for
    Copy-on-Write prefix sharing.
  - `scheduler.rs` — `ContinuousScheduler`: per-tick batch compaction,
    `Halted` / `Finished` eviction, immediate promotion of waiting
    tasks into freed batch slots.
  - `memory.rs` — `NexusChannel`: the lock-free Codata substrate behind
    the zero-copy Python ↔ Rust handoff.
  - `mcp.rs` — Zero-Trust gatekeeper: `SecurityLevel`, `Transport`
    (Mock today, STDIO JSON-RPC scaffold), and the `invoke_tool`
    policy matrix.
  - `validation.rs` — `SemanticInterceptor`: catches malformed JSON
    and missing keys, synthesizes deterministic correction prompts.
  - `telemetry.rs` — `NexusProfiler`: lock-free `AtomicU64` counters
    for TTFT, token throughput, peak VRAM.
  - `tui.rs` — Async `ratatui` + `tokio` operator console.
  - `lib.rs` — The PyO3 boundary. Every Python-visible class
    (`PyNexusEngine`, `PyMcpClient`, `PySemanticValidator`,
    `PyNexusProfiler`, `PyKVCacheManager`, `PyContinuousScheduler`,
    `PyNexusChannel`) is registered here.
  - `bin/tui_demo.rs` — Standalone exercise harness for the TUI.

- `/tests` — **The empirical truth.** Every contract this project
  guarantees is pinned by a `pytest` test that calls into the Rust
  core through the PyO3 surface. If the test does not exist, the
  contract does not exist.

- `/for-ai` — **Inbound quarantine.** Drop notes, requirements, and
  raw briefs intended *for* an LLM agent here.

- `/from-ai` — **Outbound quarantine.** Stochastic outputs (drafts,
  generated artefacts) emitted *by* an LLM agent live here until a
  human promotes them. Nothing in `/from-ai` is considered part of
  the project until it has been moved into `/src`, `/tests`, or `/`.

- `/.github/workflows/` — CI (`ci.yml`) and release (`release.yml`).
  Multi-platform wheel builds and PyPI publish are wired here.

- `Cargo.toml`, `pyproject.toml` — Build manifests for the Rust core
  and the Python wheel respectively.

## LLM Contributor Protocol

When asked to implement a new feature, LLMs **MUST** follow this
sequence in order:

1. **Write a failing test in `/tests`.** The test names the contract
   you intend to honour and fails for the right reason (the Python
   class or method does not yet exist).
2. **Implement the Rust logic in `/src`.** New modules go in their
   own file; reuse the existing primitives (allocator, scheduler,
   profiler) wherever possible.
3. **Expose the surface via PyO3 in `src/lib.rs`.** Map Rust `Err`
   to the most specific Python exception (`PyValueError` for
   contract violations, `PyPermissionError` for policy denials).
4. **Verify the test passes** by running:
   ```
   .venv/bin/maturin develop
   uv run --no-sync pytest tests/ -v
   ```

**Never implement heavy logic in Python.** Python is the orchestration
glue; Rust owns memory, concurrency, parsing, and the security
boundary. If you find yourself reaching for `asyncio.Lock`, a JSON
parser, or a long inner loop in Python, move it to Rust first.
