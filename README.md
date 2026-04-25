# Nexus-Core: Bare-Metal Agentic Orchestrator

[![CI](https://github.com/nexus-core/nexus-core/actions/workflows/ci.yml/badge.svg)](https://github.com/nexus-core/nexus-core/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

> A deterministic, Rust-backed orchestration engine for the Gemma 4 family, bypassing the Python GIL for zero-copy memory execution at the Edge.

---

## Time-to-Value < 60 seconds

Pre-compiled binary wheels are published for Linux (x86_64 / aarch64), macOS (x86_64 / aarch64), and Windows (x86_64). **You never compile Rust locally.**

```bash
uv venv
uv pip install nexus-core
```

```python
import nexus_core

# 1. Spin up the bare-metal engine: 50 KV-cache blocks, 10 concurrent agents.
engine = nexus_core.PyNexusEngine(max_blocks=50, max_batch_size=10)
engine.add_agent(required_blocks=5)

# 2. The Zero-Trust gatekeeper polices every tool call before any transport runs.
mcp = nexus_core.PyMcpClient("require_approval")
print(mcp.invoke_tool("read_logs", "{}"))   # -> "Simulated execution of read_logs"
```

That is the entire surface area you need to put a deterministic agent loop in production.

---

## Core Architecture

- **PagedAttention with Copy-on-Write** — Fixed-size physical KV-cache blocks managed by a reference-counted allocator. Branching a sequence (Tree-of-Thought fan-out, speculative decoding) bumps a ref-count instead of duplicating the prompt's KV tensors. **Memory cost grows with unique tokens, not with branches.**
- **ZeroIPC / Codata Substrate** — A lock-free, Arc-shared channel that hands Python and Rust the *same* memory buffer. NumPy arrays mutate in place under `Python::allow_threads`, so a long Rust compute path runs while the GIL is released. **Sub-millisecond FFI context switching, zero copies.**
- **Zero-Trust MCP Client** — Every Model Context Protocol egress is gated in Rust *before* the STDIO transport is touched. Three policy levels (`allow_all`, `require_approval`, `deny_all`) plus a destructive-prefix list (`write_`, `delete_`, `drop_`, `update_`) intercept dangerous tools and surface them as `PermissionError` carrying a `SECURITY_INTERCEPT` sentinel. **Air-gapped tool execution that survives a malicious or hallucinated tool name.**
- **Continuous Batching Scheduler** — Per-token-tick eviction of `Halted` and `Finished` agents with immediate promotion of waiting tasks into the freed slots. The batch stays saturated even when individual agents stall on tool round-trips.
- **Cognitive Reliability Layer** — Malformed JSON and missing-key responses never reach Python. Rust intercepts them, synthesizes a deterministic correction prompt, and retries inside a bounded loop until the LLM emits a contract-compliant payload or the request is quarantined.
- **Lock-Free Hardware Profiler** — TTFT, token throughput, and peak VRAM tracked through `AtomicU64` with `Ordering::Relaxed`. **No mutex on the inference hot path, ever.**
- **Asynchronous TUI** — A `ratatui` + `tokio` operator console that multiplexes the streaming token output and a live telemetry pane without blocking the main event loop.

---

## Target Use Cases

- **Disconnected Edge AI nodes.** Air-gapped industrial sites, field robotics, ruggedised inference appliances. The Rust core has no Python runtime dependency on the hot path; the Zero-Trust gatekeeper assumes the network is hostile by default.
- **AgTech IoT integration.** Greenhouse controllers, autonomous spraying rigs, livestock telemetry gateways. The continuous-batching scheduler keeps the GPU saturated even when sensor I/O is slow and bursty, and the lock-free profiler exposes per-tick metrics to a SCADA or MQTT bridge.
- **Hardware-constrained local deployments.** On-device assistants on consumer GPUs, NVIDIA Jetson / AGX Orin boards, ARM-class SBCs. PagedAttention CoW lets a single shared system prompt fan out across many concurrent reasoning branches without multiplying VRAM consumption.

---

## License

Apache 2.0 for the core. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the Open-Core split and the PR contract.
