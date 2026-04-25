---
base_model: google/gemma-4-8b-it
library_name: nexus-core
license: apache-2.0
pipeline_tag: text-generation
tags: [rust, bare-metal, edge-ai, mcp, zero-copy, gguf]
---

# Nexus-Core × Gemma 4 (8B-IT)

Nexus-Core wraps `google/gemma-4-8b-it` in a deterministic, Rust-backed
orchestrator designed for the Edge: PagedAttention with Copy-on-Write
prefix sharing, a Zero-Trust MCP gatekeeper, lock-free hardware
telemetry, and a continuous-batching scheduler that survives
oversubscribed workloads without OOM. The repository ships
pre-compiled wheels for Linux (x86_64 / aarch64), macOS (x86_64 /
aarch64), and Windows (x86_64); end users never touch a Rust toolchain.

## Recommended GGUF Quantizations for Nexus-Core

| Quantization | Use Case | VRAM (PagedAttention Est.) | Target Hardware |
| ------------ | -------- | -------------------------- | --------------- |
| `Q4_K_M`     | Balanced laptop / on-device assistant; best size-quality trade-off for interactive agents. | ~5.5 GB at 32k ctx, ~7 GB at 128k ctx with CoW prefix sharing. | Apple M-series (8–16 GB unified memory), NVIDIA RTX 4060 / 4070 mobile, ROCm 7900M. |
| `Q8_0`       | Server-side accuracy; near-FP16 fidelity for evaluation, distillation, or compliance-grade inference. | ~9 GB at 32k ctx, ~11 GB at 128k ctx. | NVIDIA RTX 4090 / 5090, A100 40 GB, H100 PCIe slice. |
| `AWQ`        | Pure GPU throughput; activation-aware 4-bit weights for high-QPS deployments behind the continuous-batching scheduler. | ~6 GB at 32k ctx with batched KV-cache reuse. | NVIDIA L4 / L40S, RTX 5080, Jetson AGX Orin 64 GB. |

## Contact & Community

Architectural feedback, open-source collaboration, and B2B / VC inquiries are all welcome. The fastest way to start a conversation is a direct message on either of the channels below.

- **Email:** [lucasaloisio6@gmail.com](mailto:lucasaloisio6@gmail.com)
- **LinkedIn:** [Lucas Aloisio](https://www.linkedin.com/in/lucas-aloisio-a17608240/)
