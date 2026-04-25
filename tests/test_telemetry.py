"""Empirical proof of the lock-free hardware profiler.

The profiler is meant to be poured into the inference hot path with no
mutex contention: every counter is a `AtomicU64` updated under
`Ordering::Relaxed`. We cannot directly measure the absence of locking
from Python, but we can pin the *contract* the lock-free design exists
to support — TTFT averaging, monotonic peak tracking, and a single
export that surfaces all three metrics together — so any regression
that silently re-introduces blocking would still have to keep these
invariants intact.
"""

from __future__ import annotations

import nexus_core


def test_hardware_profiler_metrics() -> None:
    profiler = nexus_core.PyNexusProfiler()

    # Three tokens generated, three TTFT samples, three VRAM updates.
    # The TTFT samples (45/55/50 ms) average to exactly 50 ms and the
    # peak VRAM (60) must survive a *lower* subsequent update because
    # `update_peak_vram` is implemented with `fetch_max`.
    profiler.record_token_generated()
    profiler.record_token_generated()
    profiler.record_token_generated()

    profiler.record_ttft(45_000)
    profiler.record_ttft(55_000)
    profiler.record_ttft(50_000)

    profiler.update_peak_vram(42)
    profiler.update_peak_vram(60)
    profiler.update_peak_vram(50)

    metrics = profiler.export_metrics()

    assert metrics["total_tokens"] == 3, (
        f"Token counter regression: expected 3, got {metrics['total_tokens']}"
    )
    assert metrics["avg_ttft_ms"] == 50.0, (
        f"TTFT averaging regression: expected 50.0 ms, got {metrics['avg_ttft_ms']}"
    )
    assert metrics["peak_vram_blocks"] == 60, (
        "Peak VRAM tracker is not monotonic: expected 60, got "
        f"{metrics['peak_vram_blocks']}"
    )
