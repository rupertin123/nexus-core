"""Empirical proof of token-iteration-level continuous batching.

Naive batching wastes compute when one slot in the batch finishes (or halts
to wait on an MCP tool call) before its peers: the GPU spins on padding
until the longest slot is done. Continuous Batching evicts the slow/halted
slot at the next token tick and immediately injects a queued task into the
freed seat, keeping the batch saturated.

This test pins that contract: with `max_batch_size = 2` and three queued
tasks, the second `step()` after halting task 0 must surface tasks 1 and 2
together — proving the scheduler compacted the batch in a single tick
without waiting for task 1 to finish.
"""

from __future__ import annotations

import nexus_core


def test_continuous_batching_compaction() -> None:
    scheduler = nexus_core.PyContinuousScheduler(2)

    t0 = scheduler.add_task()
    t1 = scheduler.add_task()
    t2 = scheduler.add_task()
    assert (t0, t1, t2) == (0, 1, 2)

    # First tick: batch is empty, two waiting tasks promote in. Task 2 stays
    # in the waiting queue because the batch is now full.
    active = scheduler.step()
    assert sorted(active) == [0, 1]

    # Task 0 calls an MCP tool and halts. The next tick must NOT wait for
    # task 1 to finish; it must compact the batch and admit task 2 in the
    # very same step.
    scheduler.mark_halted(0)

    active = scheduler.step()
    assert sorted(active) == [1, 2], (
        f"Continuous batching failed to compact: expected [1, 2], got {active}"
    )
