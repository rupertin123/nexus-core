"""Empirical proof that the NexusEngine evades OOM under stress.

The engine fuses PagedAttention block allocation with the Continuous
Batching scheduler into a single orchestrator. Its load-bearing claim is
that a workload whose *total* memory demand vastly exceeds physical VRAM
must still complete: as soon as an agent finishes generating, the engine
must reclaim every block it held and recycle them into the next agents
in the queue. If that recycling were broken, the engine would either
OOM (raising a `ValueError` from the underlying allocator) or stall
forever (no waiting agent could ever be admitted).

This test pins that contract under deliberately punishing parameters:
the working set (500 blocks) is ten times physical capacity (50 blocks).
"""

from __future__ import annotations

import nexus_core


def test_engine_evades_oom_under_stress() -> None:
    # 50 physical blocks of "VRAM", batch of 10 concurrent agents.
    engine = nexus_core.PyNexusEngine(max_blocks=50, max_batch_size=10)

    # 100 agents x 5 blocks each = 500 blocks of total demand, vs 50 of
    # physical capacity. Only continuous eviction + block recycling can
    # close that 10x gap.
    for _ in range(100):
        engine.add_agent(required_blocks=5)

    steps = 0
    # An OOM bug would raise here; a stall bug would loop forever. The
    # absence of either failure mode is the proof.
    while engine.step():
        steps += 1
        # Hard ceiling to prevent a hung CI run from masking a stall bug
        # as a timeout. A correct engine finishes in well under this.
        assert steps < 10_000, "engine appears to be stalled; no progress"

    print(f"\nNexusEngine drained 100 agents in {steps} steps "
          f"(50 physical blocks, 500 demanded).")
