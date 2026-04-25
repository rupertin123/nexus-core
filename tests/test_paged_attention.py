"""Empirical proof of PagedAttention CoW branching.

The PagedAttention substrate is meant to eliminate the textbook failure mode
of contiguous KV-cache allocation: when N agents share the same fat system
prompt, naive engines copy the prompt's KV-cache N times. Nexus-Core
instead manages fixed-size physical blocks with reference counts, so that
"branching" a sequence (e.g., Tree-of-Thought spawning two children) bumps
ref-counts on the shared blocks rather than allocating fresh ones.

This test pins that contract empirically: branching twice from a 10-block
parent does NOT change the count of available physical blocks.
"""

from __future__ import annotations

import nexus_core


def test_copy_on_write_memory_savings() -> None:
    manager = nexus_core.PyKVCacheManager(100)
    assert manager.get_available_blocks() == 100

    # Agent A loads a heavy shared system prompt occupying 10 blocks.
    seq_a = manager.allocate_sequence(10)
    assert manager.get_available_blocks() == 90

    # Agents B and C branch from A (Tree-of-Thought style fan-out). Under
    # CoW, both new sequences share A's physical blocks via incremented
    # ref-counts; they MUST NOT consume any additional physical blocks.
    seq_b = manager.branch_sequence(seq_a)
    seq_c = manager.branch_sequence(seq_a)

    assert manager.get_available_blocks() == 90, (
        "Branching duplicated physical memory; CoW contract is broken."
    )

    # Sanity: each sequence got a distinct logical id.
    assert seq_a != seq_b
    assert seq_b != seq_c
    assert seq_a != seq_c
