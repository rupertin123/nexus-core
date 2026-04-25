"""Empirical proof of the Codata Substrate (`NexusChannel`).

Per the Nexus-Core architecture, agent contexts live as *active state* inside
a shared, lock-protected memory substrate rather than as serialized messages
flowing through a queue. This test pins the contract of the Rust
`NexusChannel<String>` exposed to Python as `PyNexusChannel`:

* ``push`` deposits a context token into the substrate.
* ``peek_context`` reads the most recent context **without consuming it**,
  so subsequent peeks observe the same state until a new ``push`` overwrites
  the head.
* Pushing past the configured capacity is a hard error surfaced to Python
  as a ``ValueError`` (mapped from the Rust ``Result::Err``).
"""

from __future__ import annotations

import pytest

import nexus_core


def test_nexus_channel_codata_behavior() -> None:
    channel = nexus_core.PyNexusChannel(10)

    channel.push("Agent A Initial Context")
    state = channel.peek_context()
    assert state == "Agent A Initial Context"

    # Peek must be non-destructive: a second peek observes the same state.
    assert channel.peek_context() == "Agent A Initial Context"

    # A subsequent push must update the head observed by peek_context.
    channel.push("Agent A Updated Context")
    assert channel.peek_context() == "Agent A Updated Context"

    # Saturate the remaining capacity (2 already pushed, 8 more = 10 total).
    for index in range(8):
        channel.push(f"context-token-{index}")

    # The eleventh push must surface as a Python exception.
    with pytest.raises((ValueError, RuntimeError)):
        channel.push("overflow-token")

    # The substrate must remain readable after the rejected push, and the
    # head must still be the last successfully pushed token.
    assert channel.peek_context() == "context-token-7"
