"""Empirical proof of the Cognitive Reliability Layer.

A traditional Python agent framework crashes the moment the LLM emits
malformed JSON or omits a required key — the failure escapes into the
caller's stack trace. Nexus-Core promises the opposite: every structural
contract violation is caught in the Rust core, converted into a
deterministic correction prompt, and fed back into a bounded retry
loop. The Python developer never sees a parse error; they see either a
valid JSON payload or a single `ValueError` indicating the request was
quarantined after exhausting the retry budget.

These two tests pin both branches of that contract empirically.
"""

from __future__ import annotations

import pytest

import nexus_core


def test_semantic_auto_correction_loop() -> None:
    validator = nexus_core.PySemanticValidator(["risk_score", "factors"])

    # Attempt 1 — structurally invalid JSON (unquoted key, dangling value).
    # The Rust interceptor must parse-fail here and synthesize a correction
    # prompt instead of letting the parse error reach Python.
    attempt1 = '{ risk_score: 0.9, factors: '

    # Attempt 2 — valid JSON, but missing the `factors` key. The schema
    # check must reject it and synthesize a key-specific correction prompt.
    attempt2 = '{"risk_score": 0.9, "reason": "high"}'

    # Attempt 3 — fully compliant. The loop must stop and return this.
    attempt3 = '{"risk_score": 0.9, "factors": ["firewall", "auth"]}'

    result = validator.validate_with_retry(
        [attempt1, attempt2, attempt3],
        max_retries=3,
    )

    assert result == attempt3, (
        "Auto-correction loop must surface the first valid attempt, "
        f"got: {result!r}"
    )


def test_quarantine_on_max_retries() -> None:
    validator = nexus_core.PySemanticValidator(["risk_score", "factors"])

    # Three structurally broken outputs in a row — the loop must exhaust
    # its budget and quarantine the request as a Python ValueError. A
    # silent fallthrough or a panic would both fail this assertion.
    broken_outputs = [
        '{ not even json',
        'still: not, json',
        '{"unclosed": ',
    ]

    with pytest.raises(ValueError):
        validator.validate_with_retry(broken_outputs, max_retries=3)
