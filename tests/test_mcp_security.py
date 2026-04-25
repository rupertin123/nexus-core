"""Empirical proof of the Zero-Trust MCP gatekeeper.

The MCP client is the single authorized egress for an agent into the
outside world. Nexus-Core's contract is that this egress is policed in
the Rust core, not in Python: the security policy is evaluated *before*
any transport (mock today, STDIO JSON-RPC later) is touched. Every
denial must surface in Python as a `PermissionError` whose message
carries the `SECURITY_INTERCEPT` sentinel, so callers can distinguish
a policy block from a transport failure or an upstream tool error.

These tests pin all three branches of the policy matrix:
    * `require_approval` lets a read-only tool through.
    * `require_approval` blocks a destructive prefix.
    * `deny_all` blocks everything regardless of name.
"""

from __future__ import annotations

import pytest

import nexus_core


def test_mcp_zero_trust_gatekeeper() -> None:
    client = nexus_core.PyMcpClient("require_approval")

    # Read-only tools clear the require_approval policy and must reach
    # the (mocked) transport, returning the simulated execution string.
    result = client.invoke_tool("read_logs", "{}")
    assert result == "Simulated execution of read_logs", (
        f"Expected mocked transport response, got: {result!r}"
    )

    # Destructive prefixes are intercepted by the gatekeeper before the
    # transport is invoked. The denial must arrive as a PermissionError
    # carrying the SECURITY_INTERCEPT sentinel so callers can branch
    # on policy denials vs. transport faults.
    with pytest.raises(PermissionError) as excinfo:
        client.invoke_tool("drop_table_users", "{}")

    assert "SECURITY_INTERCEPT" in str(excinfo.value), (
        f"Denial message missing SECURITY_INTERCEPT sentinel: {excinfo.value!s}"
    )

    # Even an obviously safe tool must be rejected when the policy is
    # the most restrictive level. This proves deny_all is evaluated
    # ahead of the per-tool prefix inspection.
    locked_client = nexus_core.PyMcpClient("deny_all")
    with pytest.raises(PermissionError):
        locked_client.invoke_tool("read_logs", "{}")
