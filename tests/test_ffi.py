"""FFI smoke test: the Rust PyO3 extension must import and return its banner."""

import nexus_core


def test_engine_status_returns_expected_banner() -> None:
    assert nexus_core.engine_status() == "Nexus-Core Bare-Metal Engine Initialized"
