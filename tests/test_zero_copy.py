"""Empirical proof that Rust mutates a NumPy buffer in-place across the FFI
boundary, with the GIL released during the heavy compute path.

The contract under test is the foundation of Phase 1.1: Python owns the
allocation (a 1-D float64 buffer simulating a flattened KV-cache tensor),
hands it to Rust via the buffer protocol, and Rust performs an in-place
multiply-by-two while explicitly releasing the GIL through
``Python::allow_threads``. No copy, no return value, no GIL contention.
"""

from __future__ import annotations

import time

import numpy as np

import nexus_core


def test_zero_copy_mutation() -> None:
    tensor = np.ones(1_000_000, dtype=np.float64)
    original_data_ptr = tensor.ctypes.data

    start = time.perf_counter()
    result = nexus_core.mutate_tensor_in_place(tensor)
    elapsed = time.perf_counter() - start

    print(
        f"\n[zero-copy] mutate_tensor_in_place on 1,000,000 f64 elements: "
        f"{elapsed * 1_000:.4f} ms"
    )

    assert result is None, "Mutation must be in-place; no new object returned."
    assert tensor.ctypes.data == original_data_ptr, (
        "Underlying buffer pointer changed: a copy was made."
    )
    assert tensor.dtype == np.float64
    assert tensor.shape == (1_000_000,)
    assert np.all(tensor == 2.0), "Every element must have been doubled in place."
