from __future__ import annotations

from ...utils.precision import storage_bytes_for_spec


def elementwise_vec_size(lhs_precision, device) -> int:
    """The lane count of elementwise kernels: the elements of one vector register (`device.vector_bytes`)."""
    lhs_bytes = storage_bytes_for_spec(lhs_precision)
    return int(device.vector_bytes) // max(1, int(lhs_bytes))
