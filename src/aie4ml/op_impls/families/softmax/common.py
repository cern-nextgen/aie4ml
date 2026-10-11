from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...utils.precision import storage_bytes_for_spec

DEFAULT_INV_SHIFT = 15
SOFTMAX_I8_VEC_SIZE = 32


def softmax_vec_size(precision, device) -> int:
    if precision.format == 'int8':
        return SOFTMAX_I8_VEC_SIZE
    elem_bytes = storage_bytes_for_spec(precision)
    return int(device.vector_bytes) // max(1, int(elem_bytes))


def infer_hccs_param_sets(hccs: Dict[str, Any]) -> int:
    explicit = hccs.get('param_sets')
    lengths = []
    for name in ('B', 'S', 'Dmax'):
        size = int(np.asarray(hccs[name]).reshape(-1).size)
        if size < 1:
            raise ValueError(f'HCCS Softmax parameter {name!r} must not be empty.')
        if size > 1:
            lengths.append(size)

    inferred = int(lengths[0]) if lengths else 1
    if any(length != inferred for length in lengths):
        raise ValueError(f'HCCS Softmax non-scalar parameter lengths must match, got {lengths}.')

    param_sets = int(explicit) if explicit is not None else inferred
    if param_sets < 1:
        raise ValueError(f'HCCS Softmax param_sets must be positive, got {param_sets}.')
    if inferred > 1 and int(param_sets) != int(inferred):
        raise ValueError(f'HCCS Softmax param_sets={param_sets} must match non-scalar parameter length {inferred}.')
    return param_sets


def _as_int_array(value: Any, *, name: str, lo: int, hi: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f'HCCS Softmax parameter {name!r} must not be empty.')
    if np.any((arr < int(lo)) | (arr > int(hi))):
        raise ValueError(f'HCCS Softmax parameter {name!r} must be in [{lo}, {hi}].')
    return arr


def _compact_hccs_param(values: np.ndarray, *, name: str, param_sets: int) -> np.ndarray:
    if values.size == 1:
        return np.full((int(param_sets),), int(values[0]), dtype=np.int64)
    if values.size == int(param_sets):
        return values.astype(np.int64, copy=True)
    raise ValueError(f'HCCS Softmax parameter {name!r} length must be 1 or param_sets={param_sets}; got {values.size}.')


def validate_hccs_params(hccs: Dict[str, Any], *, cols: int, param_sets: int) -> None:
    """Validate HCCS calibration constants against kernel ABI constraints.

    A row of `cols` gaps d = min(max - x, Dmax) scores B - S*d, from B at the row max down to B - S*Dmax, and Z is the
    row's score sum. The kernels need every score >= 0 (B >= S*Dmax; a zero floor is allowed) and
    256 <= Z <= 32767: Z >= 256 keeps the reciprocal 255 << inv_shift / Z within int16, and int16 holds Z. The
    smallest Z is one row max among floor scores, B + (cols - 1)*(B - S*Dmax); the largest is cols*B.
    """
    if int(param_sets) != 1:
        raise ValueError(f'HCCS Softmax compile-time scalar parameter packing requires param_sets=1, got {param_sets}.')
    b = _compact_hccs_param(
        _as_int_array(hccs['B'], name='B', lo=0, hi=32767),
        name='B',
        param_sets=param_sets,
    )
    s = _compact_hccs_param(
        _as_int_array(hccs['S'], name='S', lo=0, hi=127),
        name='S',
        param_sets=param_sets,
    )
    dmax = _compact_hccs_param(
        _as_int_array(hccs['Dmax'], name='Dmax', lo=0, hi=127),
        name='Dmax',
        param_sets=param_sets,
    )
    min_score = b - s * dmax
    if np.any(min_score < 0):
        bad = int(np.argmin(min_score))
        raise ValueError(f'HCCS Softmax requires B - S*Dmax >= 0 for every row; row {bad} gives {int(min_score[bad])}.')
    min_sum = b + (int(cols) - 1) * min_score
    if np.any(min_sum < 256):
        bad = int(np.argmin(min_sum))
        raise ValueError(
            f'HCCS Softmax requires B + (cols - 1) * (B - S*Dmax) >= 256 for reciprocal range; '
            f'row {bad} gives {int(min_sum[bad])}.'
        )
    if np.any(b * int(cols) > 32767):
        raise ValueError('HCCS Softmax requires cols * B <= 32767 so score sum fits int16 range.')


def pack_hccs_params(
    hccs: Dict[str, Any],
    *,
    param_sets: int,
    cols: int,
    cas_num: int,
) -> Dict[str, np.ndarray]:
    validate_hccs_params(hccs, cols=cols, param_sets=param_sets)
    b = _compact_hccs_param(_as_int_array(hccs['B'], name='B', lo=0, hi=32767), name='B', param_sets=param_sets)
    s = _compact_hccs_param(_as_int_array(hccs['S'], name='S', lo=0, hi=127), name='S', param_sets=param_sets)
    dmax = _compact_hccs_param(
        _as_int_array(hccs['Dmax'], name='Dmax', lo=0, hi=127), name='Dmax', param_sets=param_sets
    )
    packed_b = np.full((int(cas_num),), int(b[0]), dtype=np.int16)
    packed_s = np.full((int(cas_num),), int(s[0]), dtype=np.int8)
    packed_dmax = np.full((int(cas_num),), int(dmax[0]), dtype=np.uint8)
    return {'B': packed_b, 'S': packed_s, 'Dmax': packed_dmax}
