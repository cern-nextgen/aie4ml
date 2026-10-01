from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from ....aie_types import FLOAT_FORMATS, FloatIntent
from ....ir.graph import STAGING_CONTRACTS, input_tensor_for_role
from ....quant_utils import apply_rounding, dtype_for_precision, handle_overflow
from ...utils import (
    STORAGE_LAYOUT_LINEAR,
    AxisPlan,
    MicrotileShape,
    TensorView,
    build_staging_descriptor,
    canonical_buffer_axes,
    ordered_view_shape,
)
from ...utils.precision import element_bytes, resolve_exact_storage_dtype
from ...utils.tensor_view import microtile_from_staging

# Keys are canonical format-string pairs (lhs_format, rhs_format).
# Integer formats: 'int8', 'int16' (sign-agnostic — both int8_t and uint8_t map here).
# Float formats: 'bfloat16', 'float32', 'fp8_e4m3'.
MICROTILE_OPTIONS: Dict[str, Dict[Tuple[str, str], List[Tuple[int, int, int]]]] = {
    'AIE': {
        ('int8', 'int8'): [(2, 8, 8), (4, 8, 4), (1, 16, 8)],
        ('int16', 'int8'): [(4, 4, 4)],
        ('int8', 'int16'): [(4, 4, 8)],
        ('float32', 'float32'): [(2, 4, 4)],
    },
    'AIE-ML': {
        ('int8', 'int8'): [(4, 8, 8), (2, 8, 8), (2, 16, 8), (4, 8, 4), (4, 16, 4), (4, 16, 8), (8, 8, 4), (8, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (2, 8, 8), (4, 4, 4), (4, 8, 4), (8, 4, 4), (8, 4, 8)],
        ('int8', 'int16'): [(4, 4, 4), (4, 4, 8)],
        ('int16', 'int16'): [(4, 4, 4), (2, 4, 8), (4, 4, 8)],  # int16 x int16 accumulates in acc64
        ('bfloat16', 'bfloat16'): [(4, 8, 4)],
        ('float32', 'float32'): [(4, 8, 4)],
    },
    'AIE-MLV2': {
        ('int8', 'int8'): [(8, 8, 8), (4, 8, 8)],
        ('int16', 'int8'): [(4, 4, 8), (8, 2, 8), (4, 8, 8)],
        ('int8', 'int16'): [(4, 4, 8), (8, 2, 8)],
        ('int16', 'int16'): [(4, 4, 8), (2, 4, 8)],  # int16 x int16 accumulates in acc64
        ('bfloat16', 'bfloat16'): [(4, 8, 8)],
        ('float32', 'float32'): [(4, 8, 4)],
        ('fp8_e4m3', 'fp8_e4m3'): [(8, 8, 8)],
    },
}


def select_generation_key(generation: str) -> str:
    norm = (generation or '').strip().upper()
    if norm not in MICROTILE_OPTIONS:
        raise ValueError(f'Unknown AIE generation {generation!r}; expected one of {", ".join(MICROTILE_OPTIONS)}.')
    return norm


_SUPPORTED_INT_WIDTH_COMBOS = {
    'AIE': frozenset({(8, 8), (16, 8), (8, 16)}),
    'AIE-ML': frozenset({(8, 8), (16, 8), (16, 16)}),
    'AIE-MLV2': frozenset({(8, 8), (16, 8), (16, 16)}),
}


def bitwidths_supported(node, device) -> bool:
    """Whether this device can run the node's lhs/rhs storage widths."""
    generation = select_generation_key(device.generation)
    lhs = input_tensor_for_role(node, 'lhs')
    rhs = input_tensor_for_role(node, 'rhs')
    if isinstance(lhs.precision, FloatIntent):
        lhs_format = lhs.precision.format.value
        rhs_format = getattr(getattr(rhs, 'precision', None), 'format', None)
        rhs_format = getattr(rhs_format, 'value', rhs_format)
        return (lhs_format, rhs_format) in MICROTILE_OPTIONS[generation]
    lhs_p = resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name)
    rhs_p = resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name)
    return (lhs_p.width, rhs_p.width) in _SUPPORTED_INT_WIDTH_COMBOS[generation]


# Policy, not a law: the contract used when nothing asks for one. A global partition
# optimizer will write the directive instead, at which point this is only the fallback for
# graphs it did not run on.
DEFAULT_CONTRACT = 'inner'


def requested_contract(node, directives) -> str:
    """The parallelism contract the directives ask of this node; selects between the family's variants."""
    contract = str((directives.get('parallelism') or {}).get('contract', DEFAULT_CONTRACT))
    if contract not in STAGING_CONTRACTS:
        raise ValueError(
            f'{node.name}: unknown parallelism contract {contract!r}; expected one of {sorted(STAGING_CONTRACTS)}.'
        )
    return contract


def _slice(extent: int, size: int, share: int, index: int) -> Tuple[int, int]:
    """Where slice `index` of a partitioned axis starts in the tensor, and the extent its port moves from there.

    The ports cut the axis, padded as a whole, into slices of `size`, as the weights and bias are packed: a slice
    starts where its buffer does, and only the last ones reach past the tensor. A port moves the tensor's elements
    from its start -- at most a slice, and at least its `share`, the word-aligned extent transfers are sized by.
    """
    start = int(index) * int(size)
    return start, min(int(size), max(int(share), int(extent) - start))


def describe_inner_lhs_staging(view: TensorView, port: int):
    """LHS staging for the 'inner' contract: the port selects a K-chain; the rows stay whole."""
    microtile_m = int(view.microtile.outer)
    microtile_k = int(view.microtile.inner)
    in_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)
    k_start, k_held = _slice(view.logical_inner, in_slice, view.tile_raw_inner, port)
    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_k, microtile_k, in_slice // microtile_k, port * in_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=list(range(view.rank)),
        logical_origin={inner_dim: k_start},
        io_tiling_overrides={inner_dim: k_held},
        boundary_shape='logical',
    )


def describe_inner_output_staging(view: TensorView, port: int):
    """Output staging for the 'inner' contract: the port selects an N-slice; the rows stay whole."""
    microtile_m = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    out_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    n_start, n_held = _slice(view.logical_inner, out_slice, view.tile_raw_inner, port)
    return build_staging_descriptor(
        view,
        access='write',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, out_slice // microtile_n, port * out_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m),
        },
        order=traversal_dims,
        logical_origin={inner_dim: n_start},
        io_tiling_overrides={inner_dim: n_held},
    )


def describe_outer_lhs_staging(view: TensorView, parallelism, port: int):
    """LHS staging for the 'outer' contract: the port selects a (row-group, K-chain) tile."""
    microtile_m = int(view.microtile.outer)
    microtile_k = int(view.microtile.inner)
    in_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    cas_length = max(1, int(parallelism.cas_length))
    row_group = int(port) // cas_length
    k_chain = int(port) % cas_length
    k_start, k_held = _slice(view.logical_inner, in_slice, view.tile_raw_inner, k_chain)
    row_start, rows_held = _slice(view.logical_outer, outer_slice, view.tile_raw_outer, row_group)

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_k, microtile_k, in_slice // microtile_k, k_chain * in_slice),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m, row_group * outer_slice),
        },
        logical_origin={inner_dim: k_start, outer_dim: row_start},
        order=list(range(view.rank)),
        io_tiling_overrides={inner_dim: k_held, outer_dim: rows_held},
        slice_dim=outer_dim,
        boundary_shape='logical',
    )


def describe_outer_output_staging(view: TensorView, port: int):
    """Output staging for the 'outer' contract: the port selects a row-group; N stays whole."""
    microtile_m = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    out_slice = view.tile_inner
    outer_slice = view.tile_outer
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    row_start, rows_held = _slice(view.logical_outer, outer_slice, view.tile_raw_outer, port)
    return build_staging_descriptor(
        view,
        access='write',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, out_slice // microtile_n),
            outer_dim: AxisPlan(microtile_m, microtile_m, outer_slice // microtile_m, int(port) * outer_slice),
        },
        logical_origin={outer_dim: row_start},
        order=traversal_dims,
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: rows_held},
        slice_dim=outer_dim,
    )


def describe_stream_staging(view: TensorView, port: int, access: str, contract: str, cas_length: int = 1):
    """Staging of one stream port: the whole padded per-port tile, row by row.

    A stream has no DMA to re-tile or de-pad, so this is exactly the element sequence the
    kernel reads or writes: the padded slice of every partitioned axis (`tile`), every row of
    the rest. The kernel still computes in microtiles, but that never reaches the wire, so the
    layout is linear and a consumer inherits no microtile from it.
    """
    inner_dim, outer_dim, traversal_dims = canonical_buffer_axes(view)
    full = ordered_view_shape(view, 'full')
    in_slice, outer_slice = view.tile_inner, view.tile_outer

    if contract == 'outer' and access == 'read':
        row_group, k_chain = divmod(int(port), max(1, int(cas_length)))
    elif contract == 'outer':
        row_group, k_chain = int(port), 0
    else:
        row_group, k_chain = 0, int(port)

    plans = {dim: AxisPlan(int(full[dim]), int(full[dim]), 1) for dim in traversal_dims}
    plans[inner_dim] = AxisPlan(in_slice, in_slice, 1, k_chain * in_slice)
    plans[outer_dim] = AxisPlan(outer_slice, outer_slice, 1, row_group * outer_slice)
    k_start, k_held = _slice(view.logical_inner, in_slice, view.tile_raw_inner, k_chain)
    row_start, rows_held = _slice(view.logical_outer, outer_slice, view.tile_raw_outer, row_group)
    return build_staging_descriptor(
        view,
        access=access,
        plans=plans,
        order=traversal_dims,
        logical_origin={inner_dim: k_start, outer_dim: row_start},
        io_tiling_overrides={inner_dim: k_held, outer_dim: rows_held},
        slice_dim=outer_dim if contract == 'outer' else inner_dim,
        boundary_shape='logical' if access == 'read' else None,
        extras={'storage_layout': STORAGE_LAYOUT_LINEAR},
    )


def adopts_blocks(microtile: MicrotileShape, transposed: bool, producer: MicrotileShape, element_size: int) -> bool:
    """Whether an operand read in `microtile`s of its view can read the blocks its producer stores: they hold the
    columns of one microtile of the stored tensor (the view, or its transpose) and divide its rows, each at least
    one 16-byte vector load."""
    rows, cols = (microtile.inner, microtile.outer) if transposed else (microtile.outer, microtile.inner)
    return (
        int(producer.inner) == int(cols)
        and int(rows) % int(producer.outer) == 0
        and int(producer.outer) * int(cols) * int(element_size) % 16 == 0
    )


def stored_block(microtile: MicrotileShape, transposed: bool, producer: MicrotileShape | None, element_size: int):
    """The blocks a kernel reads an mmul operand in, in its view's axes (see stored_operand.h): the producer's where
    it can read them, so the hand-over needs no re-layout, else the microtile's own."""
    if producer is None or not adopts_blocks(microtile, transposed, producer, element_size):
        return microtile
    if transposed:
        return MicrotileShape(outer=int(microtile.outer), inner=int(producer.outer))
    return MicrotileShape(outer=int(producer.outer), inner=int(microtile.inner))


def stored_block_rows(view: TensorView) -> int:
    """Rows of the stored tensor in one block of an operand view (stored_operand.h's BLOCK_ROWS)."""
    return int(view.microtile.inner if view.is_transposed else view.microtile.outer)


def producer_blocks(tensor, input_contracts) -> MicrotileShape | None:
    """The microtile blocks the producer of `tensor` stores it in, or None (no producer, or not microtiled)."""
    contract = input_contracts.get(tensor.name)
    return None if contract is None else microtile_from_staging(contract.port_staging[0])


def read_as_stored(node, io_views: Dict[str, TensorView], input_contracts, precision) -> Dict[str, TensorView]:
    """`io_views` with each activation operand read in its producer's blocks where it can be (see stored_block)."""
    views = dict(io_views)
    for role in ('lhs', 'rhs'):
        tensor = input_tensor_for_role(node, role)
        if tensor.is_parameter:
            continue
        producer = producer_blocks(tensor, input_contracts)
        view = views[tensor.name]
        block = stored_block(view.microtile, view.is_transposed, producer, element_bytes(precision[role]))
        views[tensor.name] = dataclasses.replace(view, microtile=block)
    return views


def describe_outer_rhs_staging(view: TensorView, parallelism, port: int):
    """RHS staging for the 'outer' contract: the port selects a K-chain; every row group shares it. Its blocks are in
    the tensor's own order, transposed or not (see `stored_block`)."""
    microtile_k = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    k_slice = view.tile_outer
    n_slice = view.tile_inner
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    k_chain = int(port) % max(1, int(parallelism.cas_length))
    k_start, k_held = _slice(view.logical_outer, k_slice, view.tile_raw_outer, k_chain)

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, max(1, n_slice // microtile_n)),
            outer_dim: AxisPlan(microtile_k, microtile_k, max(1, k_slice // microtile_k), k_chain * k_slice),
        },
        logical_origin={outer_dim: k_start},
        order=list(range(view.rank)),
        io_tiling_overrides={inner_dim: view.tile_raw_inner, outer_dim: k_held},
        boundary_shape='logical',
    )


def describe_inner_rhs_staging(view: TensorView, parallelism, port: int):
    """RHS staging for the 'inner' contract: the port selects an (N-slice, K-chain) tile, its blocks in the tensor's
    own order (see `stored_block`)."""
    microtile_k = int(view.microtile.outer)
    microtile_n = int(view.microtile.inner)
    # The rhs view encodes both K and N slices: outer dim = K, inner dim = N.
    k_slice = view.tile_outer
    n_slice = view.tile_inner
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    row = int(port) // int(parallelism.cas_length)
    col = int(port) % int(parallelism.cas_length)
    n_start, n_held = _slice(view.logical_inner, n_slice, view.tile_raw_inner, row)
    k_start, k_held = _slice(view.logical_outer, k_slice, view.tile_raw_outer, col)

    return build_staging_descriptor(
        view,
        access='read',
        plans={
            inner_dim: AxisPlan(microtile_n, microtile_n, max(1, n_slice // microtile_n), row * n_slice),
            outer_dim: AxisPlan(microtile_k, microtile_k, max(1, k_slice // microtile_k), col * k_slice),
        },
        logical_origin={inner_dim: n_start, outer_dim: k_start},
        order=list(range(view.rank)),
        io_tiling_overrides={inner_dim: n_held, outer_dim: k_held},
        boundary_shape='logical',
    )


def np_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt == 'bfloat16':
        return np.uint16
    if fmt in ('float32', 'accfloat'):
        return np.float32
    if fmt == 'fp8_e4m3':
        return np.uint8
    if fmt.startswith('uint'):
        return np.uint8 if int(spec.width) <= 8 else np.uint16
    return np.int8 if int(spec.width) <= 8 else np.int16


def np_bias_dtype_for_spec(spec) -> np.dtype:
    fmt = getattr(spec, 'format', '') or ''
    if fmt in FLOAT_FORMATS:
        return np.float32
    return np.int16 if int(spec.width) <= 16 else np.int32


def pack_as_float(array: np.ndarray, fmt) -> np.ndarray:
    """Cast weight/bias data to the float storage format required by mmul kernels."""
    if array is None:
        return None
    from ....aie_types import FloatFormat

    fmt_value = getattr(fmt, 'value', fmt)

    def _float32_to_fp8_scalar(f: np.float32) -> np.uint8:
        h = int(np.float32(f).view(np.uint32))
        h = (h + 0x00080000) & 0xFFFFFFFF
        e = (h & 0x7F800000) >> 23
        m = h & 0x007FFFFF
        sign = (h & 0x80000000) >> 24
        if e > 135:
            result = sign | 0x7F
        elif e > 120:
            result = sign | (((e - 120) << 3) & 0x78) | (m >> 20)
        elif e > 116:
            result = sign | (((0x00780000 + m) >> (140 - e)) + 1) >> 1
        else:
            result = sign
        return np.uint8(result & 0xFF)

    if fmt_value == FloatFormat.BF16.value:
        f32 = np.asarray(array, dtype=np.float32)
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    if fmt_value == FloatFormat.FP8_E4M3.value:
        vfloat32_to_fp8 = np.vectorize(_float32_to_fp8_scalar, otypes=[np.uint8])
        f32 = np.asarray(array, dtype=np.float32)
        return np.asarray(vfloat32_to_fp8(f32), dtype=np.uint8)
    return np.asarray(array, dtype=np.float32)


def quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    """Quantize float weight/bias data to fixed-point integers for mmul kernels."""
    if array is None:
        return None
    scaled = np.ldexp(np.asarray(array, dtype=np.float64), int(frac_bits))
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)


def pack_mmul_rhs_matrix(
    W,
    *,
    K: int,
    N: int,
    K_slice: int,
    N_slice: int,
    microtile_k: int,
    microtile_n: int,
    cas_length: int,
    cas_num: int,
    order: str = 'C',
    dtype=None,
):
    assert microtile_k > 0 and microtile_n > 0
    assert K_slice % microtile_k == 0
    assert N_slice % microtile_n == 0

    W = np.asarray(W)
    if dtype is not None:
        W = W.astype(dtype, copy=False)
    if W.ndim < 2:
        raise ValueError('W must have at least 2 dimensions')
    W_kn = W.reshape((-1, K, N))[-1]

    tiles_per_k = K_slice // microtile_k
    tiles_per_n = N_slice // microtile_n
    elements_per_tile = microtile_k * microtile_n
    flat_len = tiles_per_k * tiles_per_n * elements_per_tile

    packed = np.zeros((cas_num, cas_length, flat_len), dtype=W_kn.dtype)
    tile_buf = np.zeros((microtile_k, microtile_n), dtype=W_kn.dtype)

    for chain in range(cas_num):
        n_base = chain * N_slice
        for cas in range(cas_length):
            flat = packed[chain, cas]
            tile_idx = 0
            for k_tile in range(tiles_per_k):
                gk = cas * K_slice + k_tile * microtile_k
                real_k = max(0, min(microtile_k, K - gk))
                for n_tile in range(tiles_per_n):
                    tile_buf.fill(0)
                    gn = n_base + n_tile * microtile_n
                    real_n = max(0, min(microtile_n, N - gn))
                    if real_k > 0 and real_n > 0:
                        tile_buf[:real_k, :real_n] = W_kn[gk : gk + real_k, gn : gn + real_n]
                    start = tile_idx * elements_per_tile
                    flat[start : start + elements_per_tile] = tile_buf.ravel(order=order)
                    tile_idx += 1

    return packed


def pack_vector_by_n_slice(
    v,
    *,
    N: int,
    N_slice: int,
    cas_num: int,
    dtype=None,
):
    v = np.asarray(v)
    if dtype is not None:
        v = v.astype(dtype, copy=False)
    if v.ndim > 1:
        v = v.reshape((-1,))[:N]
    if v.shape[0] != N:
        raise ValueError(f'Vector length mismatch: got {v.shape[0]}, expected {N}')

    packed = np.zeros((cas_num, N_slice), dtype=v.dtype)
    for chain in range(cas_num):
        n_base = chain * N_slice
        real = max(0, min(N_slice, N - n_base))
        if real > 0:
            packed[chain, :real] = v[n_base : n_base + real]
    return packed
