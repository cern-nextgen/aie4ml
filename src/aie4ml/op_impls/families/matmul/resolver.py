from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ....aie_types import AIEDataType, legality_format
from ....errors import ConfigRefused
from ....ir import input_role, input_tensor_for_role
from ....ir.graph import STAGING_CONTRACTS
from ...family_registry import FamilyResolver, family_resolver
from ...utils import MicrotileShape, TensorView, align_up, build_tensor_view, ceildiv
from ...utils.io import view_shape
from ...utils.precision import element_bytes, resolve_operand_precision
from .common import MICROTILE_OPTIONS, select_generation_key
from .config import MatmulMicrotileConfig


@dataclass(frozen=True)
class MatmulTiling:
    """Resolved grid + per-tile extents.

    `contract` names which axis `cas_num` partitions
      'inner'  cas_num splits the output features N; every tile sees the full outer (M) extent.
      'outer'  cas_num splits the outer (M) rows; every tile sees the full N.
    `cas_length` splits the reduction K under both.
    """

    cas_num: int
    cas_length: int
    tile_inner_lhs_raw: int
    tile_inner_lhs: int
    tile_inner_rhs_raw: int
    tile_inner_rhs: int
    contract: str = 'inner'
    tile_outer: int = 0
    tile_outer_raw: int = 0


def _bank_capacity_bytes(device: Any) -> int:
    return max(1, int(getattr(device, 'bank_mem_bytes', 0) or 1))


_MATMUL_RHS_STACK_BYTES = 1024


def _tile_bank_usage(
    *,
    op_type: str,
    device: Any,
    outer_extent: int,
    tile_inner_lhs: int,
    tile_inner_rhs: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
) -> Dict[str, int]:
    # Contract-agnostic: everything is expressed in per-tile extents, so 'inner'
    # (outer_extent=M, tile_inner_rhs=N/cas_num) and 'outer' (outer_extent=M/cas_num,
    # tile_inner_rhs=N) both fall out of the same formula.
    lhs_tile_bytes = int(outer_extent) * int(tile_inner_lhs) * max(1, int(lhs_bytes))
    rhs_tile_bytes = int(tile_inner_lhs) * int(tile_inner_rhs) * max(1, int(rhs_bytes))
    rhs_tile_bytes += _MATMUL_RHS_STACK_BYTES if op_type == 'matmul' else 0
    output_tile_bytes = int(outer_extent) * int(tile_inner_rhs) * max(1, int(output_bytes))
    return {
        'lhs_tile_bytes': lhs_tile_bytes,
        'rhs_tile_bytes': rhs_tile_bytes,
        'output_tile_bytes': output_tile_bytes,
        'max_bank_tile_bytes': max(lhs_tile_bytes, rhs_tile_bytes, output_tile_bytes),
        'bank_capacity_bytes': _bank_capacity_bytes(device),
    }


def _supported_microtile_options(generation: str, lhs_dtype, rhs_dtype):
    key = (legality_format(lhs_dtype.format), legality_format(rhs_dtype.format))
    return list(MICROTILE_OPTIONS.get(select_generation_key(generation), {}).get(key, []))


def _resolve_tile_cfg(node, device, lhs_dtype, rhs_dtype, required_lhs_microtile=None) -> MatmulMicrotileConfig:
    microtiling_cfg = node.directives.get('microtiling')
    if microtiling_cfg is not None and len(microtiling_cfg) != 3:
        raise ValueError(f'{node.name}: microtiling needs microtile_m, microtile_k and microtile_n.')
    options = _supported_microtile_options(device.generation, lhs_dtype, rhs_dtype)
    if not options:
        raise ValueError(
            f'{node.name}: no supported tile configs are registered for Generation={device.generation} and '
            f'(input={lhs_dtype.format!r}, weight={rhs_dtype.format!r}).'
        )

    if microtiling_cfg is not None:
        candidate = tuple(microtiling_cfg[key] for key in ('microtile_m', 'microtile_k', 'microtile_n'))
        if candidate not in options:
            raise ValueError(
                f'{node.name}: microtiling {candidate} not supported for Generation={device.generation} and '
                f'(input={lhs_dtype.format!r}, weight={rhs_dtype.format!r}). Allowed: {options}'
            )
        if required_lhs_microtile is not None and candidate[:2] != (
            int(required_lhs_microtile.outer),
            int(required_lhs_microtile.inner),
        ):
            raise ConfigRefused(
                f'{node.name}: microtiling {candidate} does not match the producer output microtile '
                f'({required_lhs_microtile.outer}, {required_lhs_microtile.inner}).'
            )
        return MatmulMicrotileConfig(microtile_m=candidate[0], microtile_k=candidate[1], microtile_n=candidate[2])

    if required_lhs_microtile is not None:
        required = (int(required_lhs_microtile.outer), int(required_lhs_microtile.inner))
        options = [option for option in options if option[:2] == required]
        if not options:
            raise ConfigRefused(
                f'{node.name}: no supported microtiling accepts producer output microtile {required} for '
                f'Generation={device.generation}.'
            )

    default_m, default_k, default_n = options[0]
    return MatmulMicrotileConfig(microtile_m=default_m, microtile_k=default_k, microtile_n=default_n)


def _parallelism_candidate(
    *,
    op_type: str,
    device,
    in_shape: int,
    out_shape: int,
    lhs_align: int,
    rhs_align: int,
    lhs_bytes: int,
    rhs_bytes: int,
    output_bytes: int,
    full_outer: int,
    cas_num: int,
    cas_length: int,
    outer_granularity: int,
    contract: str = 'inner',
) -> Optional[MatmulTiling]:
    row_wise = contract == 'outer'
    tile_inner_rhs_raw = out_shape if row_wise else ((out_shape + cas_num - 1) // cas_num if cas_num else out_shape)
    tile_inner_lhs_raw = (in_shape + cas_length - 1) // cas_length if cas_length else in_shape
    tile_outer_raw = ((full_outer + cas_num - 1) // cas_num if cas_num else full_outer) if row_wise else full_outer

    if tile_inner_lhs_raw * lhs_bytes % 4 != 0:
        return None

    tile_inner_rhs = align_up(tile_inner_rhs_raw, rhs_align)
    tile_inner_lhs = align_up(tile_inner_lhs_raw, lhs_align)
    tile_outer = align_up(tile_outer_raw, outer_granularity) if row_wise else full_outer
    if row_wise and tile_outer * cas_num > full_outer:
        # Padding a row-slice past the real M would compute rows that do not exist.
        return None

    bank_usage = _tile_bank_usage(
        op_type=op_type,
        device=device,
        outer_extent=tile_outer,
        tile_inner_lhs=tile_inner_lhs,
        tile_inner_rhs=tile_inner_rhs,
        lhs_bytes=lhs_bytes,
        rhs_bytes=rhs_bytes,
        output_bytes=output_bytes,
    )
    if int(bank_usage['max_bank_tile_bytes']) > int(bank_usage['bank_capacity_bytes']):
        return None

    return MatmulTiling(
        cas_num=int(cas_num),
        cas_length=int(cas_length),
        tile_inner_lhs_raw=int(tile_inner_lhs_raw),
        tile_inner_lhs=int(tile_inner_lhs),
        tile_inner_rhs_raw=int(tile_inner_rhs_raw),
        tile_inner_rhs=int(tile_inner_rhs),
        contract=str(contract),
        tile_outer=int(tile_outer),
        tile_outer_raw=int(tile_outer_raw),
    )


def _partition_ranges(node, device, microtiling: MatmulMicrotileConfig, contract: str) -> tuple[range, range]:
    """The cas_num and cas_length the tiling search tries: no more than the device's rows and columns, and no more
    chains or cascade stages than the partitioned extent holds aligned tiles or a memory tile has ports."""
    lhs_shape = view_shape(node, input_tensor_for_role(node, 'lhs'), 'inputs')
    in_shape = int(lhs_shape[-1])
    out_shape = int(view_shape(node, node.outputs[0], 'outputs')[-1])
    outer_granularity = 2 * microtiling.microtile_m
    last_outer = int(lhs_shape[-2]) if len(lhs_shape) > 1 else 1
    full_outer = int(math.prod(lhs_shape[:-1])) // max(1, last_outer) * align_up(last_outer, outer_granularity)
    # cas_num partitions N under 'inner' but the outer (M) rows under 'outer', so the
    # upper bound comes from whichever axis it slices.
    partitioned_extent, partition_align = (
        (full_outer, outer_granularity) if contract == 'outer' else (out_shape, 2 * microtiling.microtile_n)
    )
    max_chains = min(
        max(1, int(device.rows) - int(device.row_start)),
        max(max(1, int(device.max_mem_out_ports)), ceildiv(partitioned_extent, partition_align)),
    )
    max_lengths = min(
        max(1, int(device.columns) - int(device.column_start)),
        max(max(1, int(device.max_mem_in_ports)), ceildiv(in_shape, 2 * microtiling.microtile_k)),
    )
    return range(1, max_chains + 1), range(1, max_lengths + 1)


def _rows_span_axes(lhs_shape) -> bool:
    """Whether the rows an 'outer' split divides span more than one axis: a leading axis longer than one."""
    return any(int(dim) > 1 for dim in lhs_shape[:-2])


def matmul_parallelism_candidates(node, device) -> tuple[Dict[str, Any], ...]:
    """Every (contract, cas_num, cas_length) the tiling search would try under any microtile the precision allows;
    whether one fits is the tiling's to decide."""
    precision, _ = resolve_operand_precision(node, device)
    lhs_shape = view_shape(node, input_tensor_for_role(node, 'lhs'), 'inputs')
    contracts = ['inner'] if _rows_span_axes(lhs_shape) else sorted(STAGING_CONTRACTS)
    found = set()
    for m, k, n in _supported_microtile_options(device.generation, precision['lhs'], precision['rhs']):
        microtiling = MatmulMicrotileConfig(microtile_m=m, microtile_k=k, microtile_n=n)
        for contract in contracts:
            chains, lengths = _partition_ranges(node, device, microtiling, contract)
            found |= {(contract, cas_num, cas_length) for cas_num in chains for cas_length in lengths}
    return tuple({'contract': c, 'cas_num': n, 'cas_length': k} for c, n, k in sorted(found))


def _resolve_parallelism(
    node,
    device,
    microtiling: MatmulMicrotileConfig,
    precision: Dict[str, AIEDataType],
    contract: str,
    parallel_cfg: Dict[str, Any],
) -> MatmulTiling:
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    lhs_shape = view_shape(node, lhs_tensor, 'inputs')
    in_shape = lhs_shape[-1]
    out_shape = view_shape(node, node.outputs[0], 'outputs')[-1]
    cas_num, cas_length = int(parallel_cfg.get('cas_num', 1)), int(parallel_cfg.get('cas_length', 1))
    if contract == 'outer' and _rows_span_axes(lhs_shape):
        raise ConfigRefused(
            f"{node.name}: its input {tuple(lhs_shape)} cannot be split by rows (contract 'outer'): the rows span "
            'more than one axis.'
        )

    lhs_bytes = element_bytes(precision['lhs'])
    rhs_bytes = element_bytes(precision['rhs'])
    output_bytes = element_bytes(precision['output'])

    lhs_align = 2 * microtiling.microtile_k
    rhs_align = 2 * microtiling.microtile_n
    outer_granularity = 2 * microtiling.microtile_m

    last_outer = int(lhs_shape[-2]) if len(lhs_shape) > 1 else 1
    outer_extent = int(math.prod(lhs_shape[:-1]))
    padded_last_outer = align_up(last_outer, outer_granularity)
    full_outer = (outer_extent // max(1, last_outer)) * padded_last_outer

    tiling = _parallelism_candidate(
        op_type=node.op_type,
        device=device,
        in_shape=int(in_shape),
        out_shape=int(out_shape),
        lhs_align=int(lhs_align),
        rhs_align=int(rhs_align),
        lhs_bytes=int(lhs_bytes),
        rhs_bytes=int(rhs_bytes),
        output_bytes=int(output_bytes),
        full_outer=int(full_outer),
        cas_num=int(cas_num),
        cas_length=int(cas_length),
        outer_granularity=int(outer_granularity),
        contract=contract,
    )
    if tiling is None:
        raise ConfigRefused(
            f'{node.name}: cas_num={cas_num} x cas_length={cas_length} does not tile it under the {contract!r} '
            'contract: a slice would end mid-word, or a tile would overflow a memory bank.'
        )
    return tiling


def _build_matmul_io_views(node, microtiling: MatmulMicrotileConfig, tiling: MatmulTiling) -> Dict[str, TensorView]:
    # 'outer' slices the rows across cas_num tiles and keeps N whole, so the output's inner
    # extent is one tile's N (not cas_num of them) and lhs/output carry a per-tile row slice.
    row_wise = tiling.contract == 'outer'
    full_inner_lhs = tiling.tile_inner_lhs * tiling.cas_length
    full_inner_out = tiling.tile_inner_rhs if row_wise else tiling.tile_inner_rhs * tiling.cas_num
    outer_granularity = 2 * microtiling.microtile_m

    lhs_microtile = MicrotileShape(outer=int(microtiling.microtile_m), inner=int(microtiling.microtile_k))
    rhs_microtile = MicrotileShape(outer=int(microtiling.microtile_k), inner=int(microtiling.microtile_n))
    out_microtile = MicrotileShape(outer=int(microtiling.microtile_m), inner=int(microtiling.microtile_n))

    shapes: Dict[str, TensorView] = {}

    for tensor in node.inputs:
        role = input_role(node, tensor.name)
        real = tuple(int(x) for x in view_shape(node, tensor, 'inputs'))

        if role == 'lhs':
            last_outer = int(real[-2]) if len(real) > 1 else 1
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=full_inner_lhs,
                tile_inner=tiling.tile_inner_lhs,
                tile_inner_raw=tiling.tile_inner_lhs_raw,
                full_outer=align_up(last_outer, outer_granularity),
                tile_outer=tiling.tile_outer if row_wise else None,
                tile_outer_raw=tiling.tile_outer_raw if row_wise else None,
                microtile=lhs_microtile,
            )
        elif role == 'rhs' and not tensor.is_parameter:
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=full_inner_out,
                tile_inner=tiling.tile_inner_rhs,
                tile_inner_raw=tiling.tile_inner_rhs_raw,
                full_outer=full_inner_lhs,
                tile_outer=tiling.tile_inner_lhs,
                tile_outer_raw=tiling.tile_inner_lhs_raw,
                microtile=rhs_microtile,
            )
        else:
            shapes[tensor.name] = build_tensor_view(
                node,
                tensor,
                'inputs',
                full_inner=real[-1] if real else 1,
                tile_inner=real[-1] if real else 1,
                tile_inner_raw=real[-1] if real else 1,
                full_outer=real[-2] if len(real) >= 2 else 1,
            )

    for tensor in node.outputs:
        real = tuple(int(x) for x in view_shape(node, tensor, 'outputs'))
        last_outer = int(real[-2]) if len(real) > 1 else 1
        shapes[tensor.name] = build_tensor_view(
            node,
            tensor,
            'outputs',
            full_inner=full_inner_out,
            tile_inner=tiling.tile_inner_rhs,
            tile_inner_raw=tiling.tile_inner_rhs_raw,
            full_outer=align_up(last_outer, outer_granularity),
            tile_outer=tiling.tile_outer if row_wise else None,
            tile_outer_raw=tiling.tile_outer_raw if row_wise else None,
            microtile=out_microtile,
        )

    return shapes


def _validate_matmul_family_rank_contract(node) -> None:
    """Reject batched RHS MatMul until a batched_matmul lowering is explicit.

    The dense/matmul family exposes a 2-D GEMM ABI. Leading LHS/output axes may be compacted
    into M only when RHS is a single broadcastable KxN matrix. A non-broadcast
    RHS leading axis means each compacted M block needs a different KxN tile,
    which must be lowered as parallel rank-2 MatMuls or a batched variant.
    """
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    rhs_tensor = input_tensor_for_role(node, 'rhs')

    lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))
    rhs_shape = tuple(int(x) for x in view_shape(node, rhs_tensor, 'inputs'))
    if len(lhs_shape) < 2:
        raise ValueError(f'{node.name}: {node.op_type}.v1 requires rank>=2 LHS tensors, got {lhs_shape}.')
    if len(rhs_shape) < 2:
        raise ValueError(f'{node.name}: {node.op_type}.v1 requires rank>=2 RHS tensors, got {rhs_shape}.')
    rhs_batch = rhs_shape[:-2] if len(rhs_shape) > 2 else ()
    if rhs_batch and any(int(dim) != 1 for dim in rhs_batch):
        raise ValueError(
            f'{node.name}: matmul.v1 does not support batched RHS MatMul with non-broadcast leading axes; '
            f'lhs shape {lhs_shape}, rhs shape {rhs_shape}. Lower to parallel rank-2 MatMul ops or add a '
            'batched_matmul variant.'
        )


class _MatmulFamilyBase(FamilyResolver):
    """Shared capabilities of the GEMM families: both reduce over their LHS rows."""

    supported_fusions = frozenset({'bias', 'relu'})

    def parallelism_candidates(self, node, device):
        return matmul_parallelism_candidates(node, device)

    def reorder_reduction_rows(self, node, tensor, order) -> None:
        rhs = input_tensor_for_role(node, 'rhs')
        order = np.asarray(order)
        if not rhs.is_parameter:
            raise NotImplementedError(
                f'{node.name}: {self.op_type} can only adopt a reordered {tensor.name!r} into a constant RHS.'
            )
        data = np.asarray(rhs.data)
        if data.ndim != 2 or data.shape[0] != order.size:
            raise ValueError(
                f'{node.name}: RHS {data.shape} does not have one row per element of {tensor.name!r} ({order.size}).'
            )
        rhs.data = data[order]


@family_resolver('dense')
class DenseFamilyResolver(_MatmulFamilyBase):
    op_type = 'dense'

    def validate_structure(self, node, _device) -> None:
        _validate_matmul_family_rank_contract(node)


@family_resolver('matmul')
class MatmulFamilyResolver(_MatmulFamilyBase):
    op_type = 'matmul'

    def validate_structure(self, node, _device) -> None:
        _validate_matmul_family_rank_contract(node)
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        if rhs_tensor.is_parameter:
            raise ValueError(
                f'{node.name}: matmul.v1 requires a runtime RHS tensor. Constant RHS MatMul must lower to dense, '
                'parallel rank-2 MatMul ops, or a static/batched matmul variant.'
            )
