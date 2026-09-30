from __future__ import annotations

from ....aie_types import FloatIntent
from ....ir.graph import ExecutionInstance, OpNode, input_role, input_tensor_for_role
from ...base import BufferLocation, OpImplFootprint, row_flow
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, parse_directives
from ...utils.precision import (
    aie_rounding_token,
    resolve_accumulator_output_shift,
    resolve_operand_precision,
    resolve_output_scale_shift,
)
from .common import (
    bitwidths_supported,
    describe_inner_lhs_staging,
    describe_inner_output_staging,
    describe_inner_rhs_staging,
    describe_outer_lhs_staging,
    describe_outer_output_staging,
    describe_outer_rhs_staging,
    requested_contract,
)
from .config import MatmulConfig, MatmulFlags
from .dense import _BaseDenseMatmulVariant
from .resolver import _build_matmul_io_views, _resolve_parallelism, _resolve_tile_cfg


class _MatmulVariantBase(_BaseDenseMatmulVariant):
    """Everything the matmul variants share, independent of which contract they implement."""

    op_type = 'matmul'
    kernel_transposes_microtile = True
    graph_header = 'matmul_graph.h'
    graph_name = 'matmul_graph'
    param_template = 'matmul'
    plevel = 10

    def matches(self, node: OpNode, device, directives) -> bool:
        return requested_contract(node, directives) == self.contract and bitwidths_supported(node, device)

    def resolve(self, node: OpNode, device, directives, _input_contracts) -> MatmulConfig:
        io_route, parallel_cfg = parse_directives(directives)
        precision, accumulator_tag = resolve_operand_precision(node, device)
        microtiling = _resolve_tile_cfg(node, device, precision['lhs'], precision['rhs'])
        tiling = _resolve_parallelism(node, device, microtiling, precision, self.contract, parallel_cfg)
        io_views = _build_matmul_io_views(node, microtiling, tiling)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        is_float = isinstance(rhs_tensor.precision, FloatIntent)

        shift = (
            0
            if is_float
            else resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision, rhs_tensor.precision)
        )
        shift += resolve_output_scale_shift(node, is_float=is_float)

        return MatmulConfig(
            precision=precision,
            parallelism=ParallelismConfig(
                cas_length=tiling.cas_length, cas_num=tiling.cas_num, contract=tiling.contract
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            flags=MatmulFlags(
                transpose_lhs=io_views[lhs_tensor.name].is_transposed,
                transpose_rhs=io_views[rhs_tensor.name].is_transposed,
            ),
        )

    def validate_config(self, node: OpNode, config: MatmulConfig, _device) -> None:
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        rhs_view = config.io_views[rhs_tensor.name]
        rhs_perm = rhs_view.perm
        if rhs_perm is not None:
            rank = len(rhs_perm)
            identity = list(range(rank))
            swapped = list(range(rank))
            if rank >= 2:
                swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
            if list(rhs_perm) not in [identity, swapped]:
                raise ValueError(f'{node.name}: matmul RHS does not support io_view permutation {rhs_perm}.')
        rhs_is_float = isinstance(rhs_tensor.precision, FloatIntent)
        if not rhs_is_float and not bool(config.precision['rhs'].signed):
            raise ValueError(f'{node.name}: matmul RHS must use a signed integer precision.')

    def pack(self, _inst: ExecutionInstance):
        return {}

    def get_artifacts(self, _inst: ExecutionInstance):
        return []

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=int(config.parallelism.cas_length),
            height=int(config.parallelism.cas_num),
        )

    def buffer_locations(self, _node, config, anchor_row):
        locations = []
        cas_num = int(config.parallelism.cas_num)
        cas_length = int(config.parallelism.cas_length)
        outer = config.parallelism.contract == 'outer'
        for row in range(cas_num):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + row, cas_length)
            for pos in range(cas_length):
                idx = row * cas_length + pos
                tile_col = cas_length - 1 - pos if flow.reversed else pos
                lhs_port = idx if outer else pos
                locations.append(BufferLocation('inA', lhs_port, tile_col + flow.input_col, row, (0, 3)))
                locations.append(BufferLocation('inB', idx, tile_col, row, (1, 2)))
            last = 0 if flow.reversed else cas_length - 1
            locations.append(BufferLocation('outC', row, last + flow.output_col, row, (0, 3)))
        return tuple(locations)

    def build_ports(self, node: OpNode, config: MatmulConfig):
        cas_length = int(config.parallelism.cas_length)
        cas_num = int(config.parallelism.cas_num)
        tiles = cas_length * cas_num
        if config.parallelism.contract == 'outer':
            lhs_endpoints = tuple((f'kk[{port}].in[0]',) for port in range(tiles))
        else:
            lhs_endpoints = tuple(
                tuple(f'kk[{chain * cas_length + port}].in[0]' for chain in range(cas_num))
                for port in range(cas_length)
            )
        rhs_endpoints = tuple((f'kk[{port}].in[1]',) for port in range(tiles))
        out_endpoints = tuple((f'kk[{chain * cas_length + cas_length - 1}].out[0]',) for chain in range(cas_num))
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding('inA', len(lhs_endpoints), endpoints=lhs_endpoints),
                rhs_tensor.name: PortBinding('inB', tiles, endpoints=rhs_endpoints),
            },
            outputs={node.outputs[0].name: PortBinding('outC', cas_num, endpoints=out_endpoints)},
        )


@register_variant
class MatmulOpImplVariant(_MatmulVariantBase):
    """Matmul with the output features partitioned across cascade chains ('inner' contract)."""

    variant_id = 'matmul.v1'
    contract = 'inner'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_full_outer

    def describe_input_staging(self, node, config, tensor_name, port, _producer=None):
        view = config.io_views[tensor_name]
        if input_role(node, tensor_name) == 'rhs':
            return describe_inner_rhs_staging(view, config.parallelism, port)
        return describe_inner_lhs_staging(view, port)

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_inner_output_staging(config.io_views[tensor_name], port)


@register_variant
class MatmulRowWiseOpImplVariant(_MatmulVariantBase):
    """Matmul with the rows partitioned across cascade chains ('outer' contract)."""

    variant_id = 'matmul.row.v1'
    contract = 'outer'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_tile_outer

    def describe_input_staging(self, node, config, tensor_name, port, _producer=None):
        view = config.io_views[tensor_name]
        if input_role(node, tensor_name) == 'rhs':
            return describe_outer_rhs_staging(view, config.parallelism, port)
        return describe_outer_lhs_staging(view, config.parallelism, port)

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_outer_output_staging(config.io_views[tensor_name], port)
