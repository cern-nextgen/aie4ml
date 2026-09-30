from __future__ import annotations

import math
from typing import Any, ClassVar, Dict

from ....aie_types import FloatIntent
from ....ir.graph import ExecutionInstance, OpNode, input_tensor_for_role
from ...base import BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PortBinding, PortMap, kernel_endpoints
from ...registry import register_variant
from ...utils import (
    ParallelismConfig,
    align_up,
    build_io_views,
    ceildiv,
    describe_partition_staging,
    extract_inner_outer,
    find_tile_split,
    parse_directives,
)
from ...utils.io import view_layout, view_shape
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
)
from ...utils.tensor_view import microtile_from_staging, staging_tile_shape
from .common import elementwise_vec_size
from .config import AddConfig, AddFlags


def _adopted_contract(node: OpNode, input_contracts):
    """The producer contract whose layout the add takes for all its tensors, or None when it lays out its own.

    An add sums element by element, so any one layout serves all its tensors: it takes its first operand's -- a conv
    frame, a matmul's microtiles, a row band -- so that operand hands over directly, and the other does too where
    its producer writes the same layout. It needs every tensor read plainly, of one shape, and each producer port to
    hold one tile of the producer's view (a halo row does not).
    """
    tensors = (*node.inputs, *node.outputs)
    if len({tuple(t.shape) for t in tensors}) != 1:
        return None
    if any(view_layout(node, t, 'inputs' if t in node.inputs else 'outputs').get('perm') for t in tensors):
        return None
    contract = next((input_contracts[t.name] for t in node.inputs if t.name in input_contracts), None)
    if contract is None or contract.inner_shards is not None or contract.view.perm is not None:
        return None
    tile = math.prod(contract.view.tile)
    if any(math.prod(staging_tile_shape(staging)) != tile for staging in contract.port_staging):
        return None
    return contract


@register_variant
class AddOpImplVariant(OpImplVariant):
    variant_id = 'add.v1'
    op_type = 'add'
    kernel_transposes_microtile = True
    graph_header = 'elementwise_add_graph.h'
    graph_name = 'elementwise_add_graph'
    param_template = 'elementwise_add'
    plevel = 10
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism'})

    def matches(self, _node: OpNode, device, _directives) -> bool:
        return device.generation in ('AIE', 'AIE-ML', 'AIE-MLV2')

    def resolve(self, node: OpNode, device, directives, input_contracts) -> AddConfig:
        io_route, parallel_cfg = parse_directives(directives)

        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        adopted = _adopted_contract(node, input_contracts)
        first = next((t for t in (lhs_tensor, rhs_tensor) if t.name in input_contracts), None)
        staging_contract = 'outer' if first is None else input_contracts[first.name].contract
        # An operand read through a transpose crosses a memory tile anyway, which can re-shard it: the add keeps
        # its microtile, not its partition, so the next row-wise reduction needs no second memory tile.
        microtile_override = None
        perm = None if first is None else view_layout(node, first, 'inputs').get('perm')
        if perm is not None and int(perm[-1]) != len(perm) - 1:
            microtile_override = microtile_from_staging(input_contracts[first.name].port_staging[0])

        lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))

        precision = {
            'lhs': resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name),
            'rhs': resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name),
        }

        is_float = isinstance(lhs_tensor.precision, FloatIntent)
        vec_size = elementwise_vec_size(precision['lhs'], device)
        bank_bytes = int(device.bank_mem_bytes)
        max_rows = max(1, int(device.rows) - int(device.row_start))
        elem_bytes = storage_bytes_for_spec(precision['lhs'])
        # The kernel adds whole vectors: rows padded to a DMA word when a tile then holds whole vectors, else to one.
        aligns = (max(1, 4 // elem_bytes), vec_size)

        if adopted is not None:
            cas_num, staging_contract = len(adopted.port_staging), adopted.contract
            io_views = {t.name: adopted.view for t in (*node.inputs, *node.outputs)}
        elif staging_contract == 'inner':
            raw_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            compacted_outer = outer_prefix * last_outer
            for align in aligns:
                full_inner = align_up(raw_inner, align)
                cas_num, tile_inner = find_tile_split(
                    partition_size=full_inner,
                    max_rows=max_rows,
                    bank_bytes=bank_bytes,
                    tile_bytes_fn=lambda ti: compacted_outer * ti * elem_bytes,
                    parallel_cfg=parallel_cfg,
                    input_contracts=input_contracts,
                    primary_tensor_name=lhs_tensor.name,
                    contract='inner',
                    require_match=microtile_override is None,
                )
                if compacted_outer * tile_inner % vec_size == 0:
                    break
            io_views = build_io_views(
                node,
                list(node.inputs),
                list(node.outputs),
                full_inner=full_inner,
                full_outer=last_outer,
                tile_inner=tile_inner,
                tile_outer=last_outer,
                tile_inner_raw=ceildiv(raw_inner, cas_num),
                tile_outer_raw=last_outer,
                microtile=microtile_override,
            )
        else:
            raw_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            for align in aligns:
                full_inner = align_up(raw_inner, align)
                cas_num, tile_outer = find_tile_split(
                    partition_size=last_outer,
                    max_rows=max_rows,
                    bank_bytes=bank_bytes,
                    tile_bytes_fn=lambda to, inner=full_inner: outer_prefix * to * inner * elem_bytes,
                    parallel_cfg=parallel_cfg,
                    input_contracts=input_contracts,
                    primary_tensor_name=lhs_tensor.name,
                    contract='outer',
                )
                if outer_prefix * tile_outer * full_inner % vec_size == 0:
                    break
            io_views = build_io_views(
                node,
                list(node.inputs),
                list(node.outputs),
                full_inner=full_inner,
                full_outer=tile_outer * cas_num,
                tile_inner=full_inner,
                tile_outer=tile_outer,
                tile_inner_raw=raw_inner,
                tile_outer_raw=tile_outer,
                microtile=microtile_override,
            )

        # The DMA walks a transposed operand's grid in view order; the kernel does the block.
        flags = AddFlags(
            transpose_lhs=io_views[lhs_tensor.name].is_transposed,
            transpose_rhs=io_views[rhs_tensor.name].is_transposed,
        )
        microtile = io_views[lhs_tensor.name].microtile
        if (flags.transpose_lhs or flags.transpose_rhs) and microtile is not None:
            vec_size = int(microtile.outer) * int(microtile.inner)

        if is_float:
            shift, accumulator_tag, rounding_mode = 0, 'accfloat', 'conv_even'
        else:
            shift = resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision)
            accumulator_tag = infer_accumulator_tag(device, precision['lhs'], precision['rhs'], precision.get('acc'))
            rounding_mode = aie_rounding_token(precision['output'])

        return AddConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num), contract=staging_contract),
            vec_size=vec_size,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode=rounding_mode,
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            adopted_staging=None if adopted is None else adopted.port_staging,
            flags=flags,
            microtile=microtile,
        )

    def validate_config(self, node: OpNode, config: AddConfig, _device) -> None:
        elements = int(math.prod(config.io_views[input_tensor_for_role(node, 'lhs').name].tile))
        if elements % int(config.vec_size):
            raise ValueError(
                f'{node.name}: a tile of {elements} elements is not whole {config.vec_size}-element vectors, which '
                'the add kernel steps through.'
            )

    def kernel_params(self, node: OpNode, config: AddConfig):
        lhs_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params['tile_elements'] = int(math.prod(lhs_view.tile))
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, _producer=None):
        if config.adopted_staging is not None:
            staging = config.adopted_staging[int(port)]
            return {**staging, 'access': 'read', 'boundary_dimension': list(staging['io_boundary_dimension'])}
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'read',
            config.parallelism.contract,
        )

    def describe_output_staging(self, node, config, tensor_name, port):
        if config.adopted_staging is not None:
            return dict(config.adopted_staging[int(port)])
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'write',
            config.parallelism.contract,
        )

    def output_staging_contract(self, node, config: AddConfig, tensor_name: str):
        return str(config.parallelism.contract)

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: ExecutionInstance):
        return []

    def footprint(self, node: OpNode, config: AddConfig) -> OpImplFootprint:
        return OpImplFootprint(
            width=1,
            height=int(config.parallelism.cas_num),
        )

    def buffer_locations(self, _node: OpNode, config: AddConfig, anchor_row: int):
        locations = []
        for row in range(int(config.parallelism.cas_num)):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + row, 1)
            locations.append(BufferLocation('in1', row, flow.input_col, row, (0, 3)))
            locations.append(BufferLocation('in2', row, 0, row, (1, 2)))
            locations.append(BufferLocation('out1', row, flow.output_col, row, (0, 3)))
        return tuple(locations)

    def build_ports(self, node: OpNode, config: AddConfig):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={
                lhs_tensor.name: PortBinding('in1', n, endpoints=kernel_endpoints(n, 'in[0]')),
                rhs_tensor.name: PortBinding('in2', n, endpoints=kernel_endpoints(n, 'in[1]')),
            },
            outputs={node.outputs[0].name: PortBinding('out1', n, endpoints=kernel_endpoints(n, 'out[0]'))},
        )
