from __future__ import annotations

import math
from typing import Any, ClassVar, Dict

from ....aie_types import FloatIntent
from ....ir.graph import ExecutionInstance, OpNode, input_tensor_for_role
from ...base import BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PortBinding, PortMap, kernel_endpoints
from ...registry import register_variant
from ...utils import (
    STORAGE_LAYOUT_INNER_BLOCKED,
    ParallelismConfig,
    align_up,
    build_io_views,
    build_tensor_view_from_staging,
    ceildiv,
    describe_partition_staging,
    extract_inner_outer,
    find_tile_split,
    parse_directives,
)
from ...utils.io import resolve_input_contract, view_shape
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_accumulator_output_shift,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
)
from .common import elementwise_vec_size
from .config import AddConfig, AddFlags


def _select_preserved_staging(tensor_names, input_contracts):
    """Pick the producer staging this add inherits its geometry from, and the name it came from. A channel-blocked
    frame (a conv's output) is not a layout the add walks, so it is never inherited; like any staging the add does
    not share, it crosses a memtile."""

    inheritable = [
        n
        for n in tensor_names
        if n in input_contracts and input_contracts[n].port_staging[0]['storage_layout'] != STORAGE_LAYOUT_INNER_BLOCKED
    ]
    primary = input_contracts[inheritable[0]].port_staging if inheritable else None
    patches = {
        n: 'memtile' for n in tensor_names if n in input_contracts and input_contracts[n].port_staging != primary
    }
    return primary, (inheritable[0] if inheritable else None), patches


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

        staging_contract, conflict_patches = resolve_input_contract(
            input_contracts,
            [lhs_tensor.name, rhs_tensor.name],
        )
        preserved_staging, geometry_tensor, staging_patches = _select_preserved_staging(
            (lhs_tensor.name, rhs_tensor.name),
            input_contracts,
        )
        # BUFFER-order data for `geometry_tensor`; its view decodes the shared VIEW-order geometry.
        inherited_view = None
        if preserved_staging is not None:
            inherited_view = build_tensor_view_from_staging(
                node,
                lhs_tensor if geometry_tensor == lhs_tensor.name else rhs_tensor,
                'inputs',
                preserved_staging[0],
            )
        # A transposed operand crosses a memtile anyway, and a memtile can re-shard. Inheriting
        # would express the partition in the producer's axes, costing a second memtile at the
        # next row-wise reduction. Re-shard here: keep the microtile, drop the partition.
        microtile_override = None
        if inherited_view is not None and inherited_view.is_transposed:
            staging_patches = {**staging_patches, geometry_tensor: 'memtile'}
            microtile_override = inherited_view.microtile
            inherited_view = None
        route_patches = {**conflict_patches, **staging_patches}
        if route_patches:
            io_route = {**io_route, 'inputs': {**io_route.get('inputs', {}), **route_patches}}

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

        if inherited_view is not None:
            cas_num = len(preserved_staging)
            io_views = build_io_views(
                node,
                list(node.inputs),
                list(node.outputs),
                full_inner=inherited_view.full_inner,
                full_outer=inherited_view.full_outer,
                tile_inner=inherited_view.tile_inner,
                tile_outer=inherited_view.tile_outer,
                tile_inner_raw=inherited_view.tile_raw_inner,
                tile_outer_raw=inherited_view.tile_raw_outer,
                microtile=inherited_view.microtile,
            )
        elif staging_contract == 'inner':
            full_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            full_inner = align_up(full_inner, vec_size)
            raw_inner = int(lhs_shape[-1])
            compacted_outer = outer_prefix * last_outer
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
            full_inner, outer_prefix, last_outer = extract_inner_outer(lhs_shape)
            full_inner = align_up(full_inner, vec_size)
            raw_inner = int(lhs_shape[-1])
            cas_num, tile_outer = find_tile_split(
                partition_size=last_outer,
                max_rows=max_rows,
                bank_bytes=bank_bytes,
                tile_bytes_fn=lambda to: outer_prefix * to * full_inner * elem_bytes,
                parallel_cfg=parallel_cfg,
                input_contracts=input_contracts,
                primary_tensor_name=lhs_tensor.name,
                contract='outer',
            )
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
        # Re-deriving a descriptor can lose the producer's exact partition offsets and traversal.
        # Keep inherited staging authoritative for matching, non-transposed inputs.
        preserved_tensors = tuple(
            t.name
            for t in (lhs_tensor, rhs_tensor)
            if preserved_staging is not None
            and t.name in input_contracts
            and input_contracts[t.name].port_staging == preserved_staging
            and not io_views[t.name].is_transposed
        )
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
            preserved_staging=preserved_staging,
            preserved_tensors=preserved_tensors,
            flags=flags,
            microtile=microtile,
        )

    def validate_config(self, node: OpNode, config: AddConfig, _device) -> None:
        if config.preserved_staging is not None:
            port_count = self.output_port_count(node, config)
            if len(config.preserved_staging) != port_count:
                raise ValueError(
                    f'{node.name}: preserved_staging length {len(config.preserved_staging)} '
                    f'does not match output_port_count {port_count}.'
                )

    def kernel_params(self, node: OpNode, config: AddConfig):
        lhs_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params['tile_elements'] = int(math.prod(lhs_view.tile))
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        if tensor_name in config.preserved_tensors:
            return dict(config.preserved_staging[int(port)])
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'read',
            config.parallelism.contract,
            buf_dims,
        )

    def describe_output_staging(self, node, config, tensor_name, port, buf_dims=None):
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'write',
            config.parallelism.contract,
            buf_dims,
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
