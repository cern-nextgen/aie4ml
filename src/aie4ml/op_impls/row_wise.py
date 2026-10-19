"""Variants of an op over whole rows (its last axis), the rows split into bands across `cas_num` tiles and no cascade:
LayerNorm and Softmax, in a linear or a tiled layout."""

from __future__ import annotations

from ..errors import ConfigRefused
from ..ir.graph import OpNode, input_tensor_for_role
from .base import PING_PONG_BANKS, BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from .common_types import PortBinding, PortMap, kernel_endpoints
from .utils import MicrotileShape, describe_partition_staging, inherited_microtile


class RowWiseOpImplVariant(OpImplVariant):
    """One kernel per row band, a buffer in and a buffer out each."""

    def describe_input_staging(self, _node, config, tensor_name, port):
        return describe_partition_staging(config.io_views[tensor_name], port, 'read', 'outer')

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_partition_staging(config.io_views[tensor_name], port, 'write', 'outer')

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return str(config.parallelism.contract)

    def footprint(self, _node: OpNode, config) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=int(config.parallelism.cas_num))

    def buffer_locations(self, _node: OpNode, config, anchor_row: int):
        locations = []
        for row in range(int(config.parallelism.cas_num)):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + row, 1)
            locations.append(BufferLocation('in1', row, flow.input_col, row, PING_PONG_BANKS, reader=(0, row)))
            locations.append(BufferLocation('out1', row, flow.output_col, row, PING_PONG_BANKS))
        return tuple(locations)

    def build_ports(self, node: OpNode, config):
        in_tensor = input_tensor_for_role(node, 'lhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={in_tensor.name: PortBinding('in1', n, endpoints=kernel_endpoints(n, 'in[0]'))},
            outputs={node.outputs[0].name: PortBinding('out1', n, endpoints=kernel_endpoints(n, 'out[0]'))},
        )


class TiledRowWiseMixin:
    """The tiled layout: rows in microtile bands, in the producer's microtile where it writes one, so a matmul on
    either side hands over directly. Preferred where no `layout` directive asks for the linear one."""

    plevel = 11

    def resolve_microtile(self, node: OpNode, directives, input_contracts):
        """Match the producer's microtile so the edge is direct; else choose our own."""
        return inherited_microtile(node, input_contracts) or self.preferred_microtile(node, directives)

    def preferred_microtile(self, node: OpNode, directives) -> MicrotileShape:
        """This kernel's own microtile when nothing upstream constrains it (e.g., a graph boundary).

        A `microtiling` directive pins it (microtile_m -> row band, microtile_n -> feature block);
        otherwise 4x8.
        """
        mt = directives.get('microtiling')
        if mt is None:
            return MicrotileShape(outer=4, inner=8)
        if set(mt) != {'microtile_m', 'microtile_n'}:
            raise ValueError(f'{node.name}: microtiling takes microtile_m and microtile_n here, got {sorted(mt)}.')
        return MicrotileShape(outer=int(mt['microtile_m']), inner=int(mt['microtile_n']))

    @staticmethod
    def check_whole_microtiles(node: OpNode, mt: MicrotileShape, rows: int, cols: int) -> None:
        if rows % mt.outer:
            raise ConfigRefused(
                f'{node.name}: rows={rows} must be a whole number of {mt.outer}-row microtile bands. '
                f'Partitioning across kernels chose this tile height, so adjust cas_num.'
            )
        if cols % mt.inner:
            raise ValueError(f'{node.name}: cols={cols} must be a whole number of {mt.inner}-wide microtiles.')
