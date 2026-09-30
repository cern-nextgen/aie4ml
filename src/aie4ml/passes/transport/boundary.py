from __future__ import annotations

import copy
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...op_impls.common_types import PORT_KIND_STREAM
from ...op_impls.utils import STORAGE_LAYOUT_LINEAR, STORAGE_LAYOUT_MICROTILED
from ...op_impls.utils.tensor_view import staging_tile_shape
from .descriptors import boundary_access_descriptor, localize_descriptor, rebase_descriptor_offset


def direct_boundary_access(
    port_staging: Dict[str, Any], io_tiling: Sequence[int], *, element_bits: int, output: bool
) -> Tuple[Dict[str, Any], Optional[List[int]]]:
    """The kernel-side DMA access of one port of a direct PLIO leg, and the shape the PLIO carries per iteration
    when it differs from `io_tiling`. The tile DMA moves only the logical elements; ConfigRefused where it cannot
    walk the port's layout, which then only a memory tile can re-stage."""
    descriptor = copy.deepcopy(port_staging)
    localize_descriptor(descriptor, [int(v) for v in descriptor['offset']], staging_tile_shape(descriptor))
    padded = prod(staging_tile_shape(descriptor)) != prod(int(v) for v in io_tiling)
    descriptor = boundary_access_descriptor(
        descriptor,
        element_bits=int(element_bits),
        project_to_io_boundary=padded and (output or descriptor.get('storage_layout') == STORAGE_LAYOUT_MICROTILED),
    )
    return descriptor, descriptor.pop('transfer_shape', None)


def graph_input_port_descs(entry, ctx, port_base: int) -> Dict[int, Dict[str, Any]]:
    consumer = entry.single_consumer()
    inst = ctx.ir.execution.get(consumer.node.name)
    binding = inst.ports.inputs[consumer.tensor]
    consumer_ports = consumer.selected_ports(binding.count)
    count = len(consumer_ports)
    if count != int(entry.producer_port_count):
        raise ValueError(
            f'{entry.logical_tensor}: graph-input producer_ports must match consumer port count '
            f'({entry.producer_port_count} != {count}).'
        )
    descs: Dict[int, Dict[str, Any]] = {}
    for local_port, port in enumerate(consumer_ports):
        graph_port = int(port_base) + int(local_port)
        descs[graph_port] = inst.variant.describe_input_staging(
            consumer.node, inst.config, consumer.tensor, int(port), None
        )
        rebase_descriptor_offset(descs[graph_port], consumer.offset_base)
        if binding.kind == PORT_KIND_STREAM:
            require_linear_stream_staging(entry.logical_tensor, descs[graph_port])
    return descs


def require_linear_stream_staging(tensor: str, desc: Dict[str, Any]) -> None:
    """A PLIO carries the tensor in linear order, so a stream endpoint on the boundary must too."""
    if desc.get('storage_layout') != STORAGE_LAYOUT_LINEAR:
        raise RuntimeError(
            f'{tensor}: a stream port on the graph boundary must stage in linear order, '
            f'got {desc.get("storage_layout")!r}.'
        )


def graph_input_full_descriptor(entry, ctx) -> Dict[str, Any]:
    consumer = entry.single_consumer()
    inst = ctx.ir.execution.get(consumer.node.name)
    port = int(consumer.selected_ports(inst.ports.inputs[consumer.tensor].count)[0])
    base = inst.variant.describe_input_staging(consumer.node, inst.config, consumer.tensor, port, None)
    rebase_descriptor_offset(base, consumer.offset_base)
    return host_visible_input_staging(base, offset=[0 for _ in base['io_tiling_dimension']])


def host_visible_input_staging(base: Dict[str, Any], *, stream: bool = False, offset=None) -> Dict[str, Any]:
    """Host-visible descriptor of one graph-input port.

    A DMA-fed buffer receives only the logical elements (`io_tiling_dimension`) and the DMA
    scatters them; a stream port receives the whole padded port tile, so its transfer shape
    is the staging `tiling_dimension`. `logical_origin` says where that window starts in the
    tensor, and travels unchanged: transport never recomputes where a port's data lives. A port
    that frames each inference in whole transfer units declares `transfer_bytes`, which travels too.
    """
    io_tile = list(base['io_tiling_dimension'])
    desc = {
        'access': 'write',
        'storage_layout': STORAGE_LAYOUT_LINEAR,
        'buffer_dimension': list(base['buffer_dimension']),
        'tiling_dimension': list(base['tiling_dimension']) if stream else list(io_tile),
        'io_tiling_dimension': list(io_tile),
        'io_boundary_dimension': list(base['io_boundary_dimension']),
        'offset': list(base['offset'] if offset is None else offset),
        'logical_origin': list(base['logical_origin']),
        'slice_dimension': int(base['slice_dimension']),
        'inner_dimension': int(base['inner_dimension']),
        'outer_dimension': int(base['outer_dimension']),
    }
    if 'transfer_bytes' in base:
        desc['transfer_bytes'] = int(base['transfer_bytes'])
    return desc


def graph_input_writer_port_descs(
    read_descs: Dict[int, Dict[str, Any]], *, stream: bool = False
) -> Dict[int, Dict[str, Any]]:
    return {int(port): host_visible_input_staging(base, stream=stream) for port, base in read_descs.items()}


def graph_input_unit_box(descs: Dict[int, Dict[str, Any]], ports: List[int]):
    if not ports:
        raise ValueError('graph-input shard unit cannot be empty.')
    first = descs[int(ports[0])]
    rank = len(first['offset'])
    base = [None for _ in range(rank)]
    limit = [None for _ in range(rank)]
    for port in ports:
        desc = descs[int(port)]
        if len(desc['offset']) != rank:
            raise ValueError('graph-input port descriptors have inconsistent rank.')
        tile = list(desc['io_tiling_dimension'])
        offset = list(desc['offset'])
        for dim in range(rank):
            start = int(offset[dim])
            end = start + int(tile[dim])
            base[dim] = start if base[dim] is None else min(int(base[dim]), start)
            limit[dim] = end if limit[dim] is None else max(int(limit[dim]), end)
    return [int(v) for v in base], [int(limit[d] - base[d]) for d in range(rank)]


def graph_input_port_descriptor(entry, port: int) -> Dict[str, Any]:
    try:
        return copy.deepcopy(entry.graph_input.port_descriptors[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.logical_tensor}: missing graph-input descriptor for port {port}.') from exc


def graph_input_writer_port_descriptor(entry, port: int) -> Dict[str, Any]:
    try:
        return copy.deepcopy(entry.graph_input.writer_descriptors[int(port)])
    except KeyError as exc:
        raise RuntimeError(f'{entry.logical_tensor}: missing graph-input writer descriptor for port {port}.') from exc
