from __future__ import annotations

import dataclasses
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...op_impls.common_types import PORT_KIND_STREAM
from ...op_impls.utils import STORAGE_LAYOUT_LINEAR, STORAGE_LAYOUT_MICROTILED, StagingDescriptor
from .descriptors import boundary_access_descriptor, localize_descriptor, rebase_descriptor_offset


def direct_boundary_access(
    port_staging: StagingDescriptor, io_tiling: Sequence[int], *, element_bits: int, output: bool
) -> Tuple[Dict[str, Any], Optional[List[int]]]:
    """The kernel-side DMA access of one port of a direct PLIO leg, and the shape the PLIO carries per iteration
    when it differs from `io_tiling`. The tile DMA moves only the logical elements; ConfigRefused where it cannot
    walk the port's layout, which then only a memory tile can re-stage."""
    local = localize_descriptor(port_staging, port_staging.offset, port_staging.tile_shape)
    padded = prod(local.tile_shape) != prod(int(v) for v in io_tiling)
    return boundary_access_descriptor(
        local,
        element_bits=int(element_bits),
        project_to_io_boundary=padded and (output or local.storage_layout == STORAGE_LAYOUT_MICROTILED),
    )


def graph_input_port_descs(entry, execution, port_base: int) -> Dict[int, StagingDescriptor]:
    consumer = entry.single_consumer()
    inst = execution.get(consumer.node.name)
    binding = inst.ports.inputs[consumer.tensor]
    consumer_ports = consumer.selected_ports(binding.count)
    count = len(consumer_ports)
    if count != int(entry.producer_port_count):
        raise ValueError(
            f'{entry.logical_tensor}: graph-input producer_ports must match consumer port count '
            f'({entry.producer_port_count} != {count}).'
        )
    descs: Dict[int, StagingDescriptor] = {}
    for local_port, port in enumerate(consumer_ports):
        graph_port = int(port_base) + int(local_port)
        desc = rebase_descriptor_offset(inst.staging[consumer.tensor][int(port)], consumer.offset_base)
        if entry.producer.offset_base:
            desc = _in_input(desc, entry.producer.offset_base, execution, inst, consumer.tensor)
        descs[graph_port] = desc
        if binding.kind == PORT_KIND_STREAM:
            require_linear_stream_staging(entry.logical_tensor, descs[graph_port])
    return descs


def _in_input(desc: StagingDescriptor, base, execution, inst, sliced: str) -> StagingDescriptor:
    """A port's staging of a slice of a graph input, in the input's frame: its data where the slice starts (`base`),
    the input's whole extent along the slice's axis, which the host holds."""
    view = execution.values[sliced].view
    source = next(t for t in inst.node.inputs if t.name == sliced).producer.inputs[0]  # the folded slice's input
    bound = list(desc.io_boundary_dimension)
    bound[inst.port_views[sliced].buffer_order.index(view.axis)] = int(source.shape[view.axis])
    origin = tuple(o + b for o, b in zip(desc.logical_origin, base))
    return dataclasses.replace(desc, logical_origin=origin, io_boundary_dimension=tuple(bound))


def require_linear_stream_staging(tensor: str, staging: StagingDescriptor) -> None:
    """A PLIO carries the tensor in linear order, so a stream endpoint on the boundary must too."""
    if staging.storage_layout != STORAGE_LAYOUT_LINEAR:
        raise RuntimeError(
            f'{tensor}: a stream port on the graph boundary must stage in linear order, got {staging.storage_layout!r}.'
        )


def host_visible_input_staging(base: StagingDescriptor, *, stream: bool = False) -> Dict[str, Any]:
    """Host-visible descriptor of one graph-input port.

    A DMA-fed buffer receives only the logical elements (`io_tiling_dimension`) and the DMA
    scatters them; a stream port receives the whole padded port tile, so its transfer shape
    is the staging `tiling_dimension`. `logical_origin` says where that window starts in the
    tensor, and travels unchanged: transport never recomputes where a port's data lives. A port
    that frames each inference in whole transfer units declares `transfer_bytes`, which travels too.
    """
    io_tile = list(base.io_tiling_dimension)
    desc = {
        'access': 'write',
        'storage_layout': STORAGE_LAYOUT_LINEAR,
        'buffer_dimension': list(base.buffer_dimension),
        'tiling_dimension': list(base.tiling_dimension) if stream else list(io_tile),
        'io_tiling_dimension': list(io_tile),
        'io_boundary_dimension': list(base.io_boundary_dimension),
        'offset': list(base.offset),
        'logical_origin': list(base.logical_origin),
        'slice_dimension': base.slice_dimension,
        'inner_dimension': base.inner_dimension,
        'outer_dimension': base.outer_dimension,
    }
    if base.transfer_bytes is not None:
        desc['transfer_bytes'] = base.transfer_bytes
    return desc


def graph_input_port_descriptor(entry, port: int) -> StagingDescriptor:
    try:
        return entry.graph_input[int(port)]
    except KeyError as exc:
        raise RuntimeError(f'{entry.logical_tensor}: missing graph-input descriptor for port {port}.') from exc
