"""How a direct edge is realised, and whether it can be one shared buffer where it must be.

A direct stream edge moves data on core streams. A direct buffer edge joins two buffer ports, and every
op pins the buffers of its buffer ports where its locations say, one copy per bank, so the AIE compiler
realises it as one buffer both kernels reach exactly where the producer's and the consumer's locations
coincide, and as a DMA copy between two pinned buffers anywhere else. An edge the execution graph marks
`shared_memory` must be the former, which is known before the compiler runs:

- each port is a buffer port bound to exactly one kernel port, and nothing else reads the producer's port, so
  nothing asks for a copy (fanout, multicast, a graph output, another view over it): a view reading a port alone,
  as a Split along the producer's ports does, leaves it one-to-one;
- once both ops are placed, the producer's port pins the buffer where the consumer's does, or in a memory the one
  core reading the consumer's port reaches (`BufferLocation.reader`, `memory_reach`): its own, west, north or south
  (east on AIE odd rows). Placement records where each port's buffer then sits (`placed_buffers`), and the
  consumer's graph pins its port there too (`pinned_buffers`).

The first does not depend on placement, and the placer refuses it outright; it asks the second while it
searches. The physical verifier asks both again of the finished plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import FrozenSet, Optional, Sequence, Set, Tuple

from ..op_impls.base import BufferLocation, memory_reach
from ..op_impls.common_types import PORT_KIND_BUFFER

Tile = Tuple[int, int]

SHARED_MEMORY = 'shared_memory'
"""A direct buffer edge realised as one buffer both kernels reach: both ports are pinned to one memory."""

DMA = 'dma'
"""A direct buffer edge not proven one buffer: its ports are pinned to different memories, which a DMA copy
joins, or other readers (fanout, multicast, a view) leave the compiler free to copy it."""

STREAM = 'stream'
"""A direct stream edge: the kernels exchange data on core streams, with no buffer to share or copy."""


def port_problem(inst, group: str, port: int, direction: str) -> Optional[str]:
    for binding in getattr(inst.ports, direction).values():
        if binding.group != group:
            continue
        if binding.kind != PORT_KIND_BUFFER:
            return f'{inst.name}.{group} is a {binding.kind} port, not a buffer'
        if len(binding.endpoints[int(port)]) != 1:
            return f'{inst.name}.{group}[{port}] multicasts to {len(binding.endpoints[int(port)])} kernels'
        return None
    return f'{inst.name} has no port group {group!r}'


def pair_problem(producer, p_group: str, p_port: int, consumer, c_group: str, c_port: int) -> Optional[str]:
    """Why these two ports could never be one buffer, however placed and whoever else reads them, or None: each must
    be a buffer port bound to exactly one kernel."""
    for inst, group, port, direction in ((producer, p_group, p_port, 'outputs'), (consumer, c_group, c_port, 'inputs')):
        problem = port_problem(inst, group, port, direction)
        if problem:
            return problem
    return None


def static_problem(ctx, producer, p_group: str, p_port: int, consumer, c_group: str, c_port: int) -> Optional[str]:
    """Why no placement could let these ports share one buffer, or None: `pair_problem`, or another reader of the
    producer's port."""
    problem = pair_problem(producer, p_group, p_port, consumer, c_group, c_port)
    if problem:
        return problem
    from .transport.collect import TransportCollector  # the transport package imports this module

    tensor = next(name for name, binding in producer.ports.outputs.items() if binding.group == p_group)
    readers = TransportCollector(ctx.ir.execution).port_readers(tensor, int(p_port))
    if readers != [consumer.name]:
        return f'{producer.name}.{p_group}[{p_port}] is read by {readers}'
    return None


def hand_over_problem(ctx, producer, p_group: str, p_port: int, consumer, c_group: str, c_port: int) -> Optional[str]:
    """Why two placed kernel ports do not hand over one buffer -- `static_problem`, else where placement put their
    buffers -- or None."""
    problem = static_problem(ctx, producer, p_group, p_port, consumer, c_group, c_port)
    written = placed_locations(ctx, producer, p_group, p_port)
    return problem or location_problem(written, placed_locations(ctx, consumer, c_group, c_port))


def pinned_locations(inst, group: str, port: int, col: int, row: int) -> Set[Tuple[int, int, Tuple[int, ...]]]:
    """Where an op placed at (col, row) lists one port's buffer: absolute (column, row, banks)."""
    return {
        (col + loc.rel_col, row + loc.rel_row, tuple(loc.banks))
        for loc in inst.variant.buffer_locations(inst.node, inst.config, row)
        if loc.port_group == group and loc.port == int(port)
    }


def placed_buffers(ctx, inst) -> Tuple[Tuple[BufferLocation, Tuple[int, int, Tuple[int, ...]]], ...]:
    """Each buffer location the op lists (`buffer_locations`), with where placement put it: absolute (column, row,
    banks)."""
    placement = ctx.ir.physical.placements[inst.name]
    locations = inst.variant.buffer_locations(inst.node, inst.config, int(placement['row']))
    if len(placement['buffers']) != len(locations):
        raise RuntimeError(f'{inst.name}: placed {len(placement["buffers"])} buffers of the {len(locations)} it lists.')
    return tuple((loc, (int(c), int(r), tuple(b))) for loc, (c, r, b) in zip(locations, placement['buffers']))


def placed_locations(ctx, inst, group: str, port: int) -> Set[Tuple[int, int, Tuple[int, ...]]]:
    """Where placement put one buffer port's buffers; every buffer port has one."""
    found = {at for loc, at in placed_buffers(ctx, inst) if loc.port_group == group and loc.port == int(port)}
    if not found:
        raise RuntimeError(f'{inst.name}.{group}[{port}]: a buffer port its variant locates nowhere.')
    return found


def reached(loc: BufferLocation, col: int, row: int, alternating_horizontal: bool) -> FrozenSet[Tile]:
    """The tiles whose memory the kernel reading an input location reaches, its op anchored at (col, row); none for
    an output."""
    if loc.reader is None:
        return frozenset()
    kernel_col, kernel_row = col + loc.reader[0], row + loc.reader[1]
    return frozenset((kernel_col + dc, kernel_row + dr) for dc, dr in memory_reach(alternating_horizontal, kernel_row))


def port_reach(
    locations: Sequence[BufferLocation], group: str, port: int, col: int, row: int, alternating_horizontal: bool
) -> FrozenSet[Tile]:
    """Where the one kernel reading an input port may read a buffer its producer pins (`reached`); none for an
    output, or a port several kernels read."""
    mine = [loc for loc in locations if loc.port_group == group and loc.port == int(port)]
    return reached(mine[0], col, row, alternating_horizontal) if len(mine) == 1 else frozenset()


def location_problem(written: Set, read: Set, reach: FrozenSet[Tile] = frozenset()) -> Optional[str]:
    """Why a producer's port pinned to `written` and a consumer's pinned to `read`, whose reading core reaches the
    memories of `reach`, cannot be one buffer, or None: one buffer where both pin it, or where the producer pins it
    within that reach."""
    if written and (written == read or (len(written) == 1 and next(iter(written))[:2] in reach)):
        return None
    return f'its ports are pinned to {sorted(written)} and {sorted(read)}, not one memory its reader reaches'


def pinned_buffers(ctx, inst) -> Tuple[BufferLocation, ...]:
    """Where the op's graph pins its buffer locations, relative to its anchor: where placement put them."""
    placement = ctx.ir.physical.placements[inst.name]
    col, row = int(placement['col']), int(placement['row'])
    return tuple(
        replace(loc, rel_col=c - col, rel_row=r - row, banks=banks) for loc, (c, r, banks) in placed_buffers(ctx, inst)
    )
