"""How a direct edge is realised, and whether it can be one shared buffer where it must be.

A direct stream edge moves data on core streams. A direct buffer edge joins two buffer ports, and every
op pins the buffers of its buffer ports where its locations say, one copy per bank, so the AIE compiler
realises it as one buffer both kernels reach exactly where the producer's and the consumer's locations
coincide, and as a DMA copy between two pinned buffers anywhere else. An edge the execution graph marks
`shared_memory` must be the former, which is known before the compiler runs:

- each port is a buffer port bound to exactly one kernel port, and the value has no other reader, so
  nothing asks for a copy (fanout, multicast, a graph output, a view);
- once both ops are placed, the two ports' locations coincide.

The first does not depend on placement, and the placer refuses it outright; it asks the second while it
searches. The physical verifier asks both again of the finished plan.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..op_impls.common_types import PORT_KIND_BUFFER

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


def static_problem(
    ctx, tensor: str, producer, p_group: str, p_port: int, consumer, c_group: str, c_port: int
) -> Optional[str]:
    """Why no placement could let these ports share one buffer, or None."""
    execution = ctx.ir.execution
    for inst, group, port, direction in ((producer, p_group, p_port, 'outputs'), (consumer, c_group, c_port, 'inputs')):
        problem = port_problem(inst, group, port, direction)
        if problem:
            return problem
    others = other_readers(execution, tensor, consumer.name)
    if others:
        return f'{tensor!r} is also read by {others}'
    return None


def other_readers(execution, tensor: str, consumer: str) -> List[str]:
    """What reads `tensor` besides the kernel graph `consumer`: other kernels, views over it, the graph boundary."""
    readers = sorted(
        {inst.name for inst in execution if any(item.tensor == tensor for item in inst.inputs)} - {consumer}
    )
    views = [v.name for v in execution.values.values() if v.view is not None and tensor in v.view.sources]
    return readers + views + (['the graph boundary'] if tensor in execution.graph_outputs else [])


def pinned_locations(inst, group: str, port: int, col: int, row: int) -> Set[Tuple[int, int, Tuple[int, ...]]]:
    """Where an op placed at (col, row) pins one port's buffer: absolute (column, row, banks)."""
    return {
        (col + loc.rel_col, row + loc.rel_row, tuple(loc.banks))
        for loc in inst.variant.buffer_locations(inst.node, inst.config, row)
        if loc.port_group == group and loc.port == int(port)
    }


def location_problem(written: Set, read: Set) -> Optional[str]:
    """Why the two ports' pinned buffer locations are not one memory, or None."""
    if not written or written != read:
        return f'its ports are pinned to {sorted(written)} and {sorted(read)}, not one memory'
    return None
