"""A design's first-inference latency along its dataflow: kernels named `<kernel graph>.kk[<index>]` as the compiler
names them, joined by the buffer handoffs of the physical plan (`graph_edges`) -- shared memory, DMA copy, or staged
through a memory tile -- each costed by its bytes over the path's bandwidth plus its `Transport` setup."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..aie_types import width_for_format
from ..op_impls.utils.precision import storage_bytes_for_spec
from ..passes.shared_buffer import DMA, SHARED_MEMORY, STREAM
from ..passes.utils import sanitize_identifier
from .chain import Link, Timeline, first_call

STAGING = 'memtile:'  # a buffer staged in a memory tile, as `graph_edges` names it
_PORT = re.compile(r'^(?P<graph>\w+)\.(?P<group>\w+)\[(?P<port>\d+)\]$')
_KERNEL = re.compile(r'^(kk\[\d+\])\.')


@dataclass(frozen=True)
class Edge:
    """A buffer handed from `source` to `target`, each a kernel, or None at the design's boundary."""

    source: Optional[str]
    target: Optional[str]
    bytes: int
    shared: bool  # producer and consumer use the same buffer: only a lock changes hands


@dataclass(frozen=True)
class Chain:
    """One cascade chain (a single kernel is a chain of one): its kernels, timelines and wrapper cycles, in chain
    order."""

    blocks: Sequence[str]
    timelines: Sequence[Timeline]
    wrappers: Sequence[float]


@dataclass(frozen=True)
class Transport:
    """A target's transfer terms, in AIE cycles: the bandwidth of a stream and of a boundary port (its PLIO), and the
    setup of each kind of handoff."""

    bytes_per_cycle: float
    boundary_bytes_per_cycle: float
    boundary_in: float
    boundary_out: float
    lock: float
    dma: float


def graph_edges(execution, physical) -> List[Edge]:
    """The buffer handoffs of the physical plan, producer kernels to consumer kernels, sized by the consumer's port
    tile; a buffer staged in a memory tile becomes the node `STAGING` + its name, copied into and out of."""
    graphs = {sanitize_identifier(inst.name): inst for inst in execution}

    def port(endpoint: str, direction: str):
        """The kernels behind a kernel graph's port, and the bytes of its buffer; None at the boundary."""
        match = _PORT.match(endpoint)
        if match is None or match['graph'] not in graphs:
            return None
        inst = graphs[match['graph']]
        bindings = getattr(inst.ports, direction)
        ((tensor, binding),) = [(t, b) for t, b in bindings.items() if b.group == match['group']]
        kernels = [f"{match['graph']}.{_KERNEL.match(e)[1]}" for e in binding.endpoints[int(match['port'])]]
        role = 'output' if direction == 'outputs' else inst.node.roles[tensor]
        size = math.prod(inst.port_views[tensor].tile) * storage_bytes_for_spec(inst.config.precision[role])
        return kernels, size

    out = []
    for edge in physical.plan['direct_edges']:
        source, target = port(edge['source'], 'outputs'), port(edge['target'], 'inputs')
        if source is None and target is None:
            raise ValueError(f"{edge['source']} -> {edge['target']}: joins no kernel graph.")
        size = (target or source)[1]
        realization = edge.get('realization')
        if source and target and realization == STREAM:
            raise NotImplementedError(f"{edge['source']} -> {edge['target']}: stream handoffs are not composed yet.")
        if source and target and realization not in (SHARED_MEMORY, DMA):
            raise ValueError(f"{edge['source']} -> {edge['target']}: the plan records no known realization.")
        shared = realization == SHARED_MEMORY
        out += [
            Edge(producer, consumer, size, shared)
            for producer in (source[0] if source else [None])
            for consumer in (target[0] if target else [None])
        ]
    for buffer in physical.plan['buffers']:
        staged = STAGING + buffer['name']
        for writer in buffer['writers']:
            source = port(writer['source'], 'outputs')
            if source is None:
                out.append(Edge(None, staged, _boundary_bytes(writer), False))
            else:
                out += [Edge(kernel, staged, source[1], False) for kernel in source[0]]
        for reader in buffer['readers']:
            target = port(reader['target'], 'inputs')
            if target is None:
                out.append(Edge(staged, None, _boundary_bytes(reader), False))
            else:
                out += [Edge(staged, kernel, target[1], False) for kernel in target[0]]
    return out


def _boundary_bytes(leg: dict) -> int:
    """The bytes a boundary port moves into or out of a staged buffer: its slice of the tensor."""
    return math.prod(leg['descriptor']['io_tiling_dimension']) * ((width_for_format(leg['dtype']['format']) + 7) // 8)


def latency(chains: Sequence[Chain], handoffs: Sequence[Edge], link: Link, transport: Transport) -> float:
    """Cycles from the first input beat to the last output of the first inference. A buffer staged in a memory tile
    is handed on once every writer's copy into it has arrived."""
    chain_of: Dict[str, Chain] = {block: chain for chain in chains for block in chain.blocks}
    ends: Dict[int, float] = {}

    def done(block: str) -> float:
        return ready(block) if block.startswith(STAGING) else end(chain_of[block])

    def ready(block: str) -> float:
        times = []
        for edge in handoffs:
            if edge.target != block:
                continue
            if edge.source is None:
                times.append(transport.boundary_in + edge.bytes / transport.boundary_bytes_per_cycle)
            else:
                handoff = transport.lock if edge.shared else transport.dma + edge.bytes / transport.bytes_per_cycle
                times.append(done(edge.source) + handoff)
        return max(times, default=0.0)

    def end(chain: Chain) -> float:
        if id(chain) not in ends:
            starts = [ready(block) for block in chain.blocks]
            ends[id(chain)] = first_call(chain.timelines, chain.wrappers, link, starts)
        return ends[id(chain)]

    outputs = [
        done(edge.source) + transport.boundary_out + edge.bytes / transport.boundary_bytes_per_cycle
        for edge in handoffs
        if edge.target is None
    ]
    if not outputs:
        raise ValueError('the design has no boundary output.')
    return max(outputs)
