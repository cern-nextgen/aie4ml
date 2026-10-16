# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Graph-aware kernel placement for the AIE backend: the kernels and the hand-overs between them as placement sees
them, and the pass placing them by the constraint solver (`placement_solver`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..op_impls.base import BufferLocation, StreamLocation
from .base import AIEPass
from .estimate import kernel_cycles, leg_cycles
from .shared_buffer import port_reach, static_problem
from .transport.collect import TransportCollector
from .transport.model import Connection

# ---------------------------------------------------------------------------
# Geometry model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortFace:
    """
    One port-bearing face of a rectangular kernel footprint.

    side:
      - 'left'
      - 'right'
      - 'top'
      - 'bottom'

    start/end are inclusive offsets along that face:
      - left/right  -> row offsets [0 .. h-1]
      - top/bottom  -> col offsets [0 .. w-1]
    """

    side: str
    start: int
    end: int


@dataclass
class Rect:
    """Concrete rectangular footprint and placement metadata."""

    w: int
    h: int

    input_face: PortFace
    output_face: PortFace

    extras: Dict[str, Any] = field(default_factory=dict)
    buffer_locations: Optional[Callable[[int], Tuple[BufferLocation, ...]]] = None
    stream_locations: Optional[Callable[[int], Tuple[StreamLocation, ...]]] = None
    _locations: Dict[int, Tuple[BufferLocation, ...]] = field(default_factory=dict, repr=False, compare=False)

    def locations_at(self, anchor_row: int) -> Tuple[BufferLocation, ...]:
        """The op's buffer locations on anchor row `anchor_row`: they depend on the row alone, and the
        search asks for them at every candidate position, so each row is asked of the op once."""
        if self.buffer_locations is None:
            return ()
        if anchor_row not in self._locations:
            self._locations[anchor_row] = tuple(self.buffer_locations(anchor_row))
        return self._locations[anchor_row]


@dataclass
class NodeSpec:
    """Placement-facing adapter for one kernel node."""

    name: str
    rect: Rect
    anchor: Optional[Tuple[int, int]] = None  # local device coordinates
    outputs: frozenset = frozenset()  # its output port groups: their unshared buffers take an MM2S channel
    cycles: int = 0  # one call of its kernels (`estimate.kernel_cycles`)
    arrival: Tuple[int, int] = (0, 0)  # (latency, interval) of its slowest leg from the graph's inputs
    departure: Tuple[int, int] = (0, 0)  # the same of its legs to the graph's outputs


@dataclass(frozen=True)
class EdgeSpec:
    """One logical tensor edge between two kernel nodes."""

    src: str
    dst: str
    tensor: Optional[str] = None
    direct: bool = False
    src_group: str = ''
    dst_group: str = ''
    port_pairs: Tuple[Tuple[int, int], ...] = ()
    dst_tensor: Optional[str] = None  # the consumer's input the edge feeds
    shared: bool = False  # the consumer requires it in shared memory: one buffer both ports reach, no DMA
    shareable: bool = False  # its port pairs may each be one buffer, placed right (`shared_buffer.static_problem`)
    cycles: Tuple[int, int] = (0, 0)  # (latency, interval) the leg takes where it is not one shared buffer


@dataclass
class GraphSpec:
    """The kernels placement places, the edges between them, and the device's rules it places them by."""

    specs: Dict[str, NodeSpec]
    edges: List[EdgeSpec]
    stream_ports: Dict[str, int]  # a switch's ports each way
    dma_channels: int = 0  # a tile DMA's channels each way; 0 counts none
    alternating_horizontal: bool = False  # AIE: odd-row cores reach east, not west (`memory_reach`)
    below: Dict[str, Tuple[Tuple[str, str, int], ...]] = field(default_factory=dict)  # `_transport_edges`


@dataclass
class Placed:
    """Concrete placement of a node in local grid coordinates."""

    name: str
    x: int
    y: int
    rect: Rect
    _banks: Dict[Tuple[str, int], frozenset] = field(default_factory=dict, repr=False, compare=False)
    _reach: Dict[Tuple[str, int], frozenset] = field(default_factory=dict, repr=False, compare=False)

    @cached_property
    def tiles(self) -> frozenset:
        """The tiles it occupies."""
        return frozenset(
            (col, row) for col in range(self.x, self.x + self.rect.w) for row in range(self.y, self.y + self.rect.h)
        )

    def banks(self, group: str, port: int) -> frozenset:
        """The (column, row, banks) its port's buffers are pinned to."""
        key = (group, int(port))
        if key not in self._banks:
            self._banks[key] = frozenset(
                (self.x + location.rel_col, self.y + location.rel_row, tuple(location.banks))
                for location in self.rect.locations_at(self.y)
                if location.port_group == group and location.port == int(port)
            )
        return self._banks[key]

    def reach(self, group: str, port: int, alternating_horizontal: bool) -> frozenset:
        """The tiles whose memory the one kernel reading an input port reaches (`shared_buffer.port_reach`)."""
        key = (group, int(port))
        if key not in self._reach:
            locations = self.rect.locations_at(self.y)
            self._reach[key] = port_reach(locations, group, port, self.x, self.y, alternating_horizontal)
        return self._reach[key]

    def port_tiles(self, group: str, port: int) -> frozenset:
        """The tiles a port streams from or to: its buffers', or its stream's kernel's (`routing.port_tiles`)."""
        tiles = frozenset((col, row) for col, row, _ in self.banks(group, port))
        if tiles or self.rect.stream_locations is None:
            return tiles
        return frozenset(
            (self.x + loc.rel_col, self.y + loc.rel_row)
            for loc in self.rect.stream_locations(self.y)
            if loc.port_group == group and loc.port == int(port)
        )


class PlacementInfeasibleError(ConfigRefused):
    """Raised when no legal placement exists within the current search domain; `proven` False where the placer only
    found none in its time."""

    def __init__(self, message: str, proven: bool = True):
        super().__init__(message)
        self.proven = proven


class PlacementOutranked(ConfigRefused):
    """Raised when no placement ranks the design better than the one the search asks it to beat
    (`ctx.ir.optimizer['beat']`, `ranking.to_beat`); `proven` False where the placer only found none in its time."""

    def __init__(self, message: str, proven: bool = True):
        super().__init__(message)
        self.proven = proven


# ---------------------------------------------------------------------------
# Footprint parsing and geometry
# ---------------------------------------------------------------------------


def _parse_face(
    raw: Optional[Dict[str, Any]],
    *,
    default_side: str,
    w: int,
    h: int,
) -> PortFace:
    side = str((raw or {}).get('side', default_side))
    if side not in ('left', 'right', 'top', 'bottom'):
        raise ValueError(f'Invalid face side: {side!r}')

    limit = h if side in ('left', 'right') else w
    start = int((raw or {}).get('start', 0))
    end = int((raw or {}).get('end', limit - 1))

    if start < 0 or end < start or end >= limit:
        raise ValueError(f'Invalid face span ({start}, {end}) for side={side!r} with limit={limit}.')

    return PortFace(side=side, start=start, end=end)


def _coerce_rect(footprint: Any) -> Rect:
    """
    Convert a kernel variant footprint object into a Rect.

    Required footprint contract:
      footprint.width
      footprint.height

    Optional footprint.extras keys:
      input_face:  {"side": ..., "start": ..., "end": ...}
      output_face: {"side": ..., "start": ..., "end": ...}
      input_side:  "left" | "right" | "top" | "bottom"
      output_side: "left" | "right" | "top" | "bottom"
    """
    w = int(getattr(footprint, 'width'))
    h = int(getattr(footprint, 'height'))
    extras = dict(getattr(footprint, 'extras', {}) or {})

    input_face = _parse_face(
        extras.get('input_face'),
        default_side=str(extras.get('input_side', 'left')),
        w=w,
        h=h,
    )
    output_face = _parse_face(
        extras.get('output_face'),
        default_side=str(extras.get('output_side', 'right')),
        w=w,
        h=h,
    )

    return Rect(
        w=w,
        h=h,
        input_face=input_face,
        output_face=output_face,
        extras=extras,
    )


def _face_abs_box(placed: Placed, face: PortFace) -> Tuple[float, float, float, float]:
    """
    Return the absolute axis-aligned span of a face as:
      (x0, x1, y0, y1)

    A vertical face has x0 == x1 and a y-interval.
    A horizontal face has y0 == y1 and an x-interval.
    """
    x = placed.x
    y = placed.y
    w = placed.rect.w
    h = placed.rect.h

    if face.side == 'left':
        return (x, x, y + face.start, y + face.end)
    if face.side == 'right':
        xr = x + w - 1
        return (xr, xr, y + face.start, y + face.end)
    if face.side == 'top':
        return (x + face.start, x + face.end, y, y)
    if face.side == 'bottom':
        yb = y + h - 1
        return (x + face.start, x + face.end, yb, yb)

    raise ValueError(f'Unsupported face side: {face.side!r}')


def _places(spec: NodeSpec, W: int, H: int, pairable: frozenset) -> List[Tuple[Placed, frozenset]]:
    """Every place of an op on a `W` x `H` array -- its pinned place alone where it has one -- with the input ports,
    (group, port), whose buffers would lie past the array's far edges there: the place requires those read in their
    producer's buffer, which only `pairable` ones may be. Its kernels lie within the array, and every other buffer
    short of its far edges: one may lie west of or below the placement region, where the device starts before it."""
    if spec.anchor is not None:
        candidates = [Placed(spec.name, spec.anchor[0], spec.anchor[1], spec.rect)]
    else:
        rows, columns = range(H - spec.rect.h + 1), range(W - spec.rect.w + 1)
        candidates = [Placed(spec.name, x, y, spec.rect) for y in rows for x in columns]
    found = []
    for p in candidates:
        if p.x < 0 or p.y < 0 or p.x + p.rect.w > W or p.y + p.rect.h > H:
            continue
        past = frozenset(
            (loc.port_group, loc.port)
            for loc in p.rect.locations_at(p.y)
            if p.x + loc.rel_col >= W or p.y + loc.rel_row >= H
        )
        if past <= pairable:
            found.append((p, past))
    if spec.anchor is not None and not found:
        raise PlacementInfeasibleError(f'Invalid fixed anchor for {spec.name}: out of bounds.')
    return found


def share_offsets(
    src: Rect, src_group: str, dst: Rect, dst_group: str, port_pairs, H: int, alternating_horizontal: bool
) -> Dict[Tuple[int, int], set]:
    """Per (producer anchor row, consumer anchor row) of an array `H` rows high, the consumer's column offsets at
    which every port pair hands over one buffer its reading core reaches (`pair_offsets`); none where no placement
    shares them all. An op's buffer locations depend on its anchor row alone, so each pair of rows shares at a few
    column offsets or none."""
    found = None
    for src_port, dst_port in port_pairs:
        aligned = pair_offsets(src, src_group, src_port, dst, dst_group, dst_port, H, alternating_horizontal)
        found = aligned if found is None else {rows: found[rows] & aligned[rows] for rows in found}
    return {rows: offsets for rows, offsets in (found or {}).items() if offsets}


def pair_offsets(
    src: Rect,
    src_group: str,
    src_port: int,
    dst: Rect,
    dst_group: str,
    dst_port: int,
    H: int,
    alternating_horizontal: bool,
    reached: bool = True,
) -> Dict[Tuple[int, int], set]:
    """Per (producer anchor row, consumer anchor row), the consumer's column offsets at which the two ports hand over
    one buffer (`_alignments`): one both pin, or with `reached`, one the consumer's reading core reaches."""
    found = {}
    for src_row in range(H - src.h + 1):
        written = Placed('', 0, src_row, src).banks(src_group, src_port)
        for dst_row in range(H - dst.h + 1):
            reader = Placed('', 0, dst_row, dst)
            reach = reader.reach(dst_group, dst_port, alternating_horizontal) if reached else frozenset()
            found[src_row, dst_row] = _alignments(written, reader.banks(dst_group, dst_port), reach)
    return found


def _alignments(written: frozenset, read: frozenset, reach: frozenset) -> set:
    """The column offsets moving a reader whose port pins `read` and whose core reaches `reach` to share the buffer
    pinned at `written` (`shared_buffer.location_problem`)."""
    if not written or not read:
        return set()
    offsets = set()
    offset = min(written)[0] - min(read)[0]
    if frozenset((col + offset, row, banks) for col, row, banks in read) == written:
        offsets.add(offset)
    if len(written) == 1:
        ((col, row, _),) = written
        offsets.update(col - reached_col for reached_col, reached_row in reach if reached_row == row)
    return offsets


def _buffers(graph: GraphSpec, placed: Dict[str, Placed], shares) -> Dict[str, List[Tuple[int, int, Tuple[int, ...]]]]:
    """Where each op's buffer locations sit once placed, as (column, row, banks) in the order it lists them: where
    it lists them, but an input port that reads its producer's buffer (`shares`, by (op, group, port)) where the
    producer pins that buffer, which its reading core reaches."""
    written = {
        (edge.dst, edge.dst_group, dst_port): (edge.src, edge.src_group, src_port)
        for edge in graph.edges
        for src_port, dst_port in edge.port_pairs
    }
    buffers = {}
    for name, p in placed.items():
        buffers[name] = []
        for loc in p.rect.locations_at(p.y):
            at = (p.x + loc.rel_col, p.y + loc.rel_row, tuple(loc.banks))
            if (name, loc.port_group, loc.port) in shares:
                src, group, port = written[name, loc.port_group, loc.port]
                source = placed[src].banks(group, port)
                if at not in source:  # read where the producer pins it, not pinned there alike
                    (at,) = source
            buffers[name].append(at)
    return buffers


# ---------------------------------------------------------------------------
# The placement graph
# ---------------------------------------------------------------------------


def _transport_entries(ctx) -> list:
    state = ctx.ir.physical.plan.get('_memory_plan_state')
    if state is None:
        raise RuntimeError('Placement requires collected transport entries.')
    return state['entries']


def _copy_cycles(ctx, entry, leg: Connection) -> Tuple[int, int]:
    """(latency, interval) a classified leg takes where it is not one shared buffer (`estimate.leg_cycles`)."""
    return leg_cycles(leg, entry.decision.realization, False, ctx.ir.execution, ctx.device)


def _boundary_cycles(ctx, kernel_names: Sequence[str]) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """By kernel, the (latency, interval) of its slowest leg from the graph's inputs and of its slowest to its
    outputs."""
    boundary = {name: [(0, 0), (0, 0)] for name in kernel_names}
    for entry in _transport_entries(ctx):
        if entry.producer.node is None:
            for connection in entry.consumers:
                ends = boundary[connection.consumer.node.name]
                ends[0] = tuple(map(max, ends[0], _copy_cycles(ctx, entry, connection)))
        elif entry.graph_output:
            leg = Connection(entry.logical_tensor, entry.producer, None)
            ends = boundary[entry.producer.node.name]
            ends[1] = tuple(map(max, ends[1], _copy_cycles(ctx, entry, leg)))
    return {name: tuple(ends) for name, ends in boundary.items()}


def _transport_edges(ctx, kernel_names: Sequence[str]) -> Tuple[List[EdgeSpec], Dict[str, tuple]]:
    """Build placement connectivity from collected semantic transport legs: the kernel-to-kernel edges, and by kernel
    its (direction, group, port) streams from or to below -- a PLIO or a memory tile."""
    entries = _transport_entries(ctx)

    def producer_ports(entry) -> Tuple[int, ...]:
        if entry.unit is not None:
            return tuple(int(port) for port in entry.unit.producer_ports)
        # producer_port_count is the number of ports this entry uses, not the total; use
        # the endpoint's explicit selection directly (e.g. a lone port 1 from a slice view).
        if entry.producer.ports is not None:
            return tuple(int(port) for port in entry.producer.ports)
        return tuple(range(int(entry.producer_port_count)))

    def consumer_ports(entry, consumer) -> Tuple[int, ...]:
        if entry.unit is not None:
            return tuple(int(port) for port in entry.unit.consumer_ports)
        inst = ctx.ir.execution.get(consumer.node.name)
        binding = inst.ports.inputs[consumer.tensor]
        return consumer.selected_ports(int(binding.count))

    kernel_set = set(kernel_names)
    below: Dict[str, set] = {name: set() for name in kernel_names}
    for entry in entries:
        memtile = entry.decision is not None and entry.decision.realization == 'memtile'
        if entry.producer.node is not None and (memtile or entry.graph_output):
            below[entry.producer.node.name].update(
                ('outputs', entry.producer.group, port) for port in producer_ports(entry)
            )
        for connection in entry.consumers:
            if connection.consumer is not None and (memtile or entry.producer.node is None):
                below[connection.consumer.node.name].update(
                    ('inputs', connection.consumer.group, port) for port in consumer_ports(entry, connection.consumer)
                )
    collector = TransportCollector(ctx.ir.execution)
    edges = []
    seen = set()
    for entry in entries:
        src = entry.producer.node
        if src is None:
            continue
        entry_producer_ports = producer_ports(entry)
        for connection in entry.consumers:
            consumer = connection.consumer
            if consumer is None or src.name not in kernel_set or consumer.node.name not in kernel_set:
                continue
            entry_consumer_ports = consumer_ports(entry, consumer)
            direct = entry.decision is not None and entry.decision.realization == 'direct'
            shared = ctx.ir.execution.get(consumer.node.name).input(consumer.tensor).shared_memory
            if shared and not direct:
                raise RuntimeError(
                    f'{entry.logical_tensor}: the edge into {consumer.node.name} must be shared memory, but '
                    'transport did not classify it direct.'
                )
            # Why these ports could never hand over one buffer, wherever they are placed.
            problems = []
            if direct:
                producer_inst, consumer_inst = ctx.ir.execution.get(src.name), ctx.ir.execution.get(consumer.node.name)
                problems = [
                    problem
                    for p_port, c_port in zip(entry_producer_ports, entry_consumer_ports)
                    for problem in (
                        static_problem(
                            entry.unit,
                            producer_inst,
                            entry.producer.group,
                            p_port,
                            consumer_inst,
                            consumer.group,
                            c_port,
                            collector.port_readers(entry.producer.tensor, p_port),
                        ),
                    )
                    if problem
                ]
            if shared and problems:
                raise PlacementInfeasibleError(
                    f'{entry.logical_tensor}: must pass into {consumer.node.name} through shared memory, but '
                    f'{problems[0]}.'
                )
            if direct and len(entry_producer_ports) != len(entry_consumer_ports):
                raise RuntimeError(
                    f'{entry.logical_tensor}: direct placement requires equal producer and consumer port counts.'
                )
            key = (
                src.name,
                consumer.node.name,
                entry.producer.tensor,
                entry_producer_ports,
                entry_consumer_ports,
            )
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                EdgeSpec(
                    src=src.name,
                    dst=consumer.node.name,
                    tensor=entry.producer.tensor,
                    direct=direct,
                    src_group=entry.producer.group,
                    dst_group=consumer.group,
                    port_pairs=tuple(zip(entry_producer_ports, entry_consumer_ports)) if direct else (),
                    dst_tensor=consumer.tensor,
                    shared=shared,
                    shareable=direct and not problems,
                    cycles=_copy_cycles(ctx, entry, connection),
                )
            )
    return edges, {name: tuple(sorted(ends)) for name, ends in below.items()}


def _placement_hint(ctx, node) -> Dict[str, Any]:
    """Where the user pins a kernel (its placement directive)."""
    return node.directives.get('placement') or {}


def placer_time(ctx) -> float:
    """The CP-SAT placer's deterministic time per solve (AIEConfig PlacerTime), in the solver's units -- about a second
    of one worker's search each, the same however loaded the machine."""
    if 'PlacerSeconds' in ctx.aie_config:
        raise ValueError(
            "AIEConfig PlacerSeconds is replaced by PlacerTime, the placer's deterministic time per solve "
            '(default 400).'
        )
    return float(ctx.aie_config.get('PlacerTime', 400))


def instance_rect(inst, row_offset: int) -> Rect:
    """The tiles an instance takes and where it pins its buffers and streams, an anchor row counted from the array's
    row `row_offset`."""
    footprint = inst.variant.footprint(inst.node, inst.config)
    if footprint is None:
        raise RuntimeError(f'{inst.node.name}: kernel variant did not provide a footprint.')
    rect = _coerce_rect(footprint)
    rect.buffer_locations = lambda anchor_row: inst.variant.buffer_locations(
        inst.node, inst.config, int(anchor_row) + row_offset
    )
    rect.stream_locations = lambda anchor_row: inst.variant.stream_locations(
        inst.node, inst.config, int(anchor_row) + row_offset
    )
    return rect


def _build_graph(ctx, col_offset: int, row_offset: int) -> GraphSpec:
    names = [inst.name for inst in ctx.ir.execution]
    edges, below = _transport_edges(ctx, names)
    boundary = _boundary_cycles(ctx, names)
    specs = {}
    for inst in ctx.ir.execution:
        placement_hint = _placement_hint(ctx, inst.node)
        anchor: Optional[Tuple[int, int]] = None
        if placement_hint.get('col') is not None and placement_hint.get('row') is not None:
            anchor = (
                int(placement_hint['col']) - col_offset,
                int(placement_hint['row']) - row_offset,
            )
        specs[inst.name] = NodeSpec(
            name=inst.name,
            rect=instance_rect(inst, row_offset),
            anchor=anchor,
            outputs=frozenset(binding.group for binding in inst.ports.outputs.values()),
            cycles=kernel_cycles(inst, ctx.device),
            arrival=boundary[inst.name][0],
            departure=boundary[inst.name][1],
        )
    device = ctx.device
    return GraphSpec(
        specs=specs,
        edges=edges,
        dma_channels=int(device.tile_dma.channels),
        alternating_horizontal=device.alternating_horizontal,
        stream_ports=dict(device.stream_switch_ports),
        below=below,
    )


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class PlaceKernels(AIEPass):
    """
    Graph-aware AIE kernel placement: the placement the constraint solver (`placement_solver`) ranks best by the
    design's ranking (`ranking.RANKING`, AIEConfig Optimize), deciding which hand-overs are one shared buffer. A design
    the search tries is ranked against its bound (`solve_placement`); the one it keeps, already ranked, is only laid
    out (`tidy_placement`).

    Parameters
    ----------
    lam:
        Weight on vertical edge distance in the layout, the last criterion:
          |Δcol| + lam * |Δrow|

    mu:
        Row-bias weight in the layout:
          + mu * Σ(node.row)
    """

    def __init__(self, lam: float = 1.0, mu: float = 0.05):
        self.name = 'place_kernels'
        self._lam = float(lam)
        self._mu = float(mu)

    def transform(self, model_or_ctx) -> bool:
        from .placement_solver import solve_placement, tidy_placement  # they build on this module's rules
        from .ranking import RANKING, optimize_mode

        ctx = get_backend_context(model_or_ctx)
        device = ctx.device
        col_offset, row_offset = int(device.column_start), int(device.row_start)
        W = int(device.columns) - col_offset
        H = int(device.rows) - row_offset
        if W <= 0 or H <= 0:
            raise ValueError(
                f'Device placement origin ({col_offset}, {row_offset}) is outside '
                f'the {device.columns}x{device.rows} AIE array.'
            )

        graph = _build_graph(ctx, col_offset, row_offset)
        if not graph.specs:
            return False

        # the design search's hand-off, which placement consumes: the IR keeps the placement it makes alone
        time, searched = placer_time(ctx), ctx.ir.optimizer.pop('placement', None)
        first = int(device.preferred_column_start) - col_offset  # the layout packs from the PLIOs' columns
        if searched is None:
            floors, beat = ctx.ir.optimizer.pop('bound', None), ctx.ir.optimizer.pop('beat', None)
            order = RANKING[optimize_mode(ctx)]
            placed, shares, estimate = solve_placement(graph, W, H, time, order, first, floors, beat)
        else:
            start = {
                name: Placed(name, int(at['col']) - col_offset, int(at['row']) - row_offset, graph.specs[name].rect)
                for name, at in searched.items()
            }
            estimate = ctx.ir.optimizer.pop('estimate')
            placed, shares, estimate = tidy_placement(graph, W, H, self._lam, self._mu, first, time, start, estimate)
        buffers = _buffers(graph, placed, shares)
        placements = {
            name: {
                'col': int(p.x + col_offset),
                'row': int(p.y + row_offset),
                'width': int(p.rect.w),  # the tiles reserved, not only the anchor
                'height': int(p.rect.h),
                'buffers': [[col + col_offset, row + row_offset, list(banks)] for col, row, banks in buffers[name]],
            }
            for name, p in placed.items()
        }
        changed = placements != ctx.ir.physical.placements
        ctx.ir.physical.placements = placements
        ctx.ir.physical.estimate = estimate
        return changed
