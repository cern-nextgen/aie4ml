# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Graph-aware kernel placement for the AIE backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..op_impls.base import BufferLocation
from .base import AIEPass
from .shared_buffer import location_problem, static_problem

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

    node: Any
    name: str
    index: int
    rect: Rect
    anchor: Optional[Tuple[int, int]] = None  # local device coordinates


@dataclass(frozen=True)
class EdgeSpec:
    """One logical tensor edge between two kernel nodes."""

    src: str
    dst: str
    tensor: Optional[str] = None
    direct: bool = False
    one_to_one: bool = True  # each port pair its own buffer: no port broadcast or gathered
    src_group: str = ''
    dst_group: str = ''
    port_pairs: Tuple[Tuple[int, int], ...] = ()
    shared: bool = False  # both ports must pin the buffer to the same memory: one buffer, no DMA
    shareable: bool = False  # the ports could hand over one buffer, placed right (`shared_buffer.static_problem`)


@dataclass
class GraphSpec:
    """Kernel-only placement DAG."""

    order: List[str]
    specs: Dict[str, NodeSpec]
    edges: List[EdgeSpec]
    preds: Dict[str, List[str]]
    succs: Dict[str, List[str]]
    _between: Dict[Tuple[str, str], Tuple[EdgeSpec, ...]] = field(default_factory=dict, repr=False, compare=False)

    def edges_between(self, src: str, dst: str) -> Tuple[EdgeSpec, ...]:
        """The edges from `src` to `dst`: the search asks for them of every pair of ops it checks, so they are
        indexed once."""
        if not self._between and self.edges:
            for edge in self.edges:
                self._between[(edge.src, edge.dst)] = self._between.get((edge.src, edge.dst), ()) + (edge,)
        return self._between.get((src, dst), ())


@dataclass
class Placed:
    """Concrete placement of a node in local grid coordinates."""

    name: str
    x: int
    y: int
    rect: Rect
    _banks: Dict[Tuple[str, int], frozenset] = field(default_factory=dict, repr=False, compare=False)

    @cached_property
    def tiles(self) -> frozenset:
        """The tiles it occupies."""
        return frozenset(
            (col, row) for col in range(self.x, self.x + self.rect.w) for row in range(self.y, self.y + self.rect.h)
        )

    def banks(self, group: str, port: int) -> frozenset:
        """The (column, row, banks) its port's buffers are pinned to; asked of a placement at every state below it."""
        key = (group, int(port))
        if key not in self._banks:
            self._banks[key] = frozenset(
                (self.x + location.rel_col, self.y + location.rel_row, tuple(location.banks))
                for location in self.rect.locations_at(self.y)
                if location.port_group == group and location.port == int(port)
            )
        return self._banks[key]

    @cached_property
    def memory(self) -> frozenset:
        """The tiles whose memory its transport-visible buffers use, as it declares them."""
        return frozenset((self.x + loc.rel_col, self.y + loc.rel_row) for loc in self.rect.locations_at(self.y))


class PlacementInfeasibleError(ConfigRefused):
    """Raised when no legal placement exists within the current search domain."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _interval_distance(a0: float, a1: float, b0: float, b1: float) -> float:
    """Distance between two closed 1D intervals."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


# ---------------------------------------------------------------------------
# Footprint parsing
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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


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


def _face_cost(
    a_box: Tuple[float, float, float, float],
    b_box: Tuple[float, float, float, float],
    lam: float,
) -> float:
    """Minimum weighted Manhattan distance between two face boxes."""
    ax0, ax1, ay0, ay1 = a_box
    bx0, bx1, by0, by1 = b_box
    dx = _interval_distance(ax0, ax1, bx0, bx1)
    dy = _interval_distance(ay0, ay1, by0, by1)
    return dx + lam * dy


_UNSHARED_BUFFER_COST = 16.0
"""What a shareable direct edge costs on top of its distance when its two advertised buffer locations do not
coincide: its buffer is then copied by a DMA rather than shared, which is worth a long detour to avoid."""


def _edge_cost(edge: EdgeSpec, src: Placed, dst: Placed, lam: float) -> float:
    cost = _edge_cost_between_placements(src, dst, lam)
    if not edge.shareable:
        return cost
    for src_port, dst_port in edge.port_pairs:
        written = _absolute_bank_locations(src, group=edge.src_group, port=src_port)
        read = _absolute_bank_locations(dst, group=edge.dst_group, port=dst_port)
        if written and read and written != read:
            cost += _UNSHARED_BUFFER_COST
    return cost


def _edge_cost_between_placements(src: Placed, dst: Placed, lam: float) -> float:
    return _face_cost(
        _face_abs_box(src, src.rect.output_face),
        _face_abs_box(dst, dst.rect.input_face),
        lam,
    )


def _absolute_bank_locations(
    placed: Placed,
    *,
    group: str,
    port: int,
) -> frozenset:
    return placed.banks(group, port)


def _aliased_tiles(producer: Placed, consumer: Placed, graph: GraphSpec) -> set[Tuple[int, int]]:
    """The tiles where a direct edge from `producer` to `consumer` puts one buffer both ports name."""
    tiles = set()
    for edge in graph.edges_between(producer.name, consumer.name):
        if not (edge.direct and edge.one_to_one):
            continue
        for src_port, dst_port in edge.port_pairs:
            source = _absolute_bank_locations(producer, group=edge.src_group, port=src_port)
            if source and source == _absolute_bank_locations(consumer, group=edge.dst_group, port=dst_port):
                tiles.update((col, row) for col, row, _ in source)
    return tiles


def _shared_edges_coincide(a: Placed, b: Placed, graph: GraphSpec) -> bool:
    """Whether every edge between the two that must be shared pins both of its ports to the same
    memory -- the same (column, row, banks) -- so the compiler has one buffer to place, and no DMA."""
    for edge in (*graph.edges_between(a.name, b.name), *graph.edges_between(b.name, a.name)):
        if not edge.shared:
            continue
        src, dst = (a, b) if edge.src == a.name else (b, a)
        for src_port, dst_port in edge.port_pairs:
            written = _absolute_bank_locations(src, group=edge.src_group, port=src_port)
            if location_problem(written, _absolute_bank_locations(dst, group=edge.dst_group, port=dst_port)):
                return False
    return True


def _placements_conflict(a: Placed, b: Placed, graph: GraphSpec) -> bool:
    """Two ops conflict when they share a tile, or when one keeps buffers in a tile's memory that the
    other occupies or also keeps buffers in -- unless it is the one buffer a direct edge between them
    shares there."""
    if a.tiles & b.tiles:
        return True
    clash = (a.memory & b.memory) | ((a.memory - a.tiles) & b.tiles) | ((b.memory - b.tiles) & a.tiles)
    return bool(clash) and bool(clash - _aliased_tiles(a, b, graph) - _aliased_tiles(b, a, graph))


def _in_bounds(p: Placed, W: int, H: int) -> bool:
    return (
        p.x >= 0
        and p.y >= 0
        and p.x + p.rect.w <= W
        and p.y + p.rect.h <= H
        # A buffer may sit west of or below the placement region -- the device starts before it -- but
        # never past its far edges, where the device ends.
        and all(col < W and row < H for col, row in p.memory)
    )


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _transport_edges(ctx, kernel_names: Sequence[str]) -> List[EdgeSpec]:
    """Build placement connectivity from collected semantic transport legs."""
    state = ctx.ir.physical.plan.get('_memory_plan_state')
    if state is None:
        raise RuntimeError('Placement requires collected transport entries.')

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
    producer_port_uses = {}
    for entry in state['entries']:
        if entry.producer.node is None:
            continue
        for port in producer_ports(entry):
            key = (entry.producer.node.name, entry.producer.tensor, int(port))
            producer_port_uses[key] = producer_port_uses.get(key, 0) + 1

    edges = []
    seen = set()
    for entry in state['entries']:
        src = entry.producer.node
        if src is None:
            continue
        entry_producer_ports = producer_ports(entry)
        producer_exclusive = all(
            producer_port_uses[(src.name, entry.producer.tensor, port)] == 1 for port in entry_producer_ports
        )
        for connection in entry.consumers:
            consumer = connection.consumer
            if consumer is None or src.name not in kernel_set or consumer.node.name not in kernel_set:
                continue
            entry_consumer_ports = consumer_ports(entry, consumer)
            direct = entry.decision is not None and entry.decision.realization == 'direct'
            # one buffer per port pair only: no port read by another leg, broadcast or gathered
            one_to_one = direct and producer_exclusive and entry.unit.one_to_one()
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
                            ctx,
                            entry.producer.tensor,
                            producer_inst,
                            entry.producer.group,
                            p_port,
                            consumer_inst,
                            consumer.group,
                            c_port,
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
                    one_to_one=one_to_one,
                    src_group=entry.producer.group,
                    dst_group=consumer.group,
                    port_pairs=tuple(zip(entry_producer_ports, entry_consumer_ports)) if direct else (),
                    shared=shared,
                    shareable=direct and one_to_one and not problems,
                )
            )
    return edges


def _topological_order(
    names: Sequence[str],
    edges: Sequence[EdgeSpec],
    stable_index: Dict[str, int],
) -> List[str]:
    indeg = {n: 0 for n in names}
    succs: Dict[str, List[str]] = {n: [] for n in names}

    for e in edges:
        if e.src in indeg and e.dst in indeg:
            indeg[e.dst] += 1
            succs[e.src].append(e.dst)

    ready = sorted([n for n, d in indeg.items() if d == 0], key=lambda n: stable_index[n])
    order: List[str] = []

    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(succs[n], key=lambda x: stable_index[x]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
                ready.sort(key=lambda x: stable_index[x])

    if len(order) != len(names):
        raise RuntimeError(f'placement: the kernel graph has a cycle through {sorted(set(names) - set(order))}.')
    return order


def _placement_hint(ctx, node) -> Dict[str, Any]:
    """Where a kernel is pinned: the user's placement directive, else the design search's choice."""
    return node.directives.get('placement') or ctx.ir.optimizer.get('placement', {}).get(node.name, {})


def _build_graph(ctx, col_offset: int, row_offset: int) -> GraphSpec:
    specs: Dict[str, NodeSpec] = {}
    stable_index: Dict[str, int] = {}

    for idx, inst in enumerate(ctx.ir.execution):
        node = inst.node
        footprint = inst.variant.footprint(node, inst.config)
        if footprint is None:
            raise RuntimeError(f'{node.name}: kernel variant did not provide a footprint.')

        rect = _coerce_rect(footprint)
        rect.buffer_locations = lambda anchor_row, inst=inst, node=node: inst.variant.buffer_locations(
            node, inst.config, int(anchor_row) + row_offset
        )

        placement_hint = _placement_hint(ctx, node)
        anchor: Optional[Tuple[int, int]] = None
        if placement_hint.get('col') is not None and placement_hint.get('row') is not None:
            anchor = (
                int(placement_hint['col']) - col_offset,
                int(placement_hint['row']) - row_offset,
            )

        specs[node.name] = NodeSpec(
            node=node,
            name=node.name,
            index=idx,
            rect=rect,
            anchor=anchor,
        )
        stable_index[node.name] = idx

    edges = _transport_edges(ctx, list(specs))

    preds = {name: [] for name in specs}
    succs = {name: [] for name in specs}

    for e in edges:
        preds[e.dst].append(e.src)
        succs[e.src].append(e.dst)

    order = _topological_order(list(specs), edges, stable_index)

    return GraphSpec(
        order=order,
        specs=specs,
        edges=edges,
        preds=preds,
        succs=succs,
    )


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class _State:
    """A partial placement: its cost, its compactness (the sum of col * H + row over its ops: smaller packs the
    array tighter, from the west and the bottom), its ops, and which ops claim each tile, by its core or memory."""

    cost: float
    compactness: int
    placed: Dict[str, Placed]
    claims: Dict[Tuple[int, int], Tuple[str, ...]]

    def extend(self, p: Placed, cost: float, compactness: int) -> '_State':
        """This placement with `p` added, which brings it to `cost` and `compactness`."""
        claims = dict(self.claims)
        for cell in p.tiles | p.memory:
            claims[cell] = claims.get(cell, ()) + (p.name,)
        return _State(cost, compactness, {**self.placed, p.name: p}, claims)

    def fits(self, p: Placed, graph: GraphSpec) -> bool:
        """Whether `p` is legal beside the ops placed: only an op claiming one of its tiles can conflict with it,
        and only a neighbour can share a buffer with it."""
        if any(_placements_conflict(p, self.placed[name], graph) for name in self._claimants(p)):
            return False
        return all(
            _shared_edges_coincide(p, self.placed[name], graph)
            for name in (*graph.preds[p.name], *graph.succs[p.name])
            if name in self.placed
        )

    def _claimants(self, p: Placed) -> set:
        return {name for cell in p.tiles | p.memory for name in self.claims.get(cell, ())}


def _places(spec: NodeSpec, W: int, H: int) -> List[List[Placed]]:
    """Every in-bounds place of an op, a list per anchor row, west to east; its pinned place alone where it has one."""
    if spec.anchor is not None:
        pinned = Placed(spec.name, spec.anchor[0], spec.anchor[1], spec.rect)
        if not _in_bounds(pinned, W, H):
            raise PlacementInfeasibleError(f'Invalid fixed anchor for {spec.name}: out of bounds.')
        return [[pinned]]
    xs, ys = range(W - spec.rect.w + 1), range(H - spec.rect.h + 1)
    rows = ([Placed(spec.name, x, y, spec.rect) for x in xs] for y in ys)
    return [places for row in rows if (places := [p for p in row if _in_bounds(p, W, H)])]


def _share_slots(graph: GraphSpec, H: int) -> Dict[Tuple[str, str], Dict[int, List[Tuple[int, int]]]]:
    """Where an op's neighbour may sit to share a buffer of a shareable edge between them: by (op, neighbour) and
    the op's anchor row, the neighbour's (column offset, anchor row) places. An op's buffer locations depend on its
    anchor row alone, so each port's pair of rows lines its two buffers up at one column offset or none."""
    slots: Dict[Tuple[str, str], Dict[int, List[Tuple[int, int]]]] = {}
    for edge in graph.edges:
        if not edge.shareable:
            continue
        src, dst = graph.specs[edge.src].rect, graph.specs[edge.dst].rect
        for src_row in range(H - src.h + 1):
            written = [Placed(edge.src, 0, src_row, src).banks(edge.src_group, port) for port, _ in edge.port_pairs]
            for dst_row in range(H - dst.h + 1):
                read = [Placed(edge.dst, 0, dst_row, dst).banks(edge.dst_group, port) for _, port in edge.port_pairs]
                for offset in sorted({_alignment(*buffers) for buffers in zip(written, read)} - {None}):
                    slots.setdefault((edge.src, edge.dst), {}).setdefault(src_row, []).append((offset, dst_row))
                    slots.setdefault((edge.dst, edge.src), {}).setdefault(dst_row, []).append((-offset, src_row))
    return slots


def _alignment(written: frozenset, read: frozenset) -> Optional[int]:
    """The column offset that moves a read buffer's locations onto its written one's, or None."""
    if not written or not read:
        return None
    offset = min(written)[0] - min(read)[0]
    return offset if frozenset((col + offset, row, banks) for col, row, banks in read) == written else None


def _placement_order(graph: GraphSpec, slots: Dict[Tuple[str, str], Any]) -> List[str]:
    """The order the beam places ops in: the pinned ones, then each time the op that may share the most buffers with
    the ops placed (`_share_slots`), then with the most neighbours among them, earliest in dataflow first. A consumer
    that may share its producer's buffer thus follows it."""
    shares: Dict[Tuple[str, str], int] = {}
    for edge in graph.edges:
        pairs = len(edge.port_pairs) if (edge.src, edge.dst) in slots else 0
        shares[edge.src, edge.dst] = shares.get((edge.src, edge.dst), 0) + pairs
        shares[edge.dst, edge.src] = shares.get((edge.dst, edge.src), 0) + pairs
    neighbours = {name: set(graph.preds[name]) | set(graph.succs[name]) for name in graph.specs}
    position = {name: i for i, name in enumerate(graph.order)}

    order = [name for name in graph.order if graph.specs[name].anchor is not None]
    placed = set(order)
    while len(order) < len(graph.order):

        def affinity(name: str) -> Tuple[int, int, int]:
            near = neighbours[name] & placed
            return (sum(shares[name, other] for other in near), len(near), -position[name])

        name = max((name for name in graph.order if name not in placed), key=affinity)
        order.append(name)
        placed.add(name)
    return order


def _keep(children: List[tuple], width: int) -> List[tuple]:
    """The partial placements the beam keeps of `children` (cost, compactness, ...): those no other is both cheaper
    and more compact than, from the cheapest to the most compact, spread evenly when there are more than `width`;
    then the cheapest of the rest. A cost-led placement can leave gaps no later op fits in; the compact ones keep a
    packed array placeable."""
    children.sort(key=lambda child: child[:5])
    front, rest = [], []
    for child in children:
        (front if not front or child[1] < front[-1][1] else rest).append(child)
    if len(front) > width:
        return [front[round(i * (len(front) - 1) / (width - 1))] for i in range(width)]
    return front + rest[: width - len(front)]


def _beam_place(graph: GraphSpec, W: int, H: int, lam: float, mu: float, width: int) -> Dict[str, Placed]:
    """
    Place the ops one at a time (`_placement_order`), keeping `width` partial placements (`_keep`). Each grows by
    up to three places per anchor row: the westmost legal one; the cheapest legal one, by the edges to the ops placed
    and the row bias; and the westmost legal one that leaves each neighbour still to place a free place sharing its
    buffers (`_share_slots`), on whichever side that is. The cheapest complete placement wins.
    """
    places = {name: _places(spec, W, H) for name, spec in graph.specs.items()}
    slots = _share_slots(graph, H)
    incident = {name: [edge for edge in graph.edges if name in (edge.src, edge.dst)] for name in graph.specs}

    def added_cost(p: Placed, placed: Dict[str, Placed]) -> float:
        cost = mu * p.y
        for edge in incident[p.name]:
            if edge.src == p.name and edge.dst in placed:
                cost += _edge_cost(edge, p, placed[edge.dst], lam)
            elif edge.dst == p.name and edge.src in placed:
                cost += _edge_cost(edge, placed[edge.src], p, lam)
        return cost

    def leaves_room(p: Placed, state: _State, partners: List[str]) -> bool:
        """Whether each of `partners` has an in-bounds place sharing `p`'s buffers whose tiles nothing holds yet."""
        held = set(state.claims) | p.tiles
        return all(
            any(
                _in_bounds(q := Placed(other, p.x + offset, row, graph.specs[other].rect), W, H) and not q.tiles & held
                for offset, row in slots[p.name, other].get(p.y, ())
            )
            for other in partners
        )

    states = [_State(0.0, 0, {}, {})]
    for name in _placement_order(graph, slots):
        neighbours = sorted(set(graph.preds[name]) | set(graph.succs[name]))
        sharers = [other for other in neighbours if (name, other) in slots]
        costs: Dict[tuple, Dict[Tuple[int, int], float]] = {}  # by the places of the op's placed neighbours
        children = []
        for rank, state in enumerate(states):
            near = tuple((n, state.placed[n].x, state.placed[n].y) for n in neighbours if n in state.placed)
            known = costs.setdefault(near, {})

            def cost(p: Placed) -> float:
                if (p.x, p.y) not in known:
                    known[p.x, p.y] = added_cost(p, state.placed)
                return known[p.x, p.y]

            partners = [other for other in sharers if other not in state.placed]
            grown = {}
            for row in places[name]:
                westmost = next((p for p in row if state.fits(p, graph)), None)
                if westmost is None:
                    continue
                cheapest = next(p for p in sorted(row, key=lambda p: (cost(p), p.x)) if state.fits(p, graph))
                grown.update({(p.x, p.y): p for p in (westmost, cheapest)})
                if partners:
                    roomy = next((p for p in row if leaves_room(p, state, partners) and state.fits(p, graph)), None)
                    if roomy is not None:
                        grown[roomy.x, roomy.y] = roomy
            children.extend(
                (state.cost + cost(p), state.compactness + p.x * H + p.y, rank, p.x, p.y, state, p)
                for p in grown.values()
            )
        if not children:
            if graph.specs[name].anchor is not None:
                raise PlacementInfeasibleError(f'Invalid fixed anchor for {name}: conflicts with another anchor.')
            raise PlacementInfeasibleError(
                f'{name}: no legal place left beside the {len(states[0].placed)} ops placed before it, in any of '
                f'the {len(states)} partial placements kept.'
            )
        states = [state.extend(p, cost, compactness) for cost, compactness, *_, state, p in _keep(children, width)]
    return min(states, key=lambda state: (state.cost, state.compactness)).placed


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class PlaceKernels(AIEPass):
    """
    Graph-aware AIE kernel placement.

    Parameters
    ----------
    lam:
        Weight on vertical edge distance in the objective:
          |Δcol| + lam * |Δrow|

    mu:
        Row-bias weight in the objective:
          + mu * Σ(node.row)

    width:
        Partial placements the beam search keeps (`_beam_place`), at least 2: the cheapest and the most compact.
    """

    def __init__(self, lam: float = 1.0, mu: float = 0.05, width: int = 16):
        if int(width) < 2:
            raise ValueError('PlaceKernels: width must be at least 2, to keep the cheapest and the most compact.')
        self.name = 'place_kernels'
        self._lam = float(lam)
        self._mu = float(mu)
        self._width = int(width)

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        device = ctx.device
        region, preferred = int(device.column_start), int(device.preferred_column_start)
        pinned = [_placement_hint(ctx, inst.node).get('col') for inst in ctx.ir.execution]
        # Start next to the PLIOs when the design fits there, else use the whole region.
        if preferred > region and all(col is None or int(col) >= preferred for col in pinned):
            try:
                placements = self._place(ctx, preferred)
            except PlacementInfeasibleError:
                placements = self._place(ctx, region)
        else:
            placements = self._place(ctx, region)
        if placements is None:
            return False
        changed = placements != ctx.ir.physical.placements
        ctx.ir.physical.placements = placements
        return changed

    def _place(self, ctx, col_offset: int) -> Optional[Dict[str, Dict[str, int]]]:
        """Every kernel's place in the region from column `col_offset`, or None when there is no kernel."""
        device = ctx.device
        row_offset = int(device.row_start)
        W = int(device.columns) - col_offset
        H = int(device.rows) - row_offset
        if W <= 0 or H <= 0:
            raise ValueError(
                f'Device placement origin ({col_offset}, {row_offset}) is outside '
                f'the {device.columns}x{device.rows} AIE array.'
            )

        graph = _build_graph(ctx, col_offset, row_offset)
        if not graph.specs:
            return None

        placed = _beam_place(graph, W, H, self._lam, self._mu, self._width)
        return {
            name: {
                'col': int(p.x + col_offset),
                'row': int(p.y + row_offset),
                'width': int(p.rect.w),  # the tiles reserved, not only the anchor
                'height': int(p.rect.h),
            }
            for name, p in placed.items()
        }
