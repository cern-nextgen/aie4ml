# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Graph-aware kernel placement for the AIE backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from itertools import permutations
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..op_impls.base import BufferLocation
from .base import AIEPass
from .shared_buffer import location_problem, static_problem

log = logging.getLogger(__name__)


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
    x_range: Optional[Tuple[int, int]] = None
    y_range: Optional[Tuple[int, int]] = None
    _domains: Dict[Tuple[PortFace, int, int], Tuple[float, float, float, float]] = field(
        default_factory=dict, repr=False, compare=False
    )


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


@dataclass(frozen=True)
class BranchBand:
    child: str
    names: Tuple[str, ...]
    inner_height: int


@dataclass(frozen=True)
class PlacementHeuristics:
    """
    Search-order heuristics only.

    These do not change legality or the final objective; they only bias the
    order in which candidates are explored.
    """

    low_row_weight: float = 0.05
    rightward_progress_weight: float = 0.25


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
      row_parity: 0 for even starting rows or 1 for odd starting rows
    """
    w = int(getattr(footprint, 'width'))
    h = int(getattr(footprint, 'height'))
    extras = dict(getattr(footprint, 'extras', {}) or {})
    if extras.get('row_parity') is not None:
        row_parity = int(extras['row_parity'])
        if row_parity not in (0, 1):
            raise ValueError(f'Invalid row_parity: {row_parity}; expected 0 or 1.')
        extras['row_parity'] = row_parity

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


def _face_local_center(rect: Rect, face: PortFace) -> Tuple[float, float]:
    """Local center point of a face span."""
    mid = 0.5 * (face.start + face.end)

    if face.side == 'left':
        return (0.0, mid)
    if face.side == 'right':
        return (float(rect.w - 1), mid)
    if face.side == 'top':
        return (mid, 0.0)
    if face.side == 'bottom':
        return (mid, float(rect.h - 1))

    raise ValueError(f'Unsupported face side: {face.side!r}')


def _face_abs_center(placed: Placed, face: PortFace) -> Tuple[float, float]:
    lx, ly = _face_local_center(placed.rect, face)
    return (placed.x + lx, placed.y + ly)


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


def _feasible(p: Placed, placed: Dict[str, Placed], graph: GraphSpec, W: int, H: int) -> bool:
    if not _in_bounds(p, W, H):
        return False
    return all(not _placements_conflict(p, q, graph) and _shared_edges_coincide(p, q, graph) for q in placed.values())


def _possible_face_domain(
    spec: NodeSpec,
    face: PortFace,
    W: int,
    H: int,
) -> Tuple[float, float, float, float]:
    """
    Bounding box of the union of all possible absolute face positions for `spec`.

    This intentionally ignores occupancy conflicts and uses only bounds/anchors.
    That makes it an admissible lower-bound domain for cut-edge estimates. It depends on the op and the grid
    alone, and the bound asks for it at every state, so each is computed once.
    """
    key = (face, W, H)
    if key not in spec._domains:
        spec._domains[key] = _face_domain(spec, face, W, H)
    return spec._domains[key]


def _face_domain(spec: NodeSpec, face: PortFace, W: int, H: int) -> Tuple[float, float, float, float]:
    rect = spec.rect

    if spec.anchor is not None:
        ax, ay = spec.anchor
        return _face_abs_box(Placed(spec.name, ax, ay, rect), face)

    max_x = W - rect.w
    max_y = H - rect.h
    if max_x < 0 or max_y < 0:
        raise RuntimeError(f'Node {spec.name} footprint ({rect.w}x{rect.h}) does not fit device ({W}x{H}).')

    min_x = 0 if spec.x_range is None else spec.x_range[0]
    max_x = max_x if spec.x_range is None else spec.x_range[1]
    min_y = 0 if spec.y_range is None else spec.y_range[0]
    max_y = max_y if spec.y_range is None else spec.y_range[1]
    if max_x < min_x or max_y < min_y:
        raise RuntimeError(f'Node {spec.name} has no legal placement domain within device ({W}x{H}).')

    if face.side == 'left':
        return (float(min_x), float(max_x), float(min_y + face.start), float(max_y + face.end))
    if face.side == 'right':
        return (
            float(min_x + rect.w - 1),
            float(max_x + rect.w - 1),
            float(min_y + face.start),
            float(max_y + face.end),
        )
    if face.side == 'top':
        return (float(min_x + face.start), float(max_x + face.end), float(min_y), float(max_y))
    if face.side == 'bottom':
        return (
            float(min_x + face.start),
            float(max_x + face.end),
            float(min_y + rect.h - 1),
            float(max_y + rect.h - 1),
        )

    raise ValueError(f'Unsupported face side: {face.side!r}')


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
        if rect.extras.get('row_parity') is not None:
            rect.extras['row_parity'] = (int(rect.extras['row_parity']) - row_offset) % 2

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
# Cost model and lower bound
# ---------------------------------------------------------------------------


def _edge_lower_bound(
    edge: EdgeSpec,
    graph: GraphSpec,
    placed: Dict[str, Placed],
    W: int,
    H: int,
    lam: float,
) -> float:
    """
    Admissible lower bound for a single edge.

    - both placed: exact edge cost
    - one placed: min cost from exact placed face to the bounded domain of the
      unplaced endpoint's face
    - neither placed: 0
    """
    src_p = placed.get(edge.src)
    dst_p = placed.get(edge.dst)

    if src_p is not None and dst_p is not None:
        return _edge_cost(edge, src_p, dst_p, lam)

    if src_p is not None:
        dst_spec = graph.specs[edge.dst]
        return _face_cost(
            _face_abs_box(src_p, src_p.rect.output_face),
            _possible_face_domain(dst_spec, dst_spec.rect.input_face, W, H),
            lam,
        )

    if dst_p is not None:
        src_spec = graph.specs[edge.src]
        return _face_cost(
            _possible_face_domain(src_spec, src_spec.rect.output_face, W, H),
            _face_abs_box(dst_p, dst_p.rect.input_face),
            lam,
        )

    return 0.0


def _lower_bound(
    graph: GraphSpec,
    placed: Dict[str, Placed],
    W: int,
    H: int,
    lam: float,
    mu: float,
) -> float:
    edge_cost = sum(_edge_lower_bound(e, graph, placed, W, H, lam) for e in graph.edges)
    row_bias = sum(mu * p.y for p in placed.values())
    return edge_cost + row_bias


def _full_cost(
    graph: GraphSpec,
    placed: Dict[str, Placed],
    lam: float,
    mu: float,
) -> float:
    edge_cost = sum(_edge_cost(e, placed[e.src], placed[e.dst], lam) for e in graph.edges)
    row_bias = sum(mu * p.y for p in placed.values())
    return edge_cost + row_bias


# ---------------------------------------------------------------------------
# Search heuristics
# ---------------------------------------------------------------------------


def _placed_neighbor_count(graph: GraphSpec, name: str, placed_names: set[str]) -> int:
    return sum(1 for n in graph.preds[name] + graph.succs[name] if n in placed_names)


def _select_next_node(graph: GraphSpec, placed: Dict[str, Placed]) -> str:
    """
    Frontier-first branching:
      0. when nothing is placed yet, start from a graph source
      1. maximize number of already-placed neighbors
      2. maximize total degree
      3. maximize area (larger boxes earlier tend to prune sooner)
      4. stabilize with the original logical order
    """
    placed_names = set(placed)
    candidates = [name for name in graph.order if name not in placed_names]

    if not placed_names:
        sources = [name for name in candidates if not graph.preds[name]]
        if sources:
            return min(sources, key=lambda name: graph.specs[name].index)

    def key(name: str) -> Tuple[int, int, int, int]:
        rect = graph.specs[name].rect
        frontier = _placed_neighbor_count(graph, name, placed_names)
        degree = len(graph.preds[name]) + len(graph.succs[name])
        area = rect.w * rect.h
        return (frontier, degree, area, -graph.specs[name].index)

    return max(candidates, key=key)


def _ideal_anchor_from_neighbors(
    spec: NodeSpec,
    graph: GraphSpec,
    placed: Dict[str, Placed],
) -> Tuple[float, float]:
    """
    Compute an ideal local (x,y) for the node anchor by projecting from already
    placed neighbors onto this node's input/output faces and taking medians.
    """
    target_xs: List[float] = []
    target_ys: List[float] = []

    in_lx, in_ly = _face_local_center(spec.rect, spec.rect.input_face)
    out_lx, out_ly = _face_local_center(spec.rect, spec.rect.output_face)

    for pred in graph.preds[spec.name]:
        pred_p = placed.get(pred)
        if pred_p is None:
            continue
        px, py = _face_abs_center(pred_p, pred_p.rect.output_face)
        target_xs.append(px - in_lx)
        target_ys.append(py - in_ly)

    for succ in graph.succs[spec.name]:
        succ_p = placed.get(succ)
        if succ_p is None:
            continue
        sx, sy = _face_abs_center(succ_p, succ_p.rect.input_face)
        target_xs.append(sx - out_lx)
        target_ys.append(sy - out_ly)

    if not target_xs:
        return (0.0, 0.0)

    return (float(median(target_xs)), float(median(target_ys)))


def _rightward_progress_penalty(
    spec: NodeSpec,
    graph: GraphSpec,
    placed: Dict[str, Placed],
    x: int,
) -> float:
    penalty = 0.0
    for pred in graph.preds[spec.name]:
        pred_p = placed.get(pred)
        if pred_p is None:
            continue
        desired_x = pred_p.x + pred_p.rect.w
        if x < desired_x:
            penalty += desired_x - x
    return penalty


def _enumerate_candidate_positions(
    spec: NodeSpec,
    graph: GraphSpec,
    placed: Dict[str, Placed],
    W: int,
    H: int,
    candidate_limit: Optional[int],
    heuristics: PlacementHeuristics,
) -> Iterable[Tuple[int, int]]:
    """
    Enumerate candidate local placements for a node, ordered by proximity to the
    median ideal location induced by placed neighbors.

    candidate_limit:
      - None: exact search over all in-bounds positions
      - int : heuristic search over the top-N closest positions
    """
    if spec.anchor is not None:
        yield spec.anchor
        return

    min_x = 0 if spec.x_range is None else spec.x_range[0]
    max_x = (W - spec.rect.w) if spec.x_range is None else spec.x_range[1]
    min_y = 0 if spec.y_range is None else spec.y_range[0]
    max_y = (H - spec.rect.h) if spec.y_range is None else spec.y_range[1]
    if max_x < 0 or max_y < 0:
        return
    if max_x < min_x or max_y < min_y:
        return

    ideal_x, ideal_y = _ideal_anchor_from_neighbors(spec, graph, placed)
    row_parity = spec.rect.extras.get('row_parity')

    def eligible_row(y: int) -> bool:
        return row_parity is None or y % 2 == int(row_parity)

    # Exact mode: enumerate all legal coordinates, biased toward the ideal.
    if candidate_limit is None:
        xs = list(range(min_x, max_x + 1))
        ys = [y for y in range(min_y, max_y + 1) if eligible_row(y)]
        xs.sort(key=lambda x: (abs(x - ideal_x), x))
        ys.sort(key=lambda y: (abs(y - ideal_y), y))
        for y in ys:
            for x in xs:
                yield (x, y)
        return

    # Heuristic mode: score the full domain and keep the best N.
    scored: List[Tuple[float, int, int, int]] = []
    for y in range(min_y, max_y + 1):
        if not eligible_row(y):
            continue
        for x in range(min_x, max_x + 1):
            score = abs(x - ideal_x) + abs(y - ideal_y)
            score += heuristics.low_row_weight * y
            score += heuristics.rightward_progress_weight * _rightward_progress_penalty(
                spec,
                graph,
                placed,
                x,
            )
            scored.append((score, y, -x, x))
    scored.sort()

    for _, y, _, x in scored[:candidate_limit]:
        yield (x, y)


def _validate_and_preplace_anchors(
    graph: GraphSpec,
    W: int,
    H: int,
) -> Dict[str, Placed]:
    """
    Validate all fixed anchors and pre-place them before the search starts.
    """
    placed: Dict[str, Placed] = {}

    for name in graph.order:
        spec = graph.specs[name]
        if spec.anchor is None:
            continue

        p = Placed(name=name, x=spec.anchor[0], y=spec.anchor[1], rect=spec.rect)
        if not _feasible(p, placed, graph, W, H):
            raise PlacementInfeasibleError(
                f'Invalid fixed anchor for {name}: out of bounds or conflicts with another anchor.'
            )
        placed[name] = p

    return placed


def _subgraph(
    graph: GraphSpec,
    names: Sequence[str],
    spec_overrides: Optional[Dict[str, NodeSpec]] = None,
) -> GraphSpec:
    selected = set(names)
    specs = {
        name: (spec_overrides[name] if spec_overrides and name in spec_overrides else graph.specs[name])
        for name in graph.order
        if name in selected
    }
    edges = [e for e in graph.edges if e.src in selected and e.dst in selected]
    preds = {name: [] for name in specs}
    succs = {name: [] for name in specs}
    for e in edges:
        preds[e.dst].append(e.src)
        succs[e.src].append(e.dst)
    order = [name for name in graph.order if name in selected]
    return GraphSpec(order=order, specs=specs, edges=edges, preds=preds, succs=succs)


def _collect_descendants(graph: GraphSpec, root: str) -> set[str]:
    pending = [root]
    out: set[str] = set()
    while pending:
        name = pending.pop()
        if name in out:
            continue
        out.add(name)
        pending.extend(graph.succs[name])
    return out


def _detect_disjoint_fanout(
    graph: GraphSpec,
) -> Optional[Tuple[str, Dict[str, Tuple[str, ...]]]]:
    all_names = set(graph.specs)

    for root in graph.order:
        children = list(graph.succs[root])
        if len(children) <= 1 or graph.preds[root]:
            continue

        branch_sets: Dict[str, set[str]] = {}
        union: set[str] = set()
        valid = True

        for child in children:
            branch = _collect_descendants(graph, child)
            if union & branch:
                valid = False
                break
            union |= branch
            branch_sets[child] = branch

        if not valid or union | {root} != all_names:
            continue

        for child, branch in branch_sets.items():
            for name in branch:
                if graph.specs[name].anchor is not None:
                    valid = False
                    break
                allowed_preds = set(branch)
                if name == child:
                    allowed_preds.add(root)
                if any(pred not in allowed_preds for pred in graph.preds[name]):
                    valid = False
                    break
                if any(succ not in branch for succ in graph.succs[name]):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            branches = {child: tuple(name for name in graph.order if name in branch_sets[child]) for child in children}
            return root, branches

    return None


def _branch_band(graph: GraphSpec, names: Sequence[str], child: str) -> BranchBand:
    rects = [graph.specs[name].rect for name in names]
    return BranchBand(
        child=child,
        names=tuple(names),
        inner_height=max(rect.h for rect in rects),
    )


def _assign_branch_bands(
    graph: GraphSpec,
    branches: Dict[str, Tuple[str, ...]],
    order: Sequence[str],
    start_row: int,
    H: int,
) -> Optional[Dict[str, Tuple[int, int, int]]]:
    """
    Pack branch bands tightly above the shared fanout root.

    The goal is to keep compute close to row 0 / memtile-facing rows and avoid
    spare vertical room that would encourage unnecessary vertical chains.
    """
    bands = {child: _branch_band(graph, names, child) for child, names in branches.items()}
    base_heights = {child: band.inner_height for child, band in bands.items()}
    required = sum(base_heights.values())
    if start_row + required > H:
        return None

    band_rows: Dict[str, Tuple[int, int, int]] = {}
    current_top = start_row
    for child in order:
        band = bands[child]
        band_height = base_heights[child]
        band_top = current_top
        inner_top = band_top
        inner_height = band.inner_height
        band_rows[child] = (inner_top, inner_height, band_top)
        current_top = band_top + band_height

    return band_rows


# ---------------------------------------------------------------------------
# Branch-and-bound search
# ---------------------------------------------------------------------------


def _first_fit(graph: GraphSpec, preplaced: Dict[str, Placed], W: int, H: int) -> Optional[Dict[str, Placed]]:
    """A placement found without search, or None: each op in dataflow order at the leftmost, then lowest, position
    legal beside those placed before it. It packs an array the cost-led search, which sets each op by its
    neighbours, leaves gaps in that no op fits; as that search's incumbent it can only make its result cheaper."""
    placed = dict(preplaced)
    for name in graph.order:
        if name in placed:
            continue
        spec = graph.specs[name]
        x_lo, x_hi = spec.x_range or (0, W - spec.rect.w)
        y_lo, y_hi = spec.y_range or (0, H - spec.rect.h)
        fit = next(
            (
                candidate
                for x in range(x_lo, x_hi + 1)
                for y in range(y_lo, y_hi + 1)
                if _feasible(candidate := Placed(name=name, x=x, y=y, rect=spec.rect), placed, graph, W, H)
            ),
            None,
        )
        if fit is None:
            return None
        placed[name] = fit
    return placed


def _bnb_place_graph(
    graph: GraphSpec,
    W: int,
    H: int,
    lam: float,
    mu: float,
    candidate_limit: Optional[int],
    heuristics: PlacementHeuristics,
    max_states: Optional[int],
) -> Dict[str, Placed]:
    """
    BnB placement over the kernel DAG.

    Nodes are selected frontier-first (most placed neighbors) and candidates
    are ordered by proximity to the median ideal position from placed neighbors.
    A first fit (`_first_fit`) is the initial incumbent where one exists, else
    the first complete path through the DFS tree; cost pruning fires from it.
    Backtracking handles infeasibility.

    Exact search when candidate_limit=None (exponential on large grids).
    Heuristic bounded search when candidate_limit is an int (recommended: 32).
    """
    if mu < 0:
        raise ValueError('mu must be non-negative for the lower bound to remain admissible.')

    preplaced = _validate_and_preplace_anchors(graph, W, H)

    best = _first_fit(graph, preplaced, W, H) or {}
    best_cost = _full_cost(graph, best, lam, mu) if best else float('inf')
    states_visited = 0
    budget_exhausted = False
    # Tiles the placed ops occupy: two ops never share one, so a candidate on any of them is refused before the
    # pairwise checks, which most candidates of a packed array would otherwise each run.
    taken = set().union(*(placed.tiles for placed in preplaced.values()))

    def dfs(placed: Dict[str, Placed]) -> None:
        nonlocal best, best_cost, states_visited, budget_exhausted

        if budget_exhausted:
            return
        states_visited += 1
        if max_states is not None and states_visited > max_states:
            budget_exhausted = True
            return

        lb = _lower_bound(graph, placed, W, H, lam, mu)
        if lb >= best_cost:
            return

        if len(placed) == len(graph.specs):
            total = _full_cost(graph, placed, lam, mu)
            if total < best_cost:
                best_cost = total
                best = dict(placed)
            return

        name = _select_next_node(graph, placed)
        spec = graph.specs[name]

        for x, y in _enumerate_candidate_positions(
            spec,
            graph,
            placed,
            W,
            H,
            candidate_limit,
            heuristics,
        ):
            cand = Placed(name=name, x=x, y=y, rect=spec.rect)
            if cand.tiles & taken or not _feasible(cand, placed, graph, W, H):
                continue

            placed[name] = cand
            taken.update(cand.tiles)
            dfs(placed)
            taken.difference_update(cand.tiles)
            del placed[name]

    dfs(dict(preplaced))

    if len(best) != len(graph.specs):
        if budget_exhausted:
            raise PlacementInfeasibleError(f'No feasible placement found within search budget ({max_states} states).')
        raise PlacementInfeasibleError('No feasible placement found for the given graph and device.')

    return best


def _place_graph_with_fallback(
    graph: GraphSpec,
    W: int,
    H: int,
    lam: float,
    mu: float,
    candidate_limit: Optional[int],
    heuristics: PlacementHeuristics,
    max_states: Optional[int],
) -> Dict[str, Placed]:
    if candidate_limit is None:
        return _bnb_place_graph(graph, W, H, lam, mu, None, heuristics, max_states)

    try:
        return _bnb_place_graph(graph, W, H, lam, mu, candidate_limit, heuristics, max_states)
    except PlacementInfeasibleError:
        log.warning(
            'AIE placement: bounded search (candidate_limit=%d) found no feasible placement; '
            'retrying with exact search.',
            candidate_limit,
        )
        return _bnb_place_graph(graph, W, H, lam, mu, None, heuristics, max_states)


def _place_disjoint_fanout(
    graph: GraphSpec,
    W: int,
    H: int,
    lam: float,
    mu: float,
    candidate_limit: Optional[int],
    heuristics: PlacementHeuristics,
    max_states: Optional[int],
) -> Optional[Dict[str, Placed]]:
    """
    Fast path for strict disjoint fanout trees.

    Known limitation: branch bands are allocated only below the shared root.
    This matches the common case where the root is already placed near row 0,
    but it intentionally leaves rows above the root unused.
    """
    detected = _detect_disjoint_fanout(graph)
    if detected is None:
        log.debug('AIE placement: disjoint-fanout fast path not applicable.')
        return None

    root, branches = detected
    if len(branches) > 4:
        log.debug(
            'AIE placement: disjoint-fanout fast path skipped for root %s with %d branches.',
            root,
            len(branches),
        )
        return None

    root_graph = _subgraph(graph, [root])
    root_placed = _place_graph_with_fallback(
        root_graph,
        W,
        H,
        lam,
        mu,
        candidate_limit,
        heuristics,
        max_states,
    )
    root_pos = root_placed[root]
    start_row = root_pos.y + root_pos.rect.h
    if start_row >= H:
        log.debug(
            """AIE placement: disjoint-fanout fast path skipped for root %s
                because branches would start at row %d outside device height %d.""",
            root,
            start_row,
            H,
        )
        return None

    log.debug(
        'AIE placement: using disjoint-fanout fast path for root %s with %d branches.',
        root,
        len(branches),
    )

    branch_children = list(branches)
    placement_orders = list(permutations(branch_children))

    for order in placement_orders:
        band_rows = _assign_branch_bands(graph, branches, order, start_row, H)
        if band_rows is None:
            continue

        placed = dict(root_placed)
        valid = True

        for child in order:
            inner_top, inner_height, _ = band_rows[child]
            specs: Dict[str, NodeSpec] = {}
            for name in [root, *branches[child]]:
                spec = graph.specs[name]
                if name == root:
                    specs[name] = NodeSpec(
                        node=spec.node,
                        name=spec.name,
                        index=spec.index,
                        rect=spec.rect,
                        anchor=(root_pos.x, root_pos.y),
                    )
                    continue

                max_y = inner_top + inner_height - spec.rect.h
                if max_y < inner_top:
                    valid = False
                    break
                specs[name] = NodeSpec(
                    node=spec.node,
                    name=spec.name,
                    index=spec.index,
                    rect=spec.rect,
                    x_range=spec.x_range,
                    y_range=(inner_top, max_y),
                )

            if not valid:
                break

            sub_graph = _subgraph(graph, [root, *branches[child]], spec_overrides=specs)

            try:
                branch_placed = _place_graph_with_fallback(
                    sub_graph,
                    W,
                    H,
                    lam,
                    mu,
                    candidate_limit,
                    heuristics,
                    max_states,
                )
            except PlacementInfeasibleError:
                valid = False
                break

            for name, pos in branch_placed.items():
                if name != root:
                    placed[name] = pos

        if valid and len(placed) == len(graph.specs):
            return placed

    log.debug('AIE placement: disjoint-fanout fast path failed; falling back to global placement.')
    return None


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

    candidate_limit:
        Maximum number of candidate positions tried per node in the BnB search,
        ordered by proximity to the ideal position from placed neighbors.
        - int (default 64): heuristic bounded search; fast for any model size.
          If it finds no feasible placement, the placer retries with exact
          search before failing.
        - None: exact search over all in-bounds positions — only feasible on
          very small grids (e.g. W*H < 30); exponential on typical AIE arrays.

    max_states:
        Maximum DFS states explored per placement solve. If the budget is
        exhausted, the best complete placement found so far is returned; if no
        complete placement was found, placement fails hard.

    low_row_weight / rightward_progress_weight:
        Search-order heuristics only. They bias candidate exploration toward
        lower rows and left-to-right growth without changing legality or the
        final placement objective.

    """

    def __init__(
        self,
        lam: float = 1.0,
        mu: float = 0.05,
        candidate_limit: Optional[int] = 64,
        max_states: Optional[int] = 50000,
        low_row_weight: float = 0.05,
        rightward_progress_weight: float = 0.25,
    ):
        self.name = 'place_kernels'
        self._lam = float(lam)
        self._mu = float(mu)
        self._candidate_limit = candidate_limit
        self._max_states = max_states
        self._heuristics = PlacementHeuristics(
            low_row_weight=float(low_row_weight),
            rightward_progress_weight=float(rightward_progress_weight),
        )

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

        placed = _place_disjoint_fanout(
            graph=graph,
            W=W,
            H=H,
            lam=self._lam,
            mu=self._mu,
            candidate_limit=self._candidate_limit,
            heuristics=self._heuristics,
            max_states=self._max_states,
        )
        if placed is None:
            placed = _place_graph_with_fallback(
                graph=graph,
                W=W,
                H=H,
                lam=self._lam,
                mu=self._mu,
                candidate_limit=self._candidate_limit,
                heuristics=self._heuristics,
                max_states=self._max_states,
            )
        return {
            name: {
                'col': int(p.x + col_offset),
                'row': int(p.y + row_offset),
                'width': int(p.rect.w),  # the tiles reserved, not only the anchor
                'height': int(p.rect.h),
            }
            for name, p in placed.items()
        }
