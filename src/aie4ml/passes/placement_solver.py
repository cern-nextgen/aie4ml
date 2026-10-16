"""Kernel placement as one constraint program (OR-Tools CP-SAT).

The rules of placement (`placement`) -- each op's places, the cores and memory banks it claims, the DMA channels of
every tile, the hand-overs a consumer requires in shared memory -- and the streams' routes on the switches
(`transport.routing`), all hard. Each port pair a shareable edge joins (`EdgeSpec.shareable`) is one buffer where the
two ports' places line up and the placement takes it so, and copied by a DMA otherwise: the solver decides which, by
the design's ranking (`ranking.RANKING`) of what follows -- the critical path through the kernels' cycles and the
copied legs', the slowest kernel or copied leg, the hand-overs copied (`solve_placement`) -- then, briefly and for the
design kept alone, the layout: the edges' lengths and how tightly it packs the array from the PL columns and the
bottom (`tidy_placement`). Both decide without the routes first, on the columns that hold the best placement
(`_window`), then check the routes of what they found (`_routes`): only where its streams do not fit do they decide
again with the routes, on the whole array.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..op_impls.base import KERNEL_BANKS
from .cpsat import cp_model
from .placement import (
    GraphSpec,
    Placed,
    PlacementInfeasibleError,
    PlacementOutranked,
    _face_abs_box,
    _places,
    pair_offsets,
)
from .ranking import PLACED, Solution, require, solve_ranked

_SCALE = 100  # costs to integers
_WORKERS = 8
# The layout only orders placements the ranking ties: each of its solves keeps the best found in this deterministic
# time, as proving it the best on a large array takes minutes for nothing the design gains.
_LAYOUT_TIME = 30.0


def solve_placement(
    graph: GraphSpec,
    W: int,
    H: int,
    time: float,
    order: Sequence[Tuple[str, bool]],
    first: int,
    floors: Optional[Dict[str, int]] = None,
    beat: Optional[Tuple[tuple, bool]] = None,
) -> Tuple[Dict[str, Placed], FrozenSet[Tuple[str, str, int]], Dict[str, Any]]:
    """The placement ranked best by `order` (`ranking.RANKING`) whose streams fit the switches, the input ports
    (op, group, port) it has read their producer's buffer, and its estimate (`PhysicalIR.estimate`): proven where each
    stage was, or reached `floors`, the design search's bound, below which no placement goes. A legal placement comes
    first, as on a tight array neither the ranking nor the routes lead a search from scratch to one. With `beat`
    (`ranking.to_beat`), only a placement ranking the design better than the best built counts: where none does, it
    raises `PlacementOutranked`, which spares ranking a design that cannot be kept.

    It is ranked without the streams' routes, within `_window` from column `first`: a relaxation, so where its best
    placement's streams fit (`_routes`), that is the best. Only where they do not is it ranked again with the routes, on
    the whole array, from that placement."""
    ranking = tuple(criterion for criterion in order if criterion[0] in PLACED)
    relaxed = _Model(graph, W, H, _window(graph, W, H, first))
    criteria = relaxed.estimate()
    if beat is not None:
        require(relaxed.model, criteria, ranking, beat)
    found, proven = relaxed.solve(time, (), {}, None)
    if found is None:
        if beat is not None:
            raise _outranked(time, proven)
        if proven:
            conflict = _conflict(graph, W, H, time)
            raise PlacementInfeasibleError(f'no placement exists{conflict and ": " + conflict + " cannot all hold"}.')
        raise PlacementInfeasibleError(
            f'the placement solver found no placement within AIEConfig PlacerTime={time:g}.', proven=False
        )
    best, proven = relaxed.solve(time, ranking, criteria, found.decisions, floors)
    if best is None:  # the hint the solve starts from places, so only time stops it finding a placement
        raise PlacementInfeasibleError(
            f'the placement solver ranked no placement within AIEConfig PlacerTime={time:g}.', proven=False
        )
    placed, shares = relaxed.placed(best), relaxed.shares(best)
    if _routes(graph, W, H, time, placed, shares):
        return placed, shares, {**{name: best.criteria[name] for name in PLACED}, 'proven': proven}

    ranked = _Model(graph, W, H)
    criteria = {'overflow': ranked.route(), **ranked.estimate()}
    if beat is not None:
        require(ranked.model, criteria, ranking, beat)
    # the streams within the switches' ports first: what the search cannot drive out refuses the placement
    found, proven = ranked.solve(time, (('overflow', False),), criteria, ranked.hint(placed))
    if found is None or found.criteria['overflow']:
        if beat is not None:
            raise _outranked(time, proven)
        what = "keeps its streams within the switches' ports" if proven else 'fits the switches'
        raise PlacementInfeasibleError(f'no placement {what} within AIEConfig PlacerTime={time:g}.', proven=proven)
    best, proven = ranked.solve(time, ranking, criteria, found.decisions, floors)
    if best is None:  # the hint the solve starts from places, so only time stops it finding a placement
        best, proven = found, False
    estimate = {name: best.criteria[name] for name in PLACED}
    return ranked.placed(best), ranked.shares(best), {**estimate, 'proven': proven}


def _outranked(time: float, proven: bool) -> PlacementOutranked:
    within = '' if proven else f' within AIEConfig PlacerTime={time:g}'
    return PlacementOutranked(f'no placement ranks the design better than the best built{within}.', proven=proven)


def tidy_placement(
    graph: GraphSpec,
    W: int,
    H: int,
    lam: float,
    mu: float,
    first: int,
    time: float,
    placed: Dict[str, Placed],
    estimate: Dict[str, Any],
) -> Tuple[Dict[str, Placed], FrozenSet[Tuple[str, str, int]], Dict[str, Any]]:
    """`solve_placement`'s `placed` and `estimate` laid out: of the placements whose streams fit the switches and
    which rank at least as well, the shortest layout found within _LAYOUT_TIME, packed east from column `first`. Laid
    out without the routes within `_window` first, as `solve_placement` ranks, and again with them on the whole array
    only where that layout's streams do not fit."""
    laid = _tidy(graph, W, H, _window(graph, W, H, first), lam, mu, first, time, placed, estimate)
    if _routes(graph, W, H, time, *laid[:2]):
        return laid
    return _tidy(graph, W, H, None, lam, mu, first, time, placed, estimate, routed=True)


def _tidy(
    graph: GraphSpec,
    W: int,
    H: int,
    window: Optional[range],
    lam: float,
    mu: float,
    first: int,
    time: float,
    placed: Dict[str, Placed],
    estimate: Dict[str, Any],
    routed: bool = False,
) -> Tuple[Dict[str, Placed], FrozenSet[Tuple[str, str, int]], Dict[str, Any]]:
    """`tidy_placement`'s layout among the placements in `window` ranking at least as well, with their streams within
    the switches' ports where `routed`."""
    tidy = _Model(graph, W, H, window)
    if routed:
        overflow, _ = tidy.route()
        tidy.model.Add(overflow == 0)
    criteria = tidy.estimate()
    for name in PLACED:
        tidy.model.Add(criteria[name][0] <= estimate[name])
    found, proven = tidy.solve(time, (), criteria, tidy.hint(placed))
    if found is None:
        if proven:
            raise RuntimeError('tidy_placement: the placement given is not a placement of this graph ranking so.')
        raise PlacementInfeasibleError(
            f'the placement solver did not find the placement given within AIEConfig PlacerTime={time:g}.', proven=False
        )
    criteria = {**criteria, 'layout': tidy.layout(lam, mu), 'compactness': tidy.compactness(first)}
    laid, _ = tidy.solve(_LAYOUT_TIME, (('layout', False), ('compactness', False)), criteria, found.decisions)
    best = found if laid is None else laid  # else the layout found nothing better in its time
    ranked = {name: best.criteria[name] for name in PLACED}
    return tidy.placed(best), tidy.shares(best), {**ranked, 'proven': estimate['proven']}


def _window(graph: GraphSpec, W: int, H: int, first: int) -> Optional[range]:
    """Columns, from `first` where they fit, that hold a placement ranking as well as the best on the array without
    the streams' routes; None for the whole array. Every other rule is local to the columns an op's tiles and buffers
    take, and the ranking depends only on the buffers shared: so the ops sharing buffers kept together, apart from the
    rest, any placement packs into the sum of those columns, ranking alike. A pinned place does not move: it takes the
    whole array."""
    width = 0
    for spec in graph.specs.values():
        columns = [loc.rel_col for row in range(H) for loc in spec.rect.locations_at(row)]
        width += max([spec.rect.w, *(col + 1 for col in columns)]) - min([0, *columns])
    if width >= W or any(spec.anchor is not None for spec in graph.specs.values()):
        return None
    start = max(0, min(first, W - width))
    return range(start, start + width)


def _routes(
    graph: GraphSpec, W: int, H: int, time: float, placed: Dict[str, Placed], shares: FrozenSet[Tuple[str, str, int]]
) -> bool:
    """Whether `placed`, reading their producer's buffer at the input ports `shares`, has routes for its streams
    within the switches' ports (`_route_streams`)."""
    pinned = {name: replace(spec, anchor=(placed[name].x, placed[name].y)) for name, spec in graph.specs.items()}
    check = _Model(replace(graph, specs=pinned), W, H)
    for port, pair in check.held.items():
        check.model.Add(pair == int(port in shares))
    overflow, _ = check.route()
    check.model.Add(overflow == 0)
    found, _ = check.solve(time, (), {}, None)
    return found is not None


def _conflict(graph: GraphSpec, W: int, H: int, time: float) -> str:
    """Of the places pinned and the hand-overs required in one buffer, some no placement holds at once (CP-SAT's
    sufficient assumptions for infeasibility), named; '' where the kernels alone do not fit."""
    explained = _Model(graph, W, H, explain=True)
    names = {literal.Index(): name for name, literal in explained.assumptions.items()}
    explained.model.AddAssumptions(list(explained.assumptions.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = float(time)
    solver.parameters.num_workers = _WORKERS
    if solver.Solve(explained.model) != cp_model.INFEASIBLE:
        return ''
    return ', '.join(sorted(names[index] for index in solver.SufficientAssumptionsForInfeasibility()))


class _Model:
    """The placement rules as one constraint program: a place per op, within the columns `window` where given, and per
    port pair of a shareable or shared edge whether it is one buffer (`held`), by (consumer, group, port). `explain`
    makes each pinned place and each pair a shared edge requires an assumption (`assumptions`, by name) rather than a
    rule (`_conflict`)."""

    def __init__(self, graph: GraphSpec, W: int, H: int, window: Optional[range] = None, explain: bool = False):
        self.graph, self.W, self.H = graph, W, H
        model = self.model = cp_model.CpModel()
        self.assumptions: Dict[str, cp_model.IntVar] = {}
        pairable = defaultdict(set)  # op -> its input ports a pair may hand over one buffer on
        for edge in graph.edges:
            if edge.shareable or edge.shared:
                pairable[edge.dst].update((edge.dst_group, dst_port) for _, dst_port in edge.port_pairs)
        found = {
            name: [
                (p, past)
                for p, past in _places(replace(spec, anchor=None) if explain else spec, W, H, frozenset(pairable[name]))
                if window is None or (window.start <= p.x and p.x + p.rect.w <= window.stop)
            ]
            for name, spec in graph.specs.items()
        }
        places = self.places = {name: [p for p, _ in options] for name, options in found.items()}
        chosen = self.chosen = {name: [model.NewBoolVar('') for _ in options] for name, options in places.items()}
        for literals in chosen.values():
            model.AddExactlyOne(literals)
        if explain:
            for name, spec in graph.specs.items():
                if spec.anchor is not None:
                    pinned = self.assumptions[f"{name}'s pinned place"] = model.NewBoolVar('')
                    model.AddImplication(pinned, chosen[name][[(p.x, p.y) for p in places[name]].index(spec.anchor)])

        # Each port pair of a shareable edge is one buffer only where the two ports' places line up: the consumer's
        # port pins it where the producer's does, or its reader reaches where the producer's pins it. The consumer
        # port's own location holds nothing then. A shared edge's pairs must be.
        held = self.held = {}
        for edge in graph.edges:
            if not (edge.shareable or edge.shared):
                continue
            src, dst = graph.specs[edge.src].rect, graph.specs[edge.dst].rect
            at = {(p.x, p.y): literal for p, literal in zip(places[edge.dst], chosen[edge.dst])}
            for src_port, dst_port in edge.port_pairs:
                pair = held[edge.dst, edge.dst_group, dst_port] = model.NewBoolVar('')
                if edge.shared and explain:
                    name = f'{edge.src} -> {edge.dst} in one buffer'
                    model.AddImplication(self.assumptions.setdefault(name, model.NewBoolVar('')), pair)
                elif edge.shared:
                    model.Add(pair == 1)
                offsets = pair_offsets(
                    src, edge.src_group, src_port, dst, edge.dst_group, dst_port, H, graph.alternating_horizontal
                )
                for p, literal in zip(places[edge.src], chosen[edge.src]):
                    lined_up = [
                        at[p.x + offset, dst_row]
                        for (src_row, dst_row), found in offsets.items()
                        if src_row == p.y
                        for offset in found
                        if (p.x + offset, dst_row) in at
                    ]
                    model.AddBoolOr([literal.Not(), pair.Not(), *lined_up])
        # A place putting an input's buffer past the array's far edges takes it in its producer's buffer.
        for name, options in found.items():
            for (_, past), literal in zip(options, chosen[name]):
                for group, port in past:
                    model.AddImplication(literal, held[name, group, port])
        # by kernel buffer port (op, group, port), the pair it may hand over one buffer on
        self.pair_of = {}
        for edge in graph.edges:
            for src_port, dst_port in edge.port_pairs:
                pair = held.get((edge.dst, edge.dst_group, dst_port))
                if pair is not None:
                    self.pair_of[edge.src, edge.src_group, src_port] = pair
                    self.pair_of[edge.dst, edge.dst_group, dst_port] = pair

        # A core or memory bank is one op's -- its kernels' cores and `KERNEL_BANKS`, its buffers' banks -- besides
        # the buffer two ops hand over there.
        claims: Dict[tuple, List] = defaultdict(list)
        for name, options in places.items():
            always: Dict[tuple, List] = defaultdict(list)
            unless: Dict[Tuple[str, int], Dict[tuple, List]] = defaultdict(lambda: defaultdict(list))
            for p, literal in zip(options, chosen[name]):
                cells = {(col, row, 'core') for col, row in p.tiles}
                cells.update((col, row, bank) for col, row in p.tiles for bank in KERNEL_BANKS)
                for loc in p.rect.locations_at(p.y):
                    banks = {(p.x + loc.rel_col, p.y + loc.rel_row, bank) for bank in loc.banks}
                    if (name, loc.port_group, loc.port) in held:
                        for cell in banks:
                            unless[loc.port_group, loc.port][cell].append(literal)
                    else:
                        cells |= banks
                for cell in cells:
                    always[cell].append(literal)
            for cell in set(always) | {cell for by_cell in unless.values() for cell in by_cell}:
                claim = model.NewBoolVar('')
                if always[cell]:
                    model.Add(claim >= sum(always[cell]))
                for (group, port), by_cell in unless.items():
                    if by_cell[cell]:
                        model.Add(claim >= sum(by_cell[cell]) - held[name, group, port])
                claims[cell].append(claim)
        for literals in claims.values():
            model.Add(sum(literals) <= 1)

        # Each kernel buffer not one shared buffer takes a DMA channel on the tile holding it, as the physical verifier
        # counts them.
        if graph.dma_channels:
            use: Dict[Tuple[int, int, str], List] = defaultdict(list)
            for name, options in places.items():
                outputs = graph.specs[name].outputs
                for p, literal in zip(options, chosen[name]):
                    for loc in p.rect.locations_at(p.y):
                        key = (p.x + loc.rel_col, p.y + loc.rel_row, 'MM2S' if loc.port_group in outputs else 'S2MM')
                        pair = self.pair_of.get((name, loc.port_group, loc.port))
                        if pair is None:
                            use[key].append(literal)
                            continue
                        taken = model.NewBoolVar('')
                        model.Add(taken >= literal - pair)
                        use[key].append(taken)
            for literals in use.values():
                model.Add(sum(literals) <= graph.dma_channels)

    # -- what the placement costs -------------------------------------------------------------------

    def estimate(self) -> Dict[str, Tuple[Any, int]]:
        """The criteria a placement changes (`ranking.PLACED`), each an expression and a bound: the latency, the
        last kernel to finish and its legs to the graph's outputs, along the kernels' cycles and each leg's, which
        is nothing where every port pair of it is one buffer; the interval, the slowest kernel or leg; the port pairs
        of direct edges a DMA copies."""
        model, graph = self.model, self.graph
        specs = graph.specs
        horizon = 1 + sum(spec.arrival[0] + spec.cycles + spec.departure[0] for spec in specs.values())
        horizon += sum(edge.cycles[0] for edge in graph.edges)
        latency, interval = model.NewIntVar(0, horizon, 'latency'), model.NewIntVar(0, horizon, 'interval')
        ready = {name: model.NewIntVar(0, horizon, '') for name in specs}
        for name, spec in specs.items():
            model.Add(ready[name] >= spec.arrival[0] + spec.cycles)
            model.Add(latency >= ready[name] + spec.departure[0])
            model.Add(interval >= max(spec.cycles, spec.arrival[1], spec.departure[1]))
        copies, pairs = [], 0
        for edge in graph.edges:
            held = [self.held.get((edge.dst, edge.dst_group, dst_port)) for _, dst_port in edge.port_pairs]
            if edge.direct:
                copies += [1 if pair is None else 1 - pair for pair in held]
                pairs += len(held)
            if held and all(pair is not None for pair in held):  # nothing where every pair is one buffer
                copied = model.NewBoolVar('')
                for pair in held:
                    model.Add(copied >= 1 - pair)
            else:
                copied = 1
            model.Add(ready[edge.dst] >= ready[edge.src] + edge.cycles[0] * copied + specs[edge.dst].cycles)
            model.Add(interval >= edge.cycles[1] * copied)
        return {'latency': (latency, horizon), 'interval': (interval, horizon), 'copies': (sum(copies), pairs)}

    def layout(self, lam: float, mu: float) -> Tuple[Any, int]:
        """The layout's cost of a placement: its edges' distances, rows weighed by `lam`, and its ops' rows by `mu`; and
        a bound on it."""
        model, graph, places, chosen = self.model, self.graph, self.places, self.chosen
        W, H = self.W, self.H
        col = {name: sum(p.x * v for p, v in zip(places[name], chosen[name])) for name in places}
        row = {name: sum(p.y * v for p, v in zip(places[name], chosen[name])) for name in places}
        cost = [round(_SCALE * mu) * row[name] for name in places]
        for edge in graph.edges:
            src, dst = graph.specs[edge.src].rect, graph.specs[edge.dst].rect
            a = _face_abs_box(Placed(edge.src, 0, 0, src), src.output_face)
            b = _face_abs_box(Placed(edge.dst, 0, 0, dst), dst.input_face)
            dx, dy = model.NewIntVar(0, W, ''), model.NewIntVar(0, H, '')
            model.AddMaxEquality(
                dx,
                [
                    0,
                    col[edge.dst] + int(b[0]) - col[edge.src] - int(a[1]),
                    col[edge.src] + int(a[0]) - col[edge.dst] - int(b[1]),
                ],
            )
            model.AddMaxEquality(
                dy,
                [
                    0,
                    row[edge.dst] + int(b[2]) - row[edge.src] - int(a[3]),
                    row[edge.src] + int(a[2]) - row[edge.dst] - int(b[3]),
                ],
            )
            cost += [_SCALE * dx, round(_SCALE * lam) * dy]
        bound = len(graph.edges) * (_SCALE * W + round(_SCALE * lam) * H) + round(_SCALE * mu) * H * len(places)
        return sum(cost), bound

    def compactness(self, first: int) -> Tuple[Any, int]:
        """How tightly a placement packs the array east from column `first` -- the first with a PL interface, by the
        PLIOs -- and from the bottom, the columns west of it last: the sum of its ops' column ranks * H + rows; and a
        bound on it."""
        W, H = self.W, self.H
        rank = [x - first if x >= first else W - 1 - x for x in range(W)]
        terms = [
            (rank[p.x] * H + p.y) * literal
            for name, options in self.places.items()
            for p, literal in zip(options, self.chosen[name])
        ]
        return sum(terms), len(self.places) * self.W * H

    def route(self) -> Tuple[Any, int]:
        """The streams' overflow past the switches' ports (`_route_streams`), and a bound on it."""
        overflow, bound = _route_streams(self.model, self.graph, self.places, self.chosen, self.held, self.W, self.H)
        return sum(overflow), bound

    # -- solving and reading a solution -------------------------------------------------------------

    @property
    def decisions(self) -> List[cp_model.IntVar]:
        return [*(literal for literals in self.chosen.values() for literal in literals), *self.held.values()]

    def solve(self, time: float, order, criteria, hint, floors=None) -> Tuple[Optional[Solution], bool]:
        """`solve_ranked` of this model by `order`, each solve within the deterministic `time`."""
        solver = cp_model.CpSolver()
        # Parallel search, which finds a tight array's placements where the deterministic interleaved one does not, so
        # a rerun may return another placement ranking alike. Deterministic time keeps the limit independent of load.
        solver.parameters.max_deterministic_time = float(time)
        solver.parameters.num_workers = _WORKERS
        return solve_ranked(self.model, solver, criteria, order, self.decisions, hint, floors)

    def hint(self, placed: Dict[str, Placed]) -> Dict[int, int]:
        """`placed` as this model's places, by variable index."""
        return {
            literal.Index(): int((option.x, option.y) == (placed[name].x, placed[name].y))
            for name, options in self.places.items()
            for option, literal in zip(options, self.chosen[name])
        }

    def placed(self, found: Solution) -> Dict[str, Placed]:
        return {
            name: next(p for p, literal in zip(options, self.chosen[name]) if found.decisions[literal.Index()])
            for name, options in self.places.items()
        }

    def shares(self, found: Solution) -> FrozenSet[Tuple[str, str, int]]:
        """The input ports (op, group, port) the solution has read their producer's buffer."""
        return frozenset(port for port, pair in self.held.items() if found.decisions[pair.Index()])


def _route_streams(model, graph, places, chosen, held, W, H) -> Tuple[List[cp_model.IntVar], int]:
    """The streams on the switches' links (`transport.routing`), as each link's overflow past its ports, and a bound
    on their sum. Each hand-over not one shared buffer is a stream from its producer port's tile to each of its
    consumer port's tiles, by one of the two routes that turn once; each port a PLIO or a memory tile streams to or
    from (`GraphSpec.below`) one up or down the column of its tile or a neighbour's, through the switches under the
    array (row -1), then one hop across on its row, within the array's columns. A stream's branches share the links
    they cross. Each column boundary holds at most its rows' ports in kernel streams, whichever routes they take:
    implied, but it lets the search see congestion before both ends of a stream are placed. No link holds more streams
    than there are: where no more than any link's ports, nothing is routed."""
    stream_ports = graph.stream_ports
    streams = {
        (edge.src, edge.src_group, src_port)
        for edge in graph.edges
        if edge.direct and not edge.shared
        for src_port, _ in edge.port_pairs
    }
    streams.update((name, group, port) for name, ends in graph.below.items() for _, group, port in ends)
    if len(streams) <= min(stream_ports.values()):
        return [], 0
    tiles: Dict[Tuple[str, str, int, int], Dict[str, Dict[int, Any]]] = {}

    def endpoint(name, group, port, index):
        """Flags of where a port's `index`-th tile (`Placed.port_tiles`) sits: its column and row, equal to, at most
        and at least each value; constants where its op has one place."""
        key = (name, group, port, index)
        if key not in tiles:
            found = {}
            axes = zip(*(sorted(p.port_tiles(group, port))[index] for p in places[name]))
            for axis, extent, values in zip(('col', 'row'), (W, H), axes):
                if len(values) == 1:
                    found[axis + 'le'] = {value: int(values[0] <= value) for value in range(-1, extent + 1)}
                    found[axis + 'eq'] = {value: int(values[0] == value) for value in range(-1, extent + 1)}
                else:
                    at = model.NewIntVar(-1, extent, '')
                    model.Add(at == sum(value * literal for value, literal in zip(values, chosen[name])))
                    found[axis + 'le'], found[axis + 'eq'] = {}, {}
                    for value in range(-1, extent + 1):
                        le, eq = model.NewBoolVar(''), model.NewBoolVar('')
                        model.Add(at <= value).OnlyEnforceIf(le)
                        model.Add(at > value).OnlyEnforceIf(le.Not())
                        model.Add(at == value).OnlyEnforceIf(eq)
                        model.Add(at != value).OnlyEnforceIf(eq.Not())
                        found[axis + 'le'][value], found[axis + 'eq'][value] = le, eq
                found[axis + 'ge'] = {value: _not(found[axis + 'le'][value - 1]) for value in range(0, extent + 1)}
            tiles[key] = found
        return tiles[key]

    branches = defaultdict(list)  # stream -> (active, source, target)
    for edge in graph.edges:
        if not edge.direct:
            continue
        for src_port, dst_port in edge.port_pairs:
            pair = held.get((edge.dst, edge.dst_group, dst_port))
            active = 1 if pair is None else 1 - pair
            source = endpoint(edge.src, edge.src_group, src_port, 0)
            for index in range(len(places[edge.dst][0].port_tiles(edge.dst_group, dst_port))):
                target = endpoint(edge.dst, edge.dst_group, dst_port, index)
                branches[edge.src, edge.src_group, src_port].append((active, source, target))
    for name, ends in graph.below.items():  # BELOW, None, at the end below
        for direction, group, port in ends:
            for index in range(len(places[name][0].port_tiles(group, port))):
                end = endpoint(name, group, port, index)
                branches[name, group, port].append((1, None, end) if direction == 'inputs' else (1, end, None))

    demand = defaultdict(list)
    cuts = defaultdict(list)  # (column, side) -> streams crossing it: each crosses on one row, whichever route
    for stream, legs in branches.items():
        links: Dict[Tuple[int, int, str], cp_model.IntVar] = {}
        crossings: Dict[Tuple[int, str], cp_model.IntVar] = {}

        def crossing(key):
            if key not in crossings:
                crossings[key] = model.NewBoolVar('')
                cuts[key].append(crossings[key])
            return crossings[key]

        def link(key):
            if key not in links:
                links[key] = model.NewBoolVar('')
                demand[key].append(links[key])
            return links[key]

        for active, s, t in legs:
            if s is None or t is None:
                continue
            for c in range(W - 1):
                _implies(model, lambda: crossing((c, 'East')), active, s['colle'][c], t['colge'][c + 1])
                _implies(model, lambda: crossing((c, 'West')), active, s['colge'][c + 1], t['colle'][c])

        for active, s, t in legs:
            if s is None or t is None:
                _route_below(model, link, t if s is None else s, s is None, W, H)
                continue
            first = model.NewBoolVar('')  # lateral on the source row, then vertical at the target column
            for r in range(H):
                if _never(s['roweq'][r]) and _never(t['roweq'][r]):
                    continue  # no route runs on the row
                on_row = model.NewBoolVar('')
                _implies(model, lambda: on_row, first, s['roweq'][r])
                _implies(model, lambda: on_row, first.Not(), t['roweq'][r])
                for c in range(W - 1):
                    _implies(model, lambda: link((c, r, 'East')), active, s['colle'][c], t['colge'][c + 1], on_row)
                    _implies(model, lambda: link((c + 1, r, 'West')), active, s['colge'][c + 1], t['colle'][c], on_row)
            for c in range(W):
                if _never(s['coleq'][c]) and _never(t['coleq'][c]):
                    continue  # no route runs in the column
                on_col = model.NewBoolVar('')
                _implies(model, lambda: on_col, first, t['coleq'][c])
                _implies(model, lambda: on_col, first.Not(), s['coleq'][c])
                for r in range(H - 1):
                    _implies(model, lambda: link((c, r, 'North')), active, s['rowle'][r], t['rowge'][r + 1], on_col)
                    _implies(model, lambda: link((c, r + 1, 'South')), active, s['rowge'][r + 1], t['rowle'][r], on_col)
    for (_, side), literals in cuts.items():
        model.Add(sum(literals) <= stream_ports[side] * H)
    overflow, bound = [], 0
    for (_, _, side), literals in demand.items():
        over = model.NewIntVar(0, len(literals), '')
        model.Add(sum(literals) <= stream_ports[side] + over)
        overflow.append(over)
        bound += len(literals)
    return overflow, bound


def _route_below(model, link, end, up: bool, W: int, H: int) -> None:
    """A stream between below and a port's tile `end` (`_route_streams`'s flags), `up` from below: vertical in the
    tile's column or a neighbour's within the array, through row -1, and the one hop across on the tile's row."""
    column = {d: model.NewBoolVar('') for d in (-1, 0, 1)}  # the column it runs in: west of the tile, its, east
    model.AddExactlyOne(column.values())
    model.Add(column[-1] <= end['colge'][1])
    model.Add(column[1] <= end['colle'][W - 2])
    for d, taken in column.items():
        for c in range(max(0, d), min(W, W + d)):
            at = end['coleq'][c - d]
            if _never(at):
                continue
            for r in range(H):
                vertical = (c, r - 1, 'North') if up else (c, r, 'South')
                _implies(model, lambda: link(vertical), taken, at, end['rowge'][r])
            if d:
                across = (c, 'East' if d < 0 else 'West') if up else (c - d, 'West' if d < 0 else 'East')
                for r in range(H):
                    _implies(model, lambda: link((across[0], r, across[1])), taken, at, end['roweq'][r])


def _never(flag) -> bool:
    """Whether `flag`, a literal or a constant (`_route_streams`), is the constant 0."""
    return isinstance(flag, int) and not flag


def _not(flag):
    return 1 - flag if isinstance(flag, int) else flag.Not()


def _implies(model, target, *flags) -> None:
    """The literal `target()` holds where all `flags` do: each a literal, 1 less one, or a constant. A constant 0
    among them implies nothing: then nothing is added, and `target` is not called."""
    if any(_never(flag) for flag in flags):
        return
    terms = [flag for flag in flags if not isinstance(flag, int)]
    model.Add(target() >= sum(terms) - len(terms) + 1)
