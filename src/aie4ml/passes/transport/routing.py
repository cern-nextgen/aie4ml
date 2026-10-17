"""Whether a design's streams fit the stream switches' ports (`DeviceSpec.stream_switch_ports`).

Each hand-over not in shared memory is a stream holding a port on every switch link it crosses; past that, the AIE
router fails. Memory tiles have no east or west links, so their streams enter through their column's north ports.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from ...op_impls.common_types import PORT_KIND_BUFFER
from ..cpsat import cp_model
from ..shared_buffer import pinned_locations

BELOW = None  # a memory tile or PLIO end

Tile = Tuple[int, int]
Link = Tuple[int, int, str]  # (column, row, side) of a switch, row -1 under the array


def port_tiles(inst, direction: str, group: str, port: int, col: int, row: int) -> Set[Tile]:
    """The tiles one graph port of `inst`, placed at (col, row), streams from or to: its buffers', or its kernels'."""
    binding = next(b for b in getattr(inst.ports, direction).values() if b.group == group)
    if binding.kind == PORT_KIND_BUFFER:
        tiles = {(c, r) for c, r, _ in pinned_locations(inst, group, port, col, row)}
    else:
        tiles = {
            (col + loc.rel_col, row + loc.rel_row)
            for loc in inst.variant.stream_locations(inst.node, inst.config, row)
            if loc.port_group == group and loc.port == int(port)
        }
    if not tiles:
        raise RuntimeError(f'{inst.name}.{group}[{port}]: its variant locates no tile for this {binding.kind} port.')
    return tiles


def _routes(source, target):
    """Candidate routes as (column, row, side) links, row -1 under the array: from or to below in the tile's column or
    a neighbour's; between tiles, turning once."""
    if source is BELOW and target is BELOW:
        return [[]]
    if source is BELOW:
        col, row = target
        return [_vertical(c, -1, row) + _lateral(c, col, row) for c in (col, col - 1, col + 1)]
    if target is BELOW:
        col, row = source
        return [_lateral(col, c, row) + _vertical(c, row, -1) for c in (col, col - 1, col + 1)]
    (c1, r1), (c2, r2) = source, target
    routes = [_lateral(c1, c2, r1) + _vertical(c2, r1, r2), _vertical(c1, r1, r2) + _lateral(c1, c2, r2)]
    return routes[:1] if c1 == c2 or r1 == r2 else routes


def _lateral(c1: int, c2: int, row: int):
    return [(c, row, 'East') for c in range(c1, c2)] + [(c, row, 'West') for c in range(c1, c2, -1)]


def _vertical(col: int, r1: int, r2: int):
    return [(col, r, 'North') for r in range(r1, r2)] + [(col, r, 'South') for r in range(r1, r2, -1)]


def _candidates(source, target, columns: Optional[int]) -> List[List[Link]]:
    """A branch's candidate routes (`_routes`) within `columns`."""
    return [r for r in _routes(source, target) if columns is None or all(0 <= c < columns for c, *_ in r)]


def route(demand: Dict[Link, frozenset], stream, source, target, ports, columns: Optional[int] = None) -> List[Link]:
    """Route one branch of `stream` into `demand` (each link's streams): of its candidate routes within `columns`, the
    one whose fullest link stays emptiest. A stream's branches share the links they cross."""
    chosen = min(
        _candidates(source, target, columns),
        key=lambda r: max((len(demand.get(link, frozenset()) | {stream}) / ports[link[2]] for link in r), default=0.0),
    )
    for link in chosen:
        demand[link] = demand.get(link, frozenset()) | {stream}
    return chosen


def route_overflow(
    streams: Dict[object, Tuple[Optional[Tile], Iterable[Optional[Tile]]]], ports, columns: Optional[int] = None
) -> Optional[str]:
    """Why `streams` ({stream: (source, targets)}) need more ports on some link than it has, or None: each branch takes
    one of its candidate routes within `columns`, a stream's branches sharing the links they cross. Each branch first
    takes the route whose fullest link stays emptiest (`route`), which fits most designs; where a link still overflows,
    CP-SAT decides whether any choice of routes fits, as the order of that greedy choice must not. Whatever fits
    routes, given memory tiles beside their readers."""
    demand: Dict[Link, frozenset] = {}
    for stream, (source, targets) in streams.items():
        for target in sorted(targets, key=str):
            route(demand, stream, source, target, ports, columns)
    over = sorted(((len(held), link) for link, held in demand.items() if len(held) > ports[link[2]]), reverse=True)
    if not over or _fits(streams, ports, columns):
        return None
    count, (col, row, side) = over[0]
    where = f'the switch under column {col}' if row < 0 else f'the switch of tile ({col}, {row})'
    return (
        f'{where} needs {count} streams {side.lower()}, beyond its {ports[side]} ports '
        f'({len(over)} switch links over their ports)'
    )


def _fits(streams, ports, columns: Optional[int]) -> bool:
    """Whether some choice of one candidate route per branch keeps every link within its ports, a stream's branches
    sharing the links they cross."""
    model = cp_model.CpModel()
    used: Dict[Tuple[object, Link], cp_model.IntVar] = {}  # (stream, link): the stream crosses the link
    load: Dict[Link, list] = {}
    for stream, (source, targets) in streams.items():
        for target in targets:
            picks = []
            for candidate in _candidates(source, target, columns):
                pick = model.NewBoolVar('')
                picks.append(pick)
                for link in candidate:
                    if (stream, link) not in used:
                        used[stream, link] = model.NewBoolVar('')
                        load.setdefault(link, []).append(used[stream, link])
                    model.AddImplication(pick, used[stream, link])
            model.AddExactlyOne(picks)
    for link, crossing in load.items():
        if len(crossing) > ports[link[2]]:
            model.Add(sum(crossing) <= ports[link[2]])
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1  # a small model, decided the same way every run
    return solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
