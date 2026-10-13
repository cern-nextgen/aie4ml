"""The constraint-solver placer: its rules, and the hand-overs it shares by the design's ranking."""

import pytest

pytest.importorskip('ortools')

from aie4ml.op_impls.base import BufferLocation  # noqa: E402
from aie4ml.passes.placement import EdgeSpec, GraphSpec, NodeSpec, PlacementInfeasibleError, PortFace, Rect  # noqa: E402
from aie4ml.passes.placement_solver import solve_placement, tidy_placement  # noqa: E402
from aie4ml.passes.ranking import RANKING  # noqa: E402
from passes.test_placement import _crossing_graph, _tail_graph, _three_branch_graph  # noqa: E402

PORTS = {'North': 6, 'South': 4, 'East': 4, 'West': 4}
LATENCY = RANKING['latency']


@pytest.mark.parametrize('case', [(_three_branch_graph, 11), (_tail_graph, 10)], ids=['branches', 'tail'])
def test_the_solver_makes_every_required_share(case):
    build, W = case
    graph = build()
    _, shares, _ = solve_placement(graph, W, 8, 5, LATENCY)
    required = {(e.dst, e.dst_group, dp) for e in graph.edges if e.shared for _, dp in e.port_pairs}
    assert required and required <= shares


def test_the_solver_refuses_where_nothing_fits():
    with pytest.raises(PlacementInfeasibleError):
        solve_placement(_tail_graph(tail_shared=False), 8, 8, 5, LATENCY)


def _two_into_one(**edge):
    """Two full-height columns, q and k, each hand a kernel s the buffer of their top tile, which s pins in its own
    tile where it copies it; its core reaches its own memory and those west, north and south of it, so at most one
    of the two. q's leg costs 40 cycles copied, k's 80; q and k take 100 cycles, s 50."""
    face = PortFace('left', 0, 0), PortFace('right', 0, 0)

    def op(name, h, locations, cycles):
        rect = Rect(1, h, *face, buffer_locations=lambda _row: locations)
        return NodeSpec(name, rect, outputs=frozenset({'out1'}), cycles=cycles)

    top = (BufferLocation('out1', 0, 0, 3, (0, 3)),)
    reads = (BufferLocation('inA', 0, 0, 0, (0, 3), reader=(0, 0)), BufferLocation('inB', 0, 0, 0, (1, 2), (0, 0)))
    specs = {'q': op('q', 4, top, 100), 'k': op('k', 4, top, 100), 's': op('s', 1, reads, 50)}
    edges = [
        EdgeSpec(src, 's', src, True, True, 'out1', group, ((0, 0),), dst_tensor=src, cycles=(cc, cc), **edge)
        for src, group, cc in (('q', 'inA', 40), ('k', 'inB', 80))
    ]
    return GraphSpec(specs, edges, stream_ports=PORTS)


def test_the_solver_refuses_required_shares_no_placement_makes_at_once():
    with pytest.raises(PlacementInfeasibleError):
        solve_placement(_two_into_one(shared=True), 8, 4, 5, LATENCY)


def test_the_solver_shares_the_hand_over_that_shortens_the_critical_path():
    """Of the two hand-overs only one can be one buffer: sharing k's leaves q's 40-cycle copy, 190 cycles in all;
    sharing q's would leave k's 80, 230."""
    graph = _two_into_one(shareable=True)
    placed, shares, estimate = solve_placement(graph, 8, 4, 5, LATENCY)
    assert shares == {('s', 'inB', 0)}
    assert estimate == {'latency': 190, 'interval': 100, 'copies': 1, 'proven': True}
    assert (placed['s'].x, placed['s'].y) == (placed['k'].x + 1, 3)  # k's top tile is west of s
    _, *laid = tidy_placement(graph, 8, 4, 1.0, 0.05, 0, 5, placed, estimate)
    assert laid == [shares, estimate]  # the layout keeps what the placement ranked


def test_the_solver_keeps_every_stream_within_the_switch_ports():
    with pytest.raises(PlacementInfeasibleError, match='streams within the switches'):
        solve_placement(_crossing_graph(5, PORTS), 10, 1, 5, LATENCY)
    solve_placement(_crossing_graph(5, {**PORTS, 'East': 5}), 10, 1, 5, LATENCY)
