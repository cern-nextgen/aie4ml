"""Focused placement legality tests."""

from dataclasses import replace

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls.base import BufferLocation, row_flow
from aie4ml.passes.placement import (
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    PlacementInfeasibleError,
    PortFace,
    Rect,
    pair_offsets,
)
from aie4ml.passes.placement_solver import solve_placement
from aie4ml.passes.ranking import RANKING
from helpers import PART, TensorProto, dq, helper, lower, make_model, numpy_helper, parallelism, qdq, qparams
from passes.test_choose_parallelism import _wide_model

PORTS = {'North': 6, 'South': 4, 'East': 4, 'West': 4}


def _place(graph: GraphSpec, W: int, H: int):
    """`solve_placement` of `graph` by the latency ranking: (places, shares, estimate)."""
    return solve_placement(graph, W, H, 5, RANKING['latency'])


def _one_tile(name, locations, anchor=None, outputs=frozenset()):
    rect = Rect(1, 1, PortFace('left', 0, 0), PortFace('right', 0, 0), buffer_locations=lambda _row: locations)
    return NodeSpec(name, rect, anchor, outputs=outputs)


@pytest.mark.parametrize('row', [0, 1], ids=['even-row', 'odd-row'])
def test_neighbours_share_only_an_exactly_aliased_buffer(row):
    """Data flows left to right on every AIE row; the hand-over buffer lives in the tile both kernels
    reach -- the producer's on an even row, the consumer's on an odd one. Neighbours may hold the same banks
    only through that one buffer; a kernel's own stack and parameter banks leave the others free."""
    flow = row_flow(True, row, 1)
    out = (BufferLocation('out', 0, flow.output_col, 0, (0, 3)),)
    into = (BufferLocation('in', 0, flow.input_col, 0, (0, 3)),)
    edge = EdgeSpec('producer', 'consumer', direct=True, src_group='out', dst_group='in', port_pairs=((0, 0),))
    pair = {
        'producer': _one_tile('producer', out, (7, row), frozenset({'out'})),
        'consumer': _one_tile('consumer', into, (8, row)),
    }
    _, shares, estimate = _place(GraphSpec(pair, [replace(edge, shareable=True)], PORTS), 10, 2)
    assert shares == {('consumer', 'in', 0)} and estimate['copies'] == 0

    # Another op writing those banks of that tile conflicts: only the edge's other end may hold them. One holding
    # only its kernel's stack and parameter banks there does not.
    shared_tile = 7 if flow.output_col == 0 else 8  # the producer's tile, west of the consumer, or the consumer's
    other = 'consumer' if shared_tile == 7 else 'producer'
    for writes, fits in (((BufferLocation('out', 0, 0, 0, (0, 3)),), False), ((), True)):
        graph = GraphSpec({other: pair[other], 'other': _one_tile('other', writes, (shared_tile, row))}, [], PORTS)
        if fits:
            _place(graph, 10, 2)
        else:
            with pytest.raises(PlacementInfeasibleError):
                _place(graph, 10, 2)

    # Other banks of that memory are another buffer: no place makes the two ports one.
    apart = _one_tile('consumer', (BufferLocation('in', 0, flow.input_col, 0, (1, 2)),)).rect
    offsets = pair_offsets(pair['producer'].rect, 'out', 0, apart, 'in', 0, 2, True)
    assert not any(offsets.values())


def test_placement_packs_from_the_plio_columns_and_widens_west_only_when_it_must(tmp_path):
    def columns(ctx):
        return sorted(p['col'] for p in ctx.ir.physical.placements.values())

    assert columns(lower(_wide_model(), tmp_path / 'default', part=PART)) == [5, 6, 7]  # VE2802's first PL column: 5
    # One tile from the first PL column on: the design widens into the columns west of it.
    narrow = {'Columns': 7, 'Rows': 1, 'PLColumnStart': 6}
    assert columns(lower(_wide_model(), tmp_path / 'narrow', part=PART, aie_config=narrow)) == [4, 5, 6]
    # A ColumnStart the user gives bounds the region.
    with pytest.raises(ConfigRefused):
        lower(_wide_model(), tmp_path / 'held', part=PART, aie_config={**narrow, 'ColumnStart': 6})


def _column_op(name, rows):
    """A one-column op of `rows` kernels, as AIE-ML's dense ops: each reads from its west neighbour's memory and
    writes to its own."""

    def locations(_row):
        return tuple(
            location
            for row in range(rows)
            for location in (BufferLocation('in1', row, -1, row, (0, 3)), BufferLocation('out1', row, 0, row, (0, 3)))
        )

    face = PortFace('left', 0, rows - 1), PortFace('right', 0, rows - 1)
    return NodeSpec(name, Rect(1, rows, *face, buffer_locations=locations), outputs=frozenset({'out1'}))


def _tail_graph(tail_shared: bool = True):
    """Five eight-row ops handing over through shared buffers, a sixth fed by a memory tile, then a five-row op
    feeding a three-row one, which reads their buffer in shared memory where `tail_shared`."""
    rows = {**{f'm{k}': 8 for k in range(6)}, 't5': 5, 't3': 3}
    specs = {name: _column_op(name, height) for name, height in rows.items()}

    def edge(src, dst, share):
        pairs = tuple((port, port) for port in range(rows[dst])) if share else ()
        return EdgeSpec(src, dst, direct=share, src_group='out1', dst_group='in1', port_pairs=pairs, shared=share)

    edges = [edge(f'm{k}', f'm{k + 1}', k < 4) for k in range(5)]
    edges += [edge('m5', 't5', False), edge('t5', 't3', True)]
    if not tail_shared:
        edges[-1] = replace(edges[-1], shared=False)
    return GraphSpec(specs, edges, PORTS)


def _three_branch_graph():
    """An eight-row source feeds three independent chains of four three-row ops on an eight-row array."""
    specs = {'src': _column_op('src', 8)}
    edges = []
    for branch in range(3):
        previous = 'src'
        for stage in range(4):
            name = f'b{branch}_{stage}'
            specs[name] = _column_op(name, 3)
            is_broadcast = previous == 'src'
            edges.append(
                EdgeSpec(
                    previous,
                    name,
                    direct=True,
                    src_group='out1',
                    dst_group='in1',
                    port_pairs=tuple((port, port) for port in range(3)),
                    shared=not is_broadcast,
                )
            )
            previous = name
    return GraphSpec(specs, edges, PORTS)


def test_placement_makes_every_required_share_or_refuses():
    """With room, the tail's ops sit side by side and share their buffer; one column narrower, a design requiring
    that share is refused, while one handing it over through a DMA stacks them in one column; narrower still,
    nothing fits."""
    side_by_side, _, _ = _place(_tail_graph(), 10, 8)
    assert (side_by_side['t3'].x, side_by_side['t3'].y) == (side_by_side['t5'].x + 1, side_by_side['t5'].y)
    with pytest.raises(PlacementInfeasibleError):
        _place(_tail_graph(), 9, 8)
    stacked, _, _ = _place(_tail_graph(tail_shared=False), 9, 8)
    assert stacked['t3'].x == stacked['t5'].x
    bands = sorted(((stacked[name].y, stacked[name].y + stacked[name].rect.h) for name in ('t5', 't3')))
    assert bands[0][0] == 0 and bands[0][1] == bands[1][0] and bands[1][1] == 8
    assert all(stacked[f'm{k + 1}'].x == stacked[f'm{k}'].x + 1 for k in range(4))  # the chain still shares
    with pytest.raises(PlacementInfeasibleError):
        _place(_tail_graph(tail_shared=False), 8, 8)


def _split_heads_model(whole_reader: bool):
    """A 16->32 Dense split by a Split into two 16-wide halves, each read by its own 16->16 Dense; optionally a third
    Dense reads the whole tensor too."""
    nodes, inits = [], [*qparams('x')]
    dq(nodes, 'x_q', 'x', 'x')

    def dense(src, name, k, n, seed):
        dq(nodes, f'{name}_w_q', f'{name}_w', f'{name}_w')
        nodes.append(helper.make_node('MatMul', [src, f'{name}_w'], [f'{name}_mm'], name=name))
        qdq(nodes, f'{name}_mm', name, f'{name}o')
        weights = np.random.default_rng(seed).integers(-3, 4, size=(k, n), dtype=np.int8)
        inits.extend(
            [*qparams(f'{name}_w'), *qparams(f'{name}o', frac=6), numpy_helper.from_array(weights, f'{name}_w_q')]
        )
        return name

    dense('x', 'a', 16, 32, 0)
    inits.append(numpy_helper.from_array(np.array([16, 16], dtype=np.int64), 'halves'))
    nodes.append(helper.make_node('Split', ['a', 'halves'], ['a0', 'a1'], axis=1, name='heads'))
    outputs = [dense('a0', 'h0', 16, 16, 1), dense('a1', 'h1', 16, 16, 2)]
    if whole_reader:
        outputs.append(dense('a', 'w', 32, 16, 3))
    return make_model(
        'split_heads',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [16, 16])],
        outputs=[(name, TensorProto.FLOAT, [16, 16]) for name in outputs],
        initializers=inits,
    )


@pytest.mark.parametrize('whole_reader', [False, True], ids=['split-only', 'also-read-whole'])
def test_a_split_along_the_producer_ports_hands_each_part_over_in_shared_memory(whole_reader, tmp_path):
    """Each half of a two-chain Dense's output is one of its ports: read by its half's Dense alone, the hand-over is
    one-to-one and placement shares it; read whole by another Dense as well, each port is broadcast and copied."""
    directives = {'a': parallelism(2), 'h0': parallelism(1), 'h1': parallelism(1)}
    if whole_reader:
        directives['w'] = parallelism(1)
    ctx = lower(_split_heads_model(whole_reader), tmp_path, directives, aie_config={'Optimize': 'latency'})
    realizations = {
        e['target'].split('.')[0]: e['realization']
        for e in ctx.ir.physical.plan['direct_edges']
        if e['source'].startswith('a_aie.')
    }
    expected = 'dma' if whole_reader else 'shared_memory'
    assert realizations['h0_aie'] == realizations['h1_aie'] == expected


def _reach_graph(producer_row: int, shareable: bool) -> GraphSpec:
    """A one-tile producer writing its own tile at (4, `producer_row`), a one-tile consumer at (4, 2) whose input
    sits in its west tile unless it reads its producer's buffer in shared memory, wherever its core reaches it, and a
    third op writing its own tile on that west tile, (3, 2)."""
    out = (BufferLocation('out1', 0, 0, 0, (0, 3)),)
    specs = {
        'p': _one_tile('p', out, (4, producer_row), frozenset({'out1'})),
        'c': _one_tile('c', (BufferLocation('in1', 0, -1, 0, (0, 3), reader=(0, 0)),), (4, 2)),
        'z': _one_tile('z', out, (3, 2), frozenset({'out1'})),
    }
    return GraphSpec(specs, [EdgeSpec('p', 'c', 'x', True, 'out1', 'in1', ((0, 0),), shareable=shareable)], PORTS)


@pytest.mark.parametrize('dy', [1, -1], ids=['north', 'south'])
def test_a_consumer_shares_the_buffer_its_core_reaches_north_or_south(dy):
    """The consumer one row below or above its producer reads the producer's buffer in place, so its west tile holds
    nothing of it: another op may sit there. Two rows away its core reaches nothing of it, so it may not sit there;
    handed over through a DMA, its buffer takes its west tile."""
    _, shares, _ = _place(_reach_graph(2 + dy, shareable=True), 8, 5)
    assert shares == {('c', 'in1', 0)}
    for graph in (_reach_graph(2 + 2 * dy, shareable=True), _reach_graph(2 + dy, shareable=False)):
        with pytest.raises(PlacementInfeasibleError):
            _place(graph, 8, 5)


def _below_column(rows: int, ports) -> GraphSpec:
    """`rows` one-tile ops stacked in a one-column array, each reading a PLIO stream into its own tile: the streams
    all climb the column from under the array."""
    ops = {f'op{r}': _one_tile(f'op{r}', (BufferLocation('in1', 0, 0, 0, (0, 3)),), (0, r)) for r in range(rows)}
    return GraphSpec(ops, [], dict(ports), below={name: (('inputs', 'in1', 0),) for name in ops})


def test_placement_routes_the_streams_from_below_within_the_switch_ports():
    """Three PLIO streams climb one column: the switch under it passes all three north, one past two ports."""
    with pytest.raises(PlacementInfeasibleError, match='streams within the switches'):
        _place(_below_column(3, {**PORTS, 'North': 2}), 1, 3)
    _place(_below_column(3, {**PORTS, 'North': 3}), 1, 3)


def _crossing_graph(streams: int, ports=None):
    """`streams` producers pinned west of their consumers on a one-row array: every hand-over crosses the link
    between columns `streams - 1` and `streams` eastward. `ports` are the switches' ports each way."""
    specs, edges = {}, []
    for i in range(streams):
        specs[f'p{i}'] = _one_tile(f'p{i}', (BufferLocation('out1', 0, 0, 0, (0, 3)),), (i, 0), frozenset({'out1'}))
        specs[f'q{i}'] = _one_tile(f'q{i}', (BufferLocation('in1', 0, 0, 0, (1, 2)),), (streams + i, 0))
        edges.append(EdgeSpec(f'p{i}', f'q{i}', direct=True, src_group='out1', dst_group='in1', port_pairs=((0, 0),)))
    return GraphSpec(specs, edges, dict(ports or PORTS), dma_channels=2)
