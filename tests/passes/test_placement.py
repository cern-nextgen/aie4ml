"""Focused placement legality tests."""

import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls.base import BufferLocation, row_flow
from aie4ml.passes.placement import (
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    Placed,
    PlacementInfeasibleError,
    PortFace,
    Rect,
    _beam_place,
    _placements_conflict,
)
from helpers import PART, lower
from passes.test_choose_parallelism import _wide_model


def _one_tile(name, col, row, locations):
    return Placed(
        name, col, row, Rect(1, 1, PortFace('left', 0, 0), PortFace('right', 0, 0), buffer_locations=locations)
    )


@pytest.mark.parametrize('row', [0, 1], ids=['even-row', 'odd-row'])
def test_neighbours_share_only_an_exactly_aliased_buffer(row):
    """Data flows left to right on every AIE row; the hand-over buffer lives in the tile both kernels
    reach -- the producer's on an even row, the consumer's on an odd one. Neighbours may touch each
    other's memory only through that one buffer, bank for bank."""
    flow = row_flow(True, row, 1)
    producer = _one_tile('producer', 7, row, lambda r: (BufferLocation('out', 0, flow.output_col, 0, (0, 3)),))
    consumer = _one_tile('consumer', 8, row, lambda r: (BufferLocation('in', 0, flow.input_col, 0, (0, 3)),))
    edge = EdgeSpec('producer', 'consumer', direct=True, src_group='out', dst_group='in', port_pairs=((0, 0),))
    graph = GraphSpec([], {}, [edge], {}, {})
    assert not _placements_conflict(producer, consumer, graph)

    # Any other op on the tile holding that buffer conflicts: only the edge's other end may be there.
    no_edge = GraphSpec([], {}, [], {}, {})
    if flow.output_col == 0:  # the producer's tile, west of the consumer
        assert _placements_conflict(_one_tile('other', 7, row, None), consumer, no_edge)
    else:  # the consumer's tile, east of the producer
        assert _placements_conflict(producer, _one_tile('other', 8, row, None), no_edge)

    other_banks = _one_tile('consumer', 8, row, lambda r: (BufferLocation('in', 0, flow.input_col, 0, (1, 2)),))
    assert _placements_conflict(producer, other_banks, graph)


def test_placement_starts_next_to_the_plio_columns_and_widens_only_when_it_must(tmp_path):
    def columns(ctx):
        return sorted(p['col'] for p in ctx.ir.physical.placements.values())

    assert columns(lower(_wide_model(), tmp_path / 'default', part=PART)) == [5, 6, 7]  # VE2802's first PL column: 5
    # One tile from the preferred column on: the design widens into the columns before it.
    narrow = {'Columns': 7, 'Rows': 1, 'PLColumnStart': 6}
    assert columns(lower(_wide_model(), tmp_path / 'narrow', part=PART, aie_config=narrow)) == [1, 2, 3]
    # A ColumnStart the user gives bounds the region.
    with pytest.raises(ConfigRefused):
        lower(_wide_model(), tmp_path / 'held', part=PART, aie_config={**narrow, 'ColumnStart': 6})


def _column_op(name, index, rows):
    """A one-column op of `rows` kernels, as AIE-ML's dense ops: each reads from its west neighbour's memory and
    writes to its own."""

    def locations(_row):
        return tuple(
            location
            for row in range(rows)
            for location in (BufferLocation('in1', row, -1, row, (0, 3)), BufferLocation('out1', row, 0, row, (0, 3)))
        )

    face = PortFace('left', 0, rows - 1), PortFace('right', 0, rows - 1)
    return NodeSpec(None, name, index, Rect(1, rows, *face, buffer_locations=locations))


def _tail_graph():
    """Five eight-row ops handing over through shared buffers, a sixth fed by a memory tile, then a five-row op
    feeding a three-row one, which may share their buffer."""
    rows = {**{f'm{k}': 8 for k in range(6)}, 't5': 5, 't3': 3}
    specs = {name: _column_op(name, index, height) for index, (name, height) in enumerate(rows.items())}

    def edge(src, dst, share):
        pairs = tuple((port, port) for port in range(rows[dst])) if share else ()
        return EdgeSpec(src, dst, direct=share, src_group='out1', dst_group='in1', port_pairs=pairs, shareable=share)

    edges = [edge(f'm{k}', f'm{k + 1}', k < 4) for k in range(5)] + [edge('m5', 't5', False), edge('t5', 't3', True)]
    preds, succs = {name: [] for name in rows}, {name: [] for name in rows}
    for e in edges:
        preds[e.dst].append(e.src)
        succs[e.src].append(e.dst)
    return GraphSpec(list(rows), specs, edges, preds, succs)


def _three_branch_graph():
    """An eight-row source feeds three independent chains of four three-row ops on an eight-row array."""
    specs = {'src': _column_op('src', 0, 8)}
    edges = []
    for branch in range(3):
        previous = 'src'
        for stage in range(4):
            name = f'b{branch}_{stage}'
            specs[name] = _column_op(name, len(specs), 3)
            is_broadcast = previous == 'src'
            edges.append(
                EdgeSpec(
                    previous,
                    name,
                    direct=True,
                    one_to_one=not is_broadcast,
                    src_group='out1',
                    dst_group='in1',
                    port_pairs=tuple((port, port) for port in range(3)),
                    shareable=not is_broadcast,
                )
            )
            previous = name
    preds = {name: [] for name in specs}
    succs = {name: [] for name in specs}
    for edge in edges:
        preds[edge.dst].append(edge.src)
        succs[edge.src].append(edge.dst)
    return GraphSpec(list(specs), specs, edges, preds, succs)


def test_placement_packs_two_branches_under_the_third():
    graph = _three_branch_graph()
    placed = _beam_place(graph, 11, 8, 1.0, 0.05, 16)

    starts = []
    for branch in range(3):
        chain = [placed[f'b{branch}_{stage}'] for stage in range(4)]
        assert {p.y for p in chain} == {chain[0].y}
        assert [p.x for p in chain] == list(range(chain[0].x, chain[0].x + 4))
        starts.append((chain[0].x, chain[0].y))

    lanes = {}
    for col, row in starts:
        lanes.setdefault(col, []).append(row)
    assert len(lanes) == 2
    assert sorted(sorted(rows) for rows in lanes.values()) == [[0], [0, 3]]


def test_placement_gives_up_a_shared_buffer_only_to_fit():
    """With room, the tail's ops sit side by side and share their buffer; one column narrower, they stack in one
    column and hand over through a DMA; narrower still, nothing fits."""
    side_by_side = _beam_place(_tail_graph(), 10, 8, 1.0, 0.05, 16)
    assert (side_by_side['t3'].x, side_by_side['t3'].y) == (side_by_side['t5'].x + 1, side_by_side['t5'].y)
    stacked = _beam_place(_tail_graph(), 9, 8, 1.0, 0.05, 16)
    assert stacked['t3'].x == stacked['t5'].x
    bands = sorted(((stacked[name].y, stacked[name].y + stacked[name].rect.h) for name in ('t5', 't3')))
    assert bands[0][0] == 0 and bands[0][1] == bands[1][0] and bands[1][1] == 8
    assert all(stacked[f'm{k + 1}'].x == stacked[f'm{k}'].x + 1 for k in range(4))  # the chain still shares
    with pytest.raises(PlacementInfeasibleError):
        _beam_place(_tail_graph(), 8, 8, 1.0, 0.05, 16)
