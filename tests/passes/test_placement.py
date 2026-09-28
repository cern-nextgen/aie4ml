"""Focused placement legality tests."""

import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls.base import BufferLocation, row_flow
from aie4ml.passes.placement import EdgeSpec, GraphSpec, Placed, PortFace, Rect, _placements_conflict
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
