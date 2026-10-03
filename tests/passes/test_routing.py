"""Stream routes (`transport.routing`): the switch-port counts placement is held to, calibrated on server builds."""

from __future__ import annotations

from aie4ml.passes.transport.routing import BELOW, route_overflow

PORTS = {'North': 6, 'South': 4, 'East': 4, 'West': 4}  # every generation (mlir-aie AIETargetModel)
COLUMNS = 38


def test_one_column_of_eight_chains_spreads_into_its_neighbours():
    """A Dense of 8 chains reading and writing a memory tile: 8 streams each way through one column, past its 6
    north and 4 south ports, which routed on AIE-ML (bench kit aie-ml_dense_max)."""
    streams = {('in', row): (BELOW, [(4, row)]) for row in range(8)}
    streams.update({('out', row): ((5, row), [BELOW]) for row in range(8)})
    assert route_overflow(streams, PORTS, COLUMNS) is None


def test_two_columns_of_memory_tile_readers_overflow():
    """A 2x8 matmul whose two operands each reach every tile from a memory tile: 32 streams into two columns, which
    the router could not route (ViT AV, 'AIE Router failed to find a legal solution')."""
    streams = {(operand, row, col): (BELOW, [(8 + col, row)]) for operand in 'ab' for row in range(8) for col in range(2)}
    assert 'north' in route_overflow(streams, PORTS, COLUMNS)


def test_a_broadcast_holds_one_port_per_link():
    """One stream to all 8 tiles of a column holds one north port per link, beside 5 others: 6 fit, 7 do not
    (a one-column array, so no neighbour takes any)."""
    streams = {'broadcast': (BELOW, [(0, row) for row in range(8)])}
    streams.update({i: (BELOW, [(0, 7)]) for i in range(5)})
    assert route_overflow(streams, PORTS, 1) is None
    streams[5] = (BELOW, [(0, 7)])
    assert 'needs 7 streams north' in route_overflow(streams, PORTS, 1)
