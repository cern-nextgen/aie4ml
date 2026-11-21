from __future__ import annotations

from types import SimpleNamespace

from aie4ml.cost.chain import Link
from aie4ml.cost.design import Chain, Edge, Transport, graph_edges, latency

LINK = Link(-2, 4)
TRANSPORT = Transport(
    bytes_per_cycle=4.0, boundary_bytes_per_cycle=2.0, boundary_in=5.0, boundary_out=7.0, lock=3.0, dma=11.0
)


def test_a_kernel_starts_when_its_input_arrives_and_its_output_then_leaves():
    kernel = Chain(['k'], [[(100.0, 'end')]], [10.0])
    edges = [Edge(None, 'k', 400, False), Edge('k', None, 100, False)]
    assert latency([kernel], edges, LINK, TRANSPORT) == (5 + 400 / 2) + (10 + 100) + (7 + 100 / 2)


def test_a_layer_waits_for_its_producer_and_the_handoff():
    first = Chain(['a'], [[(50.0, 'end')]], [0.0])
    second = Chain(['b'], [[(20.0, 'end')]], [0.0])
    arrive = 5 + 40 / 2
    shared = [Edge(None, 'a', 40, False), Edge('a', 'b', 80, True), Edge('b', None, 8, False)]
    assert latency([first, second], shared, LINK, TRANSPORT) == arrive + 50 + 3 + 20 + (7 + 8 / 2)
    copied = [Edge(None, 'a', 40, False), Edge('a', 'b', 80, False), Edge('b', None, 8, False)]
    assert latency([first, second], copied, LINK, TRANSPORT) == arrive + 50 + (11 + 80 / 4) + 20 + (7 + 8 / 2)


def _layer(name, lhs, out, in_ports, out_ports, tiles):
    int8 = SimpleNamespace(width=8)
    return SimpleNamespace(
        name=name,
        node=SimpleNamespace(roles={lhs: 'lhs'}),
        ports=SimpleNamespace(
            inputs={lhs: SimpleNamespace(group='in1', endpoints=in_ports)},
            outputs={out: SimpleNamespace(group='out1', endpoints=out_ports)},
        ),
        port_views={lhs: SimpleNamespace(tile=tiles[0]), out: SimpleNamespace(tile=tiles[1])},
        config=SimpleNamespace(precision={'lhs': int8, 'output': int8}),
    )


def test_the_physical_plan_gives_each_handoff_from_its_producer_kernels_to_its_consumer_kernels():
    # L1: two chains reading one broadcast input; L2: two chains, each reading one of L1's outputs
    l1 = _layer(
        'L1', 'x', 'r', (('kk[0].in[0]', 'kk[1].in[0]'),), (('kk[0].out[0]',), ('kk[1].out[0]',)), ([8, 128], [8, 32])
    )
    l2 = _layer('L2', 'r', 'y', (('kk[0].in[0]',), ('kk[1].in[0]',)), (('kk[1].out[0]',),), ([8, 32], [8, 16]))
    plan = {
        'buffers': [],
        'direct_edges': [
            {'source': 'ifm[0]', 'target': 'L1.in1[0]', 'tensor': 'x'},
            {'source': 'L1.out1[0]', 'target': 'L2.in1[0]', 'tensor': 'r', 'realization': 'shared_memory'},
            {'source': 'L1.out1[1]', 'target': 'L2.in1[1]', 'tensor': 'r', 'realization': 'dma'},
            {'source': 'L2.out1[0]', 'target': 'ofm[0]', 'tensor': 'y'},
        ],
    }
    assert graph_edges([l1, l2], SimpleNamespace(plan=plan)) == [
        Edge(None, 'L1.kk[0]', 1024, False),
        Edge(None, 'L1.kk[1]', 1024, False),
        Edge('L1.kk[0]', 'L2.kk[0]', 256, True),
        Edge('L1.kk[1]', 'L2.kk[1]', 256, False),
        Edge('L2.kk[1]', None, 128, False),
    ]


def test_a_buffer_staged_in_a_memory_tile_is_handed_on_once_every_writer_has_copied_into_it():
    # L1's two chains write their halves of r into a memory tile, which L2 reads whole
    l1 = _layer(
        'L1', 'x', 'r', (('kk[0].in[0]', 'kk[1].in[0]'),), (('kk[0].out[0]',), ('kk[1].out[0]',)), ([8, 64], [8, 16])
    )
    l2 = _layer('L2', 'r', 'y', (('kk[0].in[0]',),), (('kk[0].out[0]',),), ([8, 32], [8, 8]))
    plan = {
        'direct_edges': [
            {'source': 'ifm[0]', 'target': 'L1.in1[0]', 'tensor': 'x'},
            {'source': 'L2.out1[0]', 'target': 'ofm[0]', 'tensor': 'y'},
        ],
        'buffers': [
            {
                'name': 'buffer_r',
                'writers': [{'source': 'L1.out1[0]'}, {'source': 'L1.out1[1]'}],
                'readers': [{'target': 'L2.in1[0]'}],
            }
        ],
    }
    handoffs = graph_edges([l1, l2], SimpleNamespace(plan=plan))
    assert handoffs[-3:] == [
        Edge('L1.kk[0]', 'memtile:buffer_r', 128, False),
        Edge('L1.kk[1]', 'memtile:buffer_r', 128, False),
        Edge('memtile:buffer_r', 'L2.kk[0]', 256, False),
    ]
    slow, fast, second = (
        Chain([name], [[(cycles, 'end')]], [0.0])
        for name, cycles in [('L1.kk[0]', 90.0), ('L1.kk[1]', 50.0), ('L2.kk[0]', 20.0)]
    )
    arrive = 5 + 512 / 2
    staged = arrive + 90 + (11 + 128 / 4)  # the slower writer's copy
    assert latency([slow, fast, second], handoffs, LINK, TRANSPORT) == staged + (11 + 256 / 4) + 20 + (7 + 64 / 2)
