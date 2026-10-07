"""The whole-graph parallelism search (AIEConfig `Optimize`): what it chooses, what it counts, and what it refuses."""

from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.frontends.onnx import from_onnx
from aie4ml.op_impls import get_family_resolver_registry
from aie4ml.op_impls.families.conv2d.conv2d import Conv2dOpImplVariant
from aie4ml.passes import choose_parallelism, placement
from aie4ml.passes.resolve import resolve_instance
from aie4ml.passes.transport.collect import TransportCollector
from aie4ml.passes.transport.legality import direct_pairing
from frontends.test_onnx_aie1 import _dense_model, _normalization_chain_model
from frontends.test_onnx_aie1_views import _split_model
from helpers import (
    PART,
    TensorProto,
    assert_aie_matches_onnx,
    dq,
    helper,
    lower,
    make_model,
    memtiles,
    numpy_helper,
    qdq,
    qparams,
)
from ops.test_conv2d import AIE1_PART, MLV2_PART, H, W, _conv, _start, _strided_chain_model

PARTS = {'aie1': AIE1_PART, 'aie-ml': PART, 'aie-mlv2': MLV2_PART}


def _wide_model():
    """1x1 8->16, 3x3 16->64 and 1x1 64->8: the 3x3's 9 KB of weights overflow AIE1's 8 KB bank, and however it
    splits, a neighbour must match it for the hand-over to stay direct."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a0', 'c0', 8, 16, 1, pad=0, relu=True, seed=1)
    _conv(nodes, inits, 'a0', 'a1', 'c1', 16, 64, 3, pad=1, relu=True, seed=2)
    _conv(nodes, inits, 'a1', 'a2', 'c2', 64, 8, 1, pad=0, relu=False, seed=3)
    nodes.append(helper.make_node('Transpose', ['a2'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    return make_model(
        'wide',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, 8])],
        outputs=[('y', TensorProto.FLOAT, [1, H, W, 8])],
        initializers=inits,
    )


def _branch_model():
    """One kernel's frame read by two convs, each ending at its own graph output."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a0', 'c0', 8, 16, 1, pad=0, relu=True, seed=1)
    for i in (1, 2):
        _conv(nodes, inits, 'a0', f'a{i}', f'c{i}', 16, 8, 3, pad=1, relu=True, seed=1 + i)
        nodes.append(helper.make_node('Transpose', [f'a{i}'], [f'y{i}'], perm=[0, 2, 3, 1], name=f'y{i}_nhwc'))
    return make_model(
        'branch',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, 8])],
        outputs=[(f'y{i}', TensorProto.FLOAT, [1, H, W, 8]) for i in (1, 2)],
        initializers=inits,
    )


def _placed_tiles(ctx) -> int:
    footprints = (inst.variant.footprint(inst.node, inst.config) for inst in ctx.ir.execution)
    return sum(f.width * f.height for f in footprints)


def _splits(ctx) -> dict:
    return {inst.name: inst.config.parallelism for inst in ctx.ir.execution}


def test_resource_splits_only_what_does_not_fit_and_keeps_every_edge_direct(tmp_path):
    ctx = lower(_wide_model(), tmp_path, part=AIE1_PART, aie_config={'Optimize': 'resource'})
    chosen = ctx.ir.optimizer
    # The 3x3 cascades over two tiles and the 1x1 before it matches the split: every hand-over shares a buffer,
    # where splitting the 3x3's output channels would copy the 1x1's output to both chains.
    assert (chosen['tiles'], chosen['memtile_legs']) == (5, 0) == (_placed_tiles(ctx), 0)
    assert {name: (p.cas_num, p.cas_length) for name, p in _splits(ctx).items()} == {
        'c0_aie': (2, 1),
        'c1_aie': (1, 2),
        'c2_aie': (1, 1),
    }
    # Where everything fits, one tile per layer.
    ctx = lower(_wide_model(), tmp_path / 'ml', part=PART, aie_config={'Optimize': 'resource'})
    assert ctx.ir.optimizer['tiles'] == 3


@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
def test_performance_splits_the_busiest_layers_within_max_tiles(tmp_path, part):
    base = lower(_wide_model(), tmp_path / 'base', part=part, aie_config={'Optimize': 'resource'})
    ctx = lower(_wide_model(), tmp_path, part=part, aie_config={'Optimize': 'performance', 'MaxTiles': 8})
    chosen = ctx.ir.optimizer
    busiest = max(inst.variant.work(inst.node, inst.config) for inst in base.ir.execution)
    assert max(inst.variant.work(inst.node, inst.config) for inst in ctx.ir.execution) < busiest
    assert chosen['tiles'] == _placed_tiles(ctx) <= 8 and chosen['memtile_legs'] == 0
    assert chosen['parallelism'] == {name: vars(p) for name, p in _splits(ctx).items()}


@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
def test_a_frame_read_twice_is_placed_and_verified(tmp_path, part):
    """A branch keeps two readers of one frame in the search state at once; the design still places and verifies."""
    ctx = lower(_branch_model(), tmp_path, part=part, aie_config={'Optimize': 'performance', 'MaxTiles': 6})
    assert ctx.ir.optimizer['tiles'] == _placed_tiles(ctx) <= 6
    assert set(ctx.ir.optimizer['parallelism']) == {'c0_aie', 'c1_aie', 'c2_aie'}


@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_layout_conversions_count_toward_max_tiles(tmp_path, part):
    """A stride-2 conv reading a kernel's frame adds a retiler kernel, which the budget counts, and whose legs are
    decided as transport decides them (AIE1 has no memory tile to fall back on)."""
    ctx = lower(_strided_chain_model(), tmp_path, part=part, aie_config={'Optimize': 'resource'})
    kernels = {inst.name for inst in ctx.ir.execution}
    assert len(kernels) == 3  # first, the retiler, second
    assert ctx.ir.optimizer['tiles'] == _placed_tiles(ctx) == 3
    with pytest.raises(ConfigRefused, match='no design resolves within the tile budget'):
        lower(_strided_chain_model(), tmp_path / 'tight', part=PART, aie_config={'Optimize': 'resource', 'MaxTiles': 2})


def test_every_leg_is_decided_as_transport_decides_it(tmp_path):
    """Boundary and kernel legs through a memory tile are counted as the built plan routes them, and a split view
    steers its producer to ports whose slices it can read without a directive."""
    ctx = lower(_normalization_chain_model(), tmp_path / 'norm', part=MLV2_PART)
    assert ctx.ir.optimizer['memtile_legs'] == len(memtiles(ctx)) == 1  # the sum re-staged between two kernels
    ctx = lower(_split_model(), tmp_path / 'split', part=AIE1_PART, aie_config={'Optimize': 'resource'})
    assert _splits(ctx)['root_aie'].contract == 'outer' and _splits(ctx)['root_aie'].cas_num == 2


def test_a_rerun_searches_afresh_and_leaves_the_directives_as_given(tmp_path):
    model = from_onnx(
        _wide_model(),
        {'Part': PART, 'AIEConfig': {'Iterations': 1, 'Optimize': 'resource'}},
        output_dir=tmp_path,
        project_name='rerun',
    )
    ctx = model.run_pipeline().context
    assert ctx.ir.optimizer['tiles'] == 3
    ctx.aie_config.update({'Optimize': 'performance', 'MaxTiles': 8})
    model.run_pipeline()
    assert 3 < ctx.ir.optimizer['tiles'] == _placed_tiles(ctx) <= 8
    assert all(node.directives == {} for node in ctx.ir.logical if not node.is_folded_view)


def test_user_parallelism_constrains_the_choice_and_a_short_budget_names_the_refusals(tmp_path):
    """A partial directive fixes what it names and leaves the rest to the search."""
    directives = {'c1': {'parallelism': {'cas_num': 2}}}
    ctx = lower(_wide_model(), tmp_path, directives, part=PART, aie_config={'Optimize': 'resource'})
    assert ctx.ir.optimizer['parallelism']['c1_aie']['cas_num'] == _splits(ctx)['c1_aie'].cas_num == 2
    with pytest.raises(ConfigRefused, match=r'(?s)MaxTiles=3.*c2_aie: .*option\(s\) refused'):
        lower(_wide_model(), tmp_path / 'short', part=AIE1_PART, aie_config={'Optimize': 'resource', 'MaxTiles': 3})


def test_a_design_placement_refuses_gives_way_to_the_next(tmp_path, monkeypatch):
    place = placement.PlaceKernels.transform
    calls = []

    def refuse_first(self, model_or_ctx):
        calls.append(1)
        if len(calls) == 1:
            raise placement.PlacementInfeasibleError('no placement for this one')
        return place(self, model_or_ctx)

    monkeypatch.setattr(placement.PlaceKernels, 'transform', refuse_first)
    ctx = lower(_wide_model(), tmp_path, part=PART, aie_config={'Optimize': 'performance', 'MaxTiles': 8})
    assert ctx.ir.optimizer['designs_tried'] == 2


def test_a_search_that_builds_nothing_says_whether_it_hit_a_limit(tmp_path, monkeypatch):
    def refuse(self, model_or_ctx):
        raise placement.PlacementInfeasibleError('no placement')

    monkeypatch.setattr(placement.PlaceKernels, 'transform', refuse)
    pinned = {'dense': {'parallelism': {'cas_num': 1, 'cas_length': 1, 'contract': 'inner'}}}
    with pytest.raises(ConfigRefused, match=r'none of the 1 designs within the tile budget can be built'):
        lower(_dense_model(), tmp_path / 'one', pinned, part=PART)
    monkeypatch.setattr(choose_parallelism, 'MAX_PLACEMENT_TRIALS', 2)
    with pytest.raises(RuntimeError, match=r'(?s)search limit.*A design may still exist'):
        lower(_dense_model(), tmp_path / 'many', part=PART)


def test_an_error_that_is_not_a_refusal_stops_the_search(tmp_path, monkeypatch):
    def broken(self, node, config):
        raise KeyError('a bug, not a configuration the kernel refuses')

    monkeypatch.setattr(Conv2dOpImplVariant, 'work', broken)
    with pytest.raises(KeyError, match='a bug'):
        lower(_wide_model(), tmp_path, part=PART, aie_config={'Optimize': 'performance'})


def test_dense_offers_splits_its_tiling_pads(tmp_path):
    """40 output features split three ways pad each chain to 16: a candidate a divisor rule would miss."""
    model = from_onnx(
        _dense_model(out_features=40),
        {'Part': PART, 'AIEConfig': {'Iterations': 1}},
        output_dir=tmp_path,
        project_name='dense',
    )
    ctx = model.context
    (node,) = [n for n in ctx.ir.logical if n.op_type == 'dense']
    split = {'contract': 'inner', 'cas_num': 3, 'cas_length': 1}
    assert split in get_family_resolver_registry().get('dense').parallelism_candidates(node, ctx.device)
    inst = resolve_instance(node, ctx.device, {}, {'parallelism': split})
    assert inst.config.io_views[node.outputs[0].name].tile[-1] * 3 > 40


def test_latency_mode_ranks_latency_before_interval():
    """Of a design that finishes the first inference sooner and one that takes the next one sooner, 'latency' ranks
    the first best and 'performance' the second."""
    soon = choose_parallelism._Design(0, 8, 0, 2000, 5000, (), (), ())
    steady = choose_parallelism._Design(0, 8, 0, 1000, 9000, (), (), ())
    search = object.__new__(choose_parallelism._Search)
    for mode, best in (('latency', soon), ('performance', steady)):
        search.mode = mode
        assert min((soon, steady), key=search.rank) is best


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Optimize='fast'"):
        lower(_wide_model(), tmp_path, part=PART, aie_config={'Optimize': 'fast'})


def _encoder_model(tokens=16, features=64, ffn=128):
    """One pre-LN transformer block: LayerNorm, Q/K/V, Q.K^T, Softmax, .V, projection and residual Add, then
    LayerNorm, a ReLU MLP and residual Add. Q, K and V alive at once multiply the search's states."""
    rng = np.random.default_rng(0)
    nodes: list = []
    inits: list = [*qparams('x')]
    dq(nodes, 'x_q', 'x', 'x')

    def dense(src, name, k, n, relu=False):
        dq(nodes, f'{name}_wq', f'{name}_w', f'{name}_w')
        nodes.append(helper.make_node('MatMul', [src, f'{name}_w'], [f'{name}_mm'], name=name))
        out = f'{name}_mm'
        if relu:
            nodes.append(helper.make_node('Relu', [out], [f'{name}_relu']))
            out = f'{name}_relu'
        qdq(nodes, out, f'{name}_out', f'{name}_o')
        inits.extend([*qparams(f'{name}_w', frac=6), *qparams(f'{name}_o', unsigned=relu)])
        inits.append(numpy_helper.from_array(rng.integers(-3, 4, (k, n), dtype=np.int8), f'{name}_wq'))
        return f'{name}_out'

    def layernorm(src, name):
        inits.extend(
            [
                numpy_helper.from_array(np.ones(features, np.float32), f'{name}_g'),
                numpy_helper.from_array(np.zeros(features, np.float32), f'{name}_b'),
            ]
        )
        nodes.append(
            helper.make_node(
                'LayerNormalization', [src, f'{name}_g', f'{name}_b'], [f'{name}_ln'], name=name, epsilon=2.0**-8
            )
        )
        qdq(nodes, f'{name}_ln', f'{name}_out', f'{name}_o')
        inits.extend(qparams(f'{name}_o', frac=5))
        return f'{name}_out'

    def add(a, b, name):
        nodes.append(helper.make_node('Add', [a, b], [f'{name}_sum'], name=name))
        qdq(nodes, f'{name}_sum', f'{name}_out', f'{name}_o')
        inits.extend(qparams(f'{name}_o'))
        return f'{name}_out'

    h = layernorm('x', 'ln1')
    q, k, v = (dense(h, name, features, features) for name in 'qkv')
    nodes.append(helper.make_node('Transpose', [k], ['k_t'], perm=[1, 0], name='k_t'))
    nodes.append(helper.make_node('MatMul', [q, 'k_t'], ['scores_mm'], name='scores'))
    qdq(nodes, 'scores_mm', 's', 's')
    nodes.append(helper.make_node('Softmax', ['s'], ['sm'], name='softmax', axis=-1))
    qdq(nodes, 'sm', 'p', 'p')
    nodes.append(helper.make_node('MatMul', ['p', v], ['ctx_mm'], name='ctx'))
    qdq(nodes, 'ctx_mm', 'c', 'c')
    x = add('x', dense('c', 'proj', features, features), 'res1')
    y = add(x, dense(dense(layernorm(x, 'ln2'), 'fc1', features, ffn, relu=True), 'fc2', ffn, features), 'res2')
    inits.extend([*qparams('s', frac=3), *qparams('p', frac=8, unsigned=True), *qparams('c')])
    return make_model(
        'encoder',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [tokens, features])],
        outputs=[(y, TensorProto.FLOAT, [tokens, features])],
        initializers=inits,
    )


def test_row_chains_gather_through_ordered_merges(tmp_path):
    """K and V in two row chains feed single-chain Q.K^T and .V: each a gather, which an ordered packet merge makes
    direct where the part has one, as nothing else reads K or V -- and nothing else does (AIE1 has none). LayerNorm's
    two chains feed a single-chain Q too, but also K and V: a merge's producer streams carry packet headers ADF would
    hand those plain readers, so that gather takes a memory tile."""
    split = {'parallelism': {'cas_num': 2, 'cas_length': 1, 'contract': 'outer'}}
    whole = {'parallelism': {'cas_num': 1, 'cas_length': 1, 'contract': 'outer'}}
    one = {'parallelism': {'cas_num': 1, 'cas_length': 1, 'contract': 'inner'}}
    directives = {
        'ln1': split, 'k': split, 'v': split, 'q': one, 'scores': whole, 'softmax': whole, 'ctx': whole,
        'proj': one, 'ln2': one, 'fc1': one, 'fc2': one,
    }  # fmt: skip
    ctx = lower(_encoder_model(), tmp_path, directives)
    plan = ctx.ir.physical.plan
    assert [buffer['tensor'] for buffer in plan['buffers']] == ['ln1_ln']
    assert sorted((m['tensor'], len(m['writers']), len(m['readers'])) for m in plan['merges']) == [
        ('k_mm', 2, 1),
        ('v_mm', 2, 1),
    ]
    (leg,) = [
        leg
        for entry in TransportCollector(ctx.ir.execution).collect()
        for leg in entry.consumers
        if entry.producer.node is not None
        and (entry.producer.node.name, leg.consumer.node.name) == ('k_aie', 'scores_aie')
    ]
    assert direct_pairing(ctx.ir.execution, leg.logical_tensor, leg.producer, leg.consumer, merge=True) == (
        None,
        ((0, 0), (1, 0)),
    )
    assert direct_pairing(ctx.ir.execution, leg.logical_tensor, leg.producer, leg.consumer, merge=False)[0]


def test_a_direct_route_one_end_asks_for_is_honoured(tmp_path):
    """io_route names a mode at one end; the other end's default leaves it in force. LayerNorm's gather into the
    single-chain Q takes a memory tile (above): asked to be direct, it is refused rather than staged."""
    split = {'parallelism': {'cas_num': 2, 'cas_length': 1, 'contract': 'outer'}}
    whole = {'parallelism': {'cas_num': 1, 'cas_length': 1, 'contract': 'outer'}}
    one = {'parallelism': {'cas_num': 1, 'cas_length': 1, 'contract': 'inner'}}
    directives = {
        'ln1': split, 'k': split, 'v': split, 'q': {**one, 'io_route': {'inputs': {'ln1_ln': 'direct'}}},
        'scores': whole, 'softmax': whole, 'ctx': whole, 'proj': one, 'ln2': one, 'fc1': one, 'fc2': one,
    }  # fmt: skip
    with pytest.raises(ConfigRefused, match='io_route=direct requested'):
        lower(_encoder_model(), tmp_path, directives)


def test_a_transformer_block_is_searched_in_both_modes(tmp_path):
    """Q, K and V alive at once once overflowed the search, and a LayerNorm left to split itself cut its rows below
    a microtile band; both modes now find a design, and 'performance' splits the row-wise layers by rows."""
    ctx = lower(_encoder_model(), tmp_path / 'resource', aie_config={'Optimize': 'resource'})
    assert ctx.ir.optimizer['tiles'] == _placed_tiles(ctx)
    budget = 2 * ctx.ir.optimizer['tiles']
    ctx = lower(_encoder_model(), tmp_path / 'performance', aie_config={'Optimize': 'performance', 'MaxTiles': budget})
    assert ctx.ir.optimizer['tiles'] == _placed_tiles(ctx) <= budget
    assert _splits(ctx)['softmax_aie'].cas_num > 1


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
def test_an_optimized_design_runs_on_the_aie(tmp_path, part):
    """The chosen multi-tile design through the AIE compiler and aiesim, bit-exact against ONNX Runtime."""
    feeds = np.random.default_rng(21).integers(-40, 40, size=(2, 1, H, W, 8), dtype=np.int8)
    assert_aie_matches_onnx(
        _wide_model(),
        {'x_q': feeds},
        {},
        tmp_path,
        max_code_diff=0,
        part=part,
        iterations=2,
        per_iteration=True,
        aie_config={'Optimize': 'performance', 'MaxTiles': 8},
    )
