"""One head's Q·Kᵀ -> Softmax -> ·V fused into the attention op. Its oracle is its own integer arithmetic: it divides
once where the QDQ graph rounds every probability, so against the graph it stays within a code or two."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from aie4ml import from_onnx
from aie4ml.errors import ConfigRefused
from aie4ml.ir.graph import input_tensor_for_role
from aie4ml.op_impls.families.attention.common import RECIP_TABLE, THETA_SHIFT
from helpers import (
    TensorProto,
    config_of,
    direct_edges,
    dq,
    helper,
    lower,
    make_model,
    memtiles,
    numpy_helper,
    onnx_session,
    parallelism,
    qdq,
    qparams,
)

AIE_ML = 'xilinx_vek280_base_202610_1'
AIE_MLV2 = 'xc2ve3858-ssva2112-2MP-e-S'
FEATURES, QKV_FRAC, OUT_FRAC = 16, 4, 5
BIAS_FRAC = {8: 2, 16: 6}  # U's fixed point, by its width


def _dense(nodes, inits, x, name, head_dim, seed, features=FEATURES):
    w = f'{name}_w'
    dq(nodes, f'{w}_q', w, w)
    nodes.append(helper.make_node('MatMul', [x, w], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', name, f'{name}o')
    weights = np.random.default_rng(seed).integers(-3, 4, size=(features, head_dim), dtype=np.int8)
    inits += [*qparams(w), *qparams(f'{name}o', frac=QKV_FRAC), numpy_helper.from_array(weights, f'{w}_q')]


def _head(tokens: int, head_dim: int, exported: bool = False, features: int = FEATURES, bias: int = 0):
    """x -> Q, K, V Dense -> Q·Kᵀ (int8, 2^-3) -> Softmax (uint8, 2^-8) -> ·V (int8, 2^-5). exported: as PyTorch
    exports it ((T, 1, D) tensors, Q / sqrt(head_dim) requantized, scores unquantized); bias: the bits of a bias U
    (input u_q) added to the exact scores, their sum quantized as the logits (exported: neither)."""
    nodes = []
    inits = [*qparams('x'), *qparams('so', frac=3), *qparams('po', frac=8, unsigned=True)]
    inits += qparams('co', frac=OUT_FRAC)
    dq(nodes, 'x_q', 'x', 'x')
    if bias:
        u_type = TensorProto.INT16 if bias == 16 else TensorProto.INT8
        inits += [
            helper.make_tensor('u_scale', TensorProto.FLOAT, [], [float(2.0 ** -BIAS_FRAC[bias])]),
            helper.make_tensor('u_zp', u_type, [], [0]),
        ]
        dq(nodes, 'u_q', 'u', 'u')
    x = 'x'
    if exported:
        nodes.append(helper.make_node('Transpose', ['x'], ['x_seq'], perm=[1, 0, 2], name='to_seq'))
        x = 'x_seq'
    for seed, name in enumerate('qkv', 1):
        _dense(nodes, inits, x, name, head_dim, seed, features)
    lhs, k_t, v = 'q', 'k_t', 'v'
    if exported:
        inits += [*qparams('qs', frac=6), numpy_helper.from_array(np.float32(np.sqrt(head_dim)), 'root')]
        nodes.append(helper.make_node('Div', ['q', 'root'], ['q_div'], name='q_scale'))
        qdq(nodes, 'q_div', 'q_s', 'qs')
        nodes.append(helper.make_node('Transpose', ['q_s'], ['q_b'], perm=[1, 0, 2], name='q_b'))
        nodes.append(helper.make_node('Transpose', ['k'], ['k_t'], perm=[1, 2, 0], name='k_t'))
        nodes.append(helper.make_node('Transpose', ['v'], ['v_b'], perm=[1, 0, 2], name='v_b'))
        lhs, v = 'q_b', 'v_b'
        nodes.append(helper.make_node('MatMul', [lhs, k_t], ['s_mm' if bias else 's'], name='scores'))
        if bias:
            nodes.append(helper.make_node('Add', ['s_mm', 'u'], ['s'], name='biased'))
    else:
        nodes.append(helper.make_node('Transpose', ['k'], ['k_t'], perm=[1, 0], name='k_t'))
        nodes.append(helper.make_node('MatMul', [lhs, k_t], ['scores_mm'], name='scores'))
        if bias:
            nodes.append(helper.make_node('Add', ['scores_mm', 'u'], ['biased_sum'], name='biased'))
        qdq(nodes, 'biased_sum' if bias else 'scores_mm', 's', 'so')
    nodes.append(helper.make_node('Softmax', ['s'], ['p_sm'], name='attn', axis=-1))
    qdq(nodes, 'p_sm', 'p', 'po')
    nodes.append(helper.make_node('MatMul', ['p', v], ['ctx_mm'], name='ctx'))
    qdq(nodes, 'ctx_mm', 'c', 'co')
    batch = [1] if exported else []
    inputs = [('x_q', TensorProto.INT8, [*batch, tokens, features])]
    inputs += [('u_q', u_type, [*batch, tokens, tokens])] if bias else []
    return make_model(
        'fused_attention_head',
        nodes=nodes,
        inputs=inputs,
        outputs=[('c', TensorProto.FLOAT, [*batch, tokens, head_dim])],
        initializers=inits,
        opset=21 if bias == 16 else 17,  # an int16 DequantizeLinear
    )


def _srs(x, shift, lo, hi):
    """Shift right rounding half to even, then saturate: the AIE's srs."""
    x = np.asarray(x, np.int64)
    q = x >> shift
    r = x - (q << shift)
    if shift:
        q += (r > 1 << (shift - 1)) | ((r == 1 << (shift - 1)) & (q & 1 == 1))
    return np.clip(q, lo, hi)


def _oracle(q, k, v, cfg, u=None):
    """The kernel's arithmetic: logits from the products and the bias at their fixed point (int8, or int16 with the
    gaps saturated to the table), unnormalised uint8 exp, one division per output."""
    bits = 16 if cfg.logits16 else 8
    products = q @ k.T + (0 if u is None else np.asarray(u, np.int64) << cfg.bias_shift)
    s = _srs(products, cfg.score_shift, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    p = np.asarray(cfg.exp_table)[np.minimum(s.max(-1, keepdims=True) - s, 255)]
    sums, theta = p.sum(-1), _srs(p @ v, THETA_SHIFT, -(2**15), 2**15 - 1)
    n, pow2 = sums.copy(), np.ones_like(sums)
    for shift in (4, 2, 1):
        small = n < 1 << (15 - shift)
        n, pow2 = np.where(small, n << shift, n), np.where(small, pow2 << shift, pow2)
    y = np.asarray(RECIP_TABLE)[_srs(n - 2**14, 6, 0, 255)]
    t = np.clip(theta * pow2[:, None], -(2**15), 2**15 - 1)
    return _srs(t * y[:, None], cfg.out_shift, -128, 127)


def _onnx(model, feeds, names):
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend(helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in names[1:])
    sess = onnx_session(probe)
    return dict(zip(names, sess.run(names, feeds)))


PAIRS, SINGLES = {'attn': parallelism(2, cas_length=2)}, {'attn': parallelism(4)}
# a directed Dense: Q, K and V handed over, no projection fusing
HANDED_OVER_SINGLES = {'q': parallelism(4, contract='outer'), **SINGLES}
HANDED_OVER_PAIRS = {'v': parallelism(1), **PAIRS}
# (part, tokens, head dimension, directives[, exported[, bias bits]])
CASES = {
    'aie-ml-exported-pair': (AIE_ML, 32, 32, PAIRS, True),
    'aie-mlv2-single': (AIE_MLV2, 32, 16, SINGLES),
    'aie-mlv2-pair': (AIE_MLV2, 32, 16, PAIRS),
    'aie-ml-padded-head-dim': (AIE_ML, 32, 8, HANDED_OVER_SINGLES),
    'aie-ml-bias8-pair': (AIE_ML, 32, 16, PAIRS, False, 8),
    'aie-mlv2-bias16-single': (AIE_MLV2, 32, 16, SINGLES, False, 16),
    'aie-ml-bias16-handed-over-single': (AIE_ML, 32, 16, HANDED_OVER_SINGLES, False, 16),
    # 1/sqrt(16): a power-of-two score scale, which the bias aligns to
    'aie-ml-bias8-exported-handed-over-pair': (AIE_ML, 32, 16, HANDED_OVER_PAIRS, True, 8),
}


def _model(case):
    """A case's head, its part and directives, and its bias's bits (0: none)."""
    part, tokens, head_dim, directives, *model = CASES[case]
    exported, bias = (model + [False, 0])[:2]
    return _head(tokens, head_dim, exported, bias=bias), part, directives, bias


@pytest.mark.parametrize('case', CASES)
def test_the_head_fuses_into_one_op(case, tmp_path):
    """Its projections too, where no Dense is directed, and its bias's Add."""
    model, part, directives, bias = _model(case)
    ctx = lower(model, tmp_path, directives, part=part)
    projections = [] if set(directives) == {'attn'} else ['dense', 'dense', 'dense']
    assert [n.op_type for n in ctx.ir.logical.nodes] == [*projections, 'attention']
    par = config_of(ctx, 'attn_aie').parallelism
    assert (par.cas_num, par.cas_length) == tuple(directives['attn']['parallelism'].values())
    inst = ctx.ir.execution.get('attn_aie')
    if not projections:  # x in two halves of its columns for the chains' first tiles, then for their last
        halves = inst.ports.inputs[input_tensor_for_role(inst.node, 'x').name].endpoints
        assert [len(tiles) for tiles in halves] == [par.cas_num] * (2 * par.cas_length)
    if bias:  # a chain's rows of U to its first tile
        assert inst.config.bias_bits == bias and '.bias.' in inst.variant.variant_id
        assert len(inst.ports.inputs[input_tensor_for_role(inst.node, 'bias').name].endpoints) == par.cas_num


def test_a_dense_padded_head_dimension_hands_over_directly(tmp_path):
    """Q split like the queries: the Dense's 4x8 microtiles and its head dimension padded to 16 columns are read in its
    memory, the zero columns as more of the head dimension."""
    ctx = lower(_head(32, 8), tmp_path, CASES['aie-ml-padded-head-dim'][3], part=AIE_ML)
    inst = ctx.ir.execution.get('attn_aie')
    q = input_tensor_for_role(inst.node, 'q').name
    assert inst.config.columns == 16 and inst.config.io_views[q].microtile.outer == 4
    assert ('q_aie', 'attn_aie') in direct_edges(ctx) and q not in memtiles(ctx)


def test_a_chain_is_one_tile_or_two(tmp_path):
    with pytest.raises(ConfigRefused, match='a chain is one tile or two'):
        lower(_head(32, 16), tmp_path, {'attn': parallelism(1, cas_length=3)}, part=AIE_ML)


def test_a_lone_tile_past_a_bank_is_refused_and_a_pair_fits(tmp_path):
    """128 keys and values of head dimension 64 handed over fill a 16 KB bank before a lone tile's output; a pair keeps
    the keys on one tile, the values and output on the other."""
    handed_over = {'v': parallelism(1)}  # a directed Dense: no projection fuses
    with pytest.raises(ConfigRefused, match='B of buffers'):
        lower(_head(128, 64), tmp_path / 'one', {**handed_over, 'attn': parallelism(8)}, part=AIE_ML)
    lower(_head(128, 64), tmp_path / 'two', {**handed_over, 'attn': parallelism(8, cas_length=2)}, part=AIE_ML)


def test_a_lone_tile_spilling_past_its_stack_leaves_the_projections_and_a_pair_takes_them(tmp_path):
    """Projected, 64 keys and values of head dimension 64 leave a lone tile 5 KB of stack beside their scratch, under
    the 8 KB it spills: Dense ops project them; a pair keeps the keys on one tile, the values on the other."""
    head = _head(64, 64, features=64)
    ctx = lower(head, tmp_path / 'one', {'attn': parallelism(8)}, part=AIE_ML)
    assert [n.op_type for n in ctx.ir.logical.nodes] == ['dense', 'dense', 'dense', 'attention']
    ctx = lower(head, tmp_path / 'two', {'attn': parallelism(8, cas_length=2)}, part=AIE_ML)
    assert [n.op_type for n in ctx.ir.logical.nodes] == ['attention']


@pytest.mark.parametrize(('part', 'tokens', 'head_dim'), [(AIE_ML, 128, 64), (AIE_ML, 128, 80), (AIE_MLV2, 64, 64)])
def test_a_head_spilling_past_a_pairs_stack_keeps_its_projections(tmp_path, part, tokens, head_dim):
    """Even a pair's first tile has too little stack to project them: the Dense ops stay, and the attention reads
    their Q, K and V."""
    ctx = lower(_head(tokens, head_dim, features=64), tmp_path, {}, part=part)
    assert [n.op_type for n in ctx.ir.logical.nodes] == ['dense', 'dense', 'dense', 'attention']


def test_x_reaches_eight_tiles_from_its_plios_directly(tmp_path):
    """x's halves, panels of 8 columns a DMA walks at once, each broadcast from a PLIO to 4 chains of 2: no memory
    tile, and no refusal (two PLIOs' broadcasts deliver exactly, unlike two kernels')."""
    ctx = lower(_head(32, 8, features=64), tmp_path, {'attn': parallelism(4, cas_length=2)}, part=AIE_ML)
    assert 'x_q' not in memtiles(ctx)


def test_k_and_v_broadcast_to_more_than_four_common_chains_are_refused(tmp_path):
    with pytest.raises(ConfigRefused, match='common tiles'):
        directives = {'q': parallelism(8, contract='outer'), 'k': parallelism(1), 'v': parallelism(1)}
        lower(_head(64, 16), tmp_path, {**directives, 'attn': parallelism(8)}, part=AIE_ML)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('case', CASES)
def test_x86_matches_the_kernels_arithmetic(case, tmp_path):
    model, part, directives, bias = _model(case)
    tokens, head_dim, exported = CASES[case][1], CASES[case][2], len(CASES[case]) > 4 and CASES[case][4]
    batch = (1,) if exported else ()  # an exported head's input is batch-first
    rng = np.random.default_rng(5)
    feeds = {'x_q': rng.integers(-40, 40, size=(*batch, tokens, FEATURES), dtype=np.int8)}
    if bias:  # within the logits' range at the bias's fixed point
        span = 2 ** (BIAS_FRAC[bias] + 1)
        u = rng.integers(-span, span, size=(*batch, tokens, tokens))
        feeds['u_q'] = u.astype(np.int16 if bias == 16 else np.int8)
    aie_model = from_onnx(
        model,
        {'Part': part, 'AIEConfig': {'Iterations': 1}, 'LayerDirectives': directives},
        output_dir=tmp_path / 'proj',
        project_name='proj',
    )
    aie_model.compile()
    got = np.asarray(aie_model.predict(feeds, simulator='x86', quantize_in=False, dequantize_out=False), np.int64)

    ref = _onnx(model, feeds, ['c', 'q', 'k', 'v'])
    q, k, v = (np.rint(ref[n] * 2**QKV_FRAC).astype(np.int64).reshape(tokens, head_dim) for n in 'qkv')
    u = feeds['u_q'].reshape(tokens, tokens) if bias else None
    want = _oracle(q, k, v, config_of(aie_model.context, 'attn_aie'), u)
    assert got.shape == ref['c'].shape
    np.testing.assert_array_equal(got.reshape(want.shape), want)
    graph = np.clip(np.rint(ref['c'] * 2**OUT_FRAC), -128, 127).reshape(want.shape)
    assert np.abs(want - graph).max() <= 2
