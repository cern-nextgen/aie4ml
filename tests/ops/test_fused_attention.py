"""One head's Q·Kᵀ -> Softmax -> ·V fused into the attention op.

The op is not bit-exact with the QDQ graph (it divides once where the graph rounds every probability), so its oracle
is its own integer arithmetic; against the graph it stays within a code or two.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
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
    parallelism,
    qdq,
    qparams,
)

from aie4ml import from_onnx
from aie4ml.errors import ConfigRefused
from aie4ml.ir.graph import input_tensor_for_role
from aie4ml.op_impls.families.attention.fused import RECIP_TABLE, THETA_SHIFT

AIE_ML = 'xilinx_vek280_base_202610_1'
AIE_MLV2 = 'xc2ve3858-ssva2112-2MP-e-S'
FEATURES, QKV_FRAC, OUT_FRAC = 16, 4, 5


def _dense(nodes, inits, x, name, head_dim, seed):
    w = f'{name}_w'
    dq(nodes, f'{w}_q', w, w)
    nodes.append(helper.make_node('MatMul', [x, w], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', name, f'{name}o')
    weights = np.random.default_rng(seed).integers(-3, 4, size=(FEATURES, head_dim), dtype=np.int8)
    inits += [*qparams(w), *qparams(f'{name}o', frac=QKV_FRAC), numpy_helper.from_array(weights, f'{w}_q')]


def _head(tokens: int, head_dim: int, exported: bool = False):
    """x -> Q, K, V Dense -> Q·Kᵀ (int8 logits, 2^-3) -> Softmax (uint8, 2^-8) -> ·V (int8, 2^-5).

    exported: as PyTorch exports it -- batch-first ends around sequence-first (T, 1, D) tensors, Q divided by
    sqrt(head_dim) and quantized again (2^-6), the scores not quantized."""
    nodes = []
    inits = [*qparams('x'), *qparams('so', frac=3), *qparams('po', frac=8, unsigned=True)]
    inits += qparams('co', frac=OUT_FRAC)
    dq(nodes, 'x_q', 'x', 'x')
    x = 'x'
    if exported:
        nodes.append(helper.make_node('Transpose', ['x'], ['x_seq'], perm=[1, 0, 2], name='to_seq'))
        x = 'x_seq'
    for seed, name in enumerate('qkv', 1):
        _dense(nodes, inits, x, name, head_dim, seed)
    lhs, k_t, v = 'q', 'k_t', 'v'
    if exported:
        inits += [*qparams('qs', frac=6), numpy_helper.from_array(np.float32(np.sqrt(head_dim)), 'root')]
        nodes.append(helper.make_node('Div', ['q', 'root'], ['q_div'], name='q_scale'))
        qdq(nodes, 'q_div', 'q_s', 'qs')
        nodes.append(helper.make_node('Transpose', ['q_s'], ['q_b'], perm=[1, 0, 2], name='q_b'))
        nodes.append(helper.make_node('Transpose', ['k'], ['k_t'], perm=[1, 2, 0], name='k_t'))
        nodes.append(helper.make_node('Transpose', ['v'], ['v_b'], perm=[1, 0, 2], name='v_b'))
        lhs, v = 'q_b', 'v_b'
        nodes.append(helper.make_node('MatMul', [lhs, k_t], ['s'], name='scores'))
    else:
        nodes.append(helper.make_node('Transpose', ['k'], ['k_t'], perm=[1, 0], name='k_t'))
        nodes.append(helper.make_node('MatMul', [lhs, k_t], ['scores_mm'], name='scores'))
        qdq(nodes, 'scores_mm', 's', 'so')
    nodes.append(helper.make_node('Softmax', ['s'], ['p_sm'], name='attn', axis=-1))
    qdq(nodes, 'p_sm', 'p', 'po')
    nodes.append(helper.make_node('MatMul', ['p', v], ['ctx_mm'], name='ctx'))
    qdq(nodes, 'ctx_mm', 'c', 'co')
    batch = [1] if exported else []
    return make_model(
        'fused_attention_head',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [*batch, tokens, FEATURES])],
        outputs=[('c', TensorProto.FLOAT, [*batch, tokens, head_dim])],
        initializers=inits,
    )


def _srs(x, shift, lo, hi):
    """Shift right rounding half to even, then saturate: the AIE's srs."""
    x = np.asarray(x, np.int64)
    q = x >> shift
    r = x - (q << shift)
    if shift:
        q += (r > 1 << (shift - 1)) | ((r == 1 << (shift - 1)) & (q & 1 == 1))
    return np.clip(q, lo, hi)


def _oracle(q, k, v, cfg):
    """The kernel's arithmetic (attention.cpp): int8 logits (int16, their gaps saturated to the table's end, where the
    op chose their format), unnormalised uint8 exp, one division per output."""
    bits = 16 if cfg.logits16 else 8
    s = _srs(q @ k.T, cfg.score_shift, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
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
    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    probe.graph.output.extend(helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in names[1:])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(probe.SerializeToString(), options, providers=['CPUExecutionProvider'])
    return dict(zip(names, sess.run(names, feeds)))


# (part, tokens, head dimension, directives[, exported])
CASES = {
    'aie-ml-exported-pair': (AIE_ML, 32, 32, {'attn': parallelism(2, cas_length=2)}, True),
    'aie-mlv2-single': (AIE_MLV2, 32, 16, {'attn': parallelism(4)}),
    'aie-ml-padded-head-dim': (AIE_ML, 32, 8, {'q': parallelism(4, contract='outer'), 'attn': parallelism(4)}),
}


@pytest.mark.parametrize('case', CASES)
def test_the_head_fuses_into_one_op(case, tmp_path):
    part, tokens, head_dim, directives, *exported = CASES[case]
    ctx = lower(_head(tokens, head_dim, *exported), tmp_path, directives, part=part)
    assert [n.op_type for n in ctx.ir.logical.nodes] == ['dense', 'dense', 'dense', 'attention']
    par = config_of(ctx, 'attn_aie').parallelism
    assert (par.cas_num, par.cas_length) == tuple(directives['attn']['parallelism'].values())


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
    """128 keys and values of head dimension 64 fill a 16 KB bank before a lone tile's output; a pair keeps the keys on
    one tile, the values and output on the other."""
    with pytest.raises(ConfigRefused, match='B of buffers'):
        lower(_head(128, 64), tmp_path / 'one', {'attn': parallelism(8)}, part=AIE_ML)
    lower(_head(128, 64), tmp_path / 'two', {'attn': parallelism(8, cas_length=2)}, part=AIE_ML)


def test_k_and_v_broadcast_to_more_than_four_common_chains_are_refused(tmp_path):
    with pytest.raises(ConfigRefused, match='common tiles'):
        directives = {'q': parallelism(8, contract='outer'), 'k': parallelism(1), 'v': parallelism(1)}
        lower(_head(64, 16), tmp_path, {**directives, 'attn': parallelism(8)}, part=AIE_ML)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('case', CASES)
def test_x86_matches_the_kernels_arithmetic(case, tmp_path):
    part, tokens, head_dim, directives, *exported = CASES[case]
    model = _head(tokens, head_dim, *exported)
    shape = (1,) * len(exported) + (tokens, FEATURES)  # an exported head's input is batch-first
    feeds = {'x_q': np.random.default_rng(5).integers(-40, 40, size=shape, dtype=np.int8)}
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
    want = _oracle(q, k, v, config_of(aie_model.context, 'attn_aie'))
    assert got.shape == ref['c'].shape
    np.testing.assert_array_equal(got.reshape(want.shape), want)
    graph = np.clip(np.rint(ref['c'] * 2**OUT_FRAC), -128, 127).reshape(want.shape)
    assert np.abs(want - graph).max() <= 2
