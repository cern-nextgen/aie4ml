"""One attention head: Q/K/V projections, Q·Kᵀ, ·V and the output projection.

The matmul reads K transposed, and V, as their projections store them, so both hand over with no memory tile.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    TensorProto,
    assert_x86_matches_onnx,
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

AIE1 = 'xcvp2802-vsva5601-2MHP-e-S'
AIE_ML = 'xilinx_vek280_base_202610_1'
AIE_MLV2 = 'xc2ve3858-ssva2112-2MP-e-S'
WIDTH, FRAC_OUT = 32, 5


def _dense(nodes, inits, src, name, k, n, seed, frac):
    w = f'{name}_w'
    dq(nodes, f'{w}_q', w, w)
    nodes.append(helper.make_node('MatMul', [src, w], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', name, f'{name}o')
    weights = np.random.default_rng(seed).integers(-3, 4, size=(k, n), dtype=np.int8)
    inits += [*qparams(w), *qparams(f'{name}o', frac=frac), numpy_helper.from_array(weights, f'{w}_q')]
    return name


def _head(tokens: int, features: int, head_dim: int):
    nodes, inits = [], [*qparams('x'), *qparams('scoreso', frac=5), *qparams('ctxo', frac=5)]
    dq(nodes, 'x_q', 'x', 'x')
    embedded = _dense(nodes, inits, 'x', 'embed', features, WIDTH, 0, 5)
    q, k, v = (_dense(nodes, inits, embedded, name, WIDTH, head_dim, seed, 6) for seed, name in enumerate('qkv', 1))
    nodes.append(helper.make_node('Transpose', [k], ['k_t'], perm=[1, 0], name='k_t'))
    nodes.append(helper.make_node('MatMul', [q, 'k_t'], ['scores_mm'], name='scores'))
    qdq(nodes, 'scores_mm', 's', 'scoreso')
    nodes.append(helper.make_node('MatMul', ['s', v], ['ctx_mm'], name='ctx'))
    qdq(nodes, 'ctx_mm', 'c', 'ctxo')
    out = _dense(nodes, inits, 'c', 'o', head_dim, features, 4, FRAC_OUT)
    return make_model(
        'attention_head',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [tokens, features])],
        outputs=[(out, TensorProto.FLOAT, [tokens, features])],
        initializers=inits,
    )


# Q and K split by head dimension feed a two-stage cascade; V, split likewise, feeds two chains.
SPLIT = {
    'q': parallelism(2),
    'k': parallelism(2),
    'scores': parallelism(1, cas_length=2),
    'v': parallelism(2),
    'ctx': parallelism(2),
}
# (part, tokens, features, head dimension, directives). AIE1 takes 8 features: its PLIO re-tiles a wider input
# with more BDs than a tile has.
CASES = {
    'aie1': (AIE1, 16, 8, 16, {}),
    'aie-ml': (AIE_ML, 16, 16, 16, {}),
    'aie-mlv2': (AIE_MLV2, 16, 16, 16, {}),
    'aie1-split': (AIE1, 16, 8, 32, SPLIT),
}


@pytest.mark.parametrize('case', CASES)
def test_attention_hands_k_and_v_over_directly(case, tmp_path):
    part, tokens, features, head_dim, directives = CASES[case]
    ctx = lower(_head(tokens, features, head_dim), tmp_path, directives, part=part)
    assert not memtiles(ctx)
    assert {('k_aie', 'scores_aie'), ('v_aie', 'ctx_aie')} <= direct_edges(ctx)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('case', CASES)
def test_attention_matches_onnx(case, tmp_path):
    part, tokens, features, head_dim, directives = CASES[case]
    feeds = {'x_q': np.random.default_rng(5).integers(-40, 40, size=(tokens, features), dtype=np.int8)}
    model = _head(tokens, features, head_dim)
    assert_x86_matches_onnx(model, feeds, directives, tmp_path, frac=FRAC_OUT, max_code_diff=0, part=part)


def test_a_long_sequence_crosses_the_boundary_through_memory_tiles(tmp_path):
    """128 tokens re-tiled at the PLIO would need more BDs than a tile has; the core still hands over directly."""
    ctx = lower(_head(128, 16, 16), tmp_path, part=AIE_ML)
    assert memtiles(ctx) == {'x_q', 'o'}
    assert {('k_aie', 'scores_aie'), ('v_aie', 'ctx_aie')} <= direct_edges(ctx)
