"""One attention head: Q/K/V projections, Q·Kᵀ, ·V and the output projection, with no memory tile.

The matmul reads K transposed, and V, as their projections store them, so both hand over directly on every
generation -- AIE1 too, which has no memory tile to re-lay them out.
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
WIDTH = 32
FRAC_OUT = 0  # the attention sums grow past int8 at the input's scale


def _dense(nodes, inits, src, name, k, n, seed, frac=4):
    w = f'{name}_w'
    dq(nodes, f'{w}_q', w, w)
    nodes.append(helper.make_node('MatMul', [src, w], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', name, f'{name}o')
    weights = np.random.default_rng(seed).integers(-3, 4, size=(k, n), dtype=np.int8)
    inits += [*qparams(w), *qparams(f'{name}o', frac=frac), numpy_helper.from_array(weights, f'{w}_q')]
    return name


def _head(tokens: int, head_dim: int):
    nodes, inits = [], [*qparams('x')]
    dq(nodes, 'x_q', 'x', 'x')
    q, k, v = (_dense(nodes, inits, 'x', name, WIDTH, head_dim, seed) for seed, name in enumerate('qkv', 1))
    nodes.append(helper.make_node('Transpose', [k], ['k_t'], perm=[1, 0], name='k_t'))
    nodes.append(helper.make_node('MatMul', [q, 'k_t'], ['scores_mm'], name='scores'))
    qdq(nodes, 'scores_mm', 's', 'scoreso')
    nodes.append(helper.make_node('MatMul', ['s', v], ['ctx_mm'], name='ctx'))
    qdq(nodes, 'ctx_mm', 'c', 'ctxo')
    inits += [*qparams('scoreso', frac=2), *qparams('ctxo', frac=FRAC_OUT)]
    out = _dense(nodes, inits, 'c', 'o', head_dim, WIDTH, 4, frac=FRAC_OUT)
    return make_model(
        'attention_head',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [tokens, WIDTH])],
        outputs=[(out, TensorProto.FLOAT, [tokens, WIDTH])],
        initializers=inits,
    )


# Each part's mmul stores K and V in blocks of 2 (AIE1), 4 (AIE-ML) or 8 (AIE-MLv2) rows: the matmul stacks them.
DEFAULT = {}
# Q and K split by head dimension feed a two-stage cascade; V, split likewise, feeds two chains.
SPLIT = {
    'q': parallelism(2),
    'k': parallelism(2),
    'scores': parallelism(1, cas_length=2),
    'v': parallelism(2),
    'ctx': parallelism(2),
}
CASES = {
    'aie1': (AIE1, 16, 16, DEFAULT),
    'aie-ml': (AIE_ML, 16, 16, DEFAULT),
    'aie-mlv2': (AIE_MLV2, 16, 16, DEFAULT),
    'aie1-split': (AIE1, 32, 32, SPLIT),
}


@pytest.mark.parametrize('case', CASES)
def test_attention_hands_k_and_v_over_directly(case, tmp_path):
    part, tokens, head_dim, directives = CASES[case]
    ctx = lower(_head(tokens, head_dim), tmp_path, directives, part=part)
    assert not memtiles(ctx)
    assert {('k_aie', 'scores_aie'), ('v_aie', 'ctx_aie')} <= direct_edges(ctx)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('case', CASES)
def test_attention_matches_onnx(case, tmp_path):
    part, tokens, head_dim, directives = CASES[case]
    feeds = {'x_q': np.random.default_rng(5).integers(-40, 40, size=(tokens, WIDTH), dtype=np.int8)}
    assert_x86_matches_onnx(
        _head(tokens, head_dim), feeds, directives, tmp_path, frac=FRAC_OUT, max_code_diff=0, part=part
    )
