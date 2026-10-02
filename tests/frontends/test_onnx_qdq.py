"""QDQ patterns a PyTorch-style export puts a quantizer behind: a Dense's bias and ReLU, and a scale."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    PART,
    TensorProto,
    assert_x86_matches_onnx,
    dq,
    helper,
    lower,
    make_model,
    numpy_helper,
    qdq,
    qparams,
)

ROWS, FEATURES = 16, 32


def _model(softmax=False):
    """x -> MatMul -> Add(bias) -> Relu -> Q (uint8), then -> MatMul -> Mul(1/8) -> Q, and optionally -> Softmax: each
    quantizer sits after the ops it measures, so the frontend types the accumulators behind them."""
    rng = np.random.default_rng(3)
    nodes: list = []
    dq(nodes, 'x_q', 'x', 'x')
    dq(nodes, 'w1_q', 'w1', 'w1')
    dq(nodes, 'b1_q', 'b1', 'b1')
    nodes += [
        helper.make_node('MatMul', ['x', 'w1'], ['mm1'], name='fc1'),
        helper.make_node('Add', ['mm1', 'b1'], ['biased'], name='fc1_bias'),
        helper.make_node('Relu', ['biased'], ['relu'], name='fc1_relu'),
    ]
    qdq(nodes, 'relu', 'h', 'h')
    dq(nodes, 'w2_q', 'w2', 'w2')
    nodes += [
        helper.make_node('MatMul', ['h', 'w2'], ['mm2'], name='fc2'),
        helper.make_node('Mul', ['mm2', 'eighth'], ['scaled'], name='fc2_scale'),
    ]
    qdq(nodes, 'scaled', 'y', 'y')
    out = 'y'
    if softmax:
        nodes.append(helper.make_node('Softmax', ['y'], ['sm'], name='softmax', axis=-1))
        qdq(nodes, 'sm', 'p', 'p')
        out = 'p'
    return make_model(
        'qdq_patterns',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [ROWS, FEATURES])],
        outputs=[(out, TensorProto.FLOAT, [ROWS, FEATURES])],
        initializers=[
            *qparams('p', frac=8, unsigned=True),
            *qparams('x', frac=5),
            *qparams('w1', frac=7),
            *qparams('b1', frac=5),
            *qparams('h', frac=8, unsigned=True),
            *qparams('w2', frac=7),
            *qparams('y', frac=3),
            numpy_helper.from_array(rng.integers(-64, 64, (FEATURES, FEATURES), dtype=np.int8), 'w1_q'),
            numpy_helper.from_array(rng.integers(-32, 32, FEATURES, dtype=np.int8), 'b1_q'),
            numpy_helper.from_array(rng.integers(-64, 64, (FEATURES, FEATURES), dtype=np.int8), 'w2_q'),
            numpy_helper.from_array(np.float32(0.125), 'eighth'),
        ],
    )


def test_quantizers_behind_a_bias_relu_and_scale_fold_into_the_dense_kernels(tmp_path):
    """The scale before y's quantizer is fc2's requantization, even with a Softmax after y that could scale its
    input: y's codes are the scaled product's, not fc2's."""
    ctx = lower(_model(softmax=True), tmp_path, part=PART)
    assert [node.op_type for node in ctx.ir.logical if not node.is_folded_view] == ['dense', 'dense', 'softmax']
    fc1, fc2, softmax = (ctx.ir.execution.get(name) for name in ('fc1_aie', 'fc2_aie', 'softmax_aie'))
    assert fc1.config.flags.use_bias and fc1.config.flags.use_relu
    assert 'output_scale' in fc2.node.traits and 'input_scale' not in softmax.node.traits


@pytest.mark.requires_vitis
def test_quantizers_behind_a_bias_relu_and_scale_match_onnx(tmp_path):
    feeds = {'x_q': np.random.default_rng(5).integers(-64, 64, size=(ROWS, FEATURES), dtype=np.int8)}
    assert_x86_matches_onnx(_model(), feeds, {}, tmp_path, frac=3, max_code_diff=0, part=PART)
