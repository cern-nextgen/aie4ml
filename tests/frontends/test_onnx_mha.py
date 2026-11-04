"""Multi-head attention as PyTorch exports it: sequence-first tensors, one packed projection, every head in one op."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import PART, TensorProto, dq, helper, lower, make_model, numpy_helper, qdq, qparams

from aie4ml.frontends.onnx.constants import fold_constants
from aie4ml.frontends.onnx.heads import unbatch_heads
from aie4ml.frontends.onnx.importer import _named
from aie4ml.frontends.onnx.projections import split_packed_projections
from aie4ml.frontends.onnx.shapes import build_shape_table
from aie4ml.frontends.onnx.units import drop_unit_axes
from aie4ml.frontends.onnx.utils import initializer_map, input_maps
from aie4ml.ir.graph import has_input_role

ort = pytest.importorskip('onnxruntime')
T, D, H = 16, 32, 2
HD = D // H


def _mha():
    """x (1, T, D) -> (T, 1, D) -> x W_in + b -> Q, K, V slices -> heads (T, H, HD) -> (H, T, HD) -> Q / sqrt(HD),
    softmax(Q Kᵀ) V -> (T, H, HD) -> (T, 1, D) -> W_out -> (1, T, D)."""
    rng = np.random.default_rng(0)
    nodes, inits = [], [*qparams('x'), *qparams('wi', frac=6), *qparams('wo', frac=6)]
    inits += [p for prefix in ('hq', 'hk', 'hv', 'hc') for p in qparams(prefix)]
    inits += [*qparams('qs', frac=5), *qparams('p', frac=8, unsigned=True), *qparams('o', frac=3)]
    inits.append(helper.make_tensor('b_scale32', TensorProto.FLOAT, [], [2.0**-10]))
    inits.append(helper.make_tensor('b_zp32', TensorProto.INT32, [], [0]))

    def const(name, value):
        inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    dq(nodes, 'x_q', 'x', 'x')
    nodes.append(helper.make_node('Transpose', ['x'], ['xs'], perm=[1, 0, 2]))
    const('w_in_q', rng.integers(-40, 40, size=(D, 3 * D), dtype=np.int8))
    const('w_out_q', rng.integers(-40, 40, size=(D, D), dtype=np.int8))
    const('b_in_q', rng.integers(-500, 500, size=3 * D, dtype=np.int32))
    dq(nodes, 'w_in_q', 'w_in', 'wi')
    dq(nodes, 'w_out_q', 'w_out', 'wo')
    nodes.append(helper.make_node('DequantizeLinear', ['b_in_q', 'b_scale32', 'b_zp32'], ['b_in']))
    nodes += [helper.make_node('MatMul', ['xs', 'w_in'], ['mm']), helper.make_node('Add', ['mm', 'b_in'], ['qkv'])]
    split = const('heads', [T, H, HD])
    for i, role in enumerate('qkv'):
        bounds = [const(f'{role}_{n}', [v]) for n, v in (('start', i * D), ('end', (i + 1) * D), ('axis', -1))]
        nodes.append(helper.make_node('Slice', ['qkv', *bounds], [f'{role}_raw']))
        qdq(nodes, f'{role}_raw', role, f'h{role}')
    nodes.append(helper.make_node('Div', ['q', const('root', np.float32(np.sqrt(HD)))], ['q_div']))
    qdq(nodes, 'q_div', 'q_s', 'qs')
    for role, src, perm in (('q', 'q_s', [1, 0, 2]), ('k', 'k', [1, 2, 0]), ('v', 'v', [1, 0, 2])):
        nodes.append(helper.make_node('Reshape', [src, split], [f'{role}_h']))
        nodes.append(helper.make_node('Transpose', [f'{role}_h'], [f'{role}_t'], perm=perm))
    nodes += [helper.make_node('MatMul', ['q_t', 'k_t'], ['s']), helper.make_node('Softmax', ['s'], ['p_sm'], axis=-1)]
    qdq(nodes, 'p_sm', 'p', 'p')
    nodes.append(helper.make_node('MatMul', ['p', 'v_t'], ['c_mm']))
    qdq(nodes, 'c_mm', 'c', 'hc')
    nodes.append(helper.make_node('Transpose', ['c'], ['c_t'], perm=[1, 0, 2]))
    nodes.append(helper.make_node('Reshape', ['c_t', const('merged', [T, 1, D])], ['c_m']))
    nodes.append(helper.make_node('MatMul', ['c_m', 'w_out'], ['y_mm']))
    qdq(nodes, 'y_mm', 'y_s', 'o')
    nodes.append(helper.make_node('Transpose', ['y_s'], ['y'], perm=[1, 0, 2]))
    return make_model(
        'mha', nodes=nodes, inputs=[('x_q', TensorProto.INT8, [1, T, D])],
        outputs=[('y', TensorProto.FLOAT, [1, T, D])], initializers=inits,
    )  # fmt: skip


def _run(model, x):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return sess.run(None, {'x_q': x.reshape(shape)})[0]


def test_the_import_rewrite_computes_what_the_export_does():
    model = _mha()
    initializers = initializer_map(model.graph, numpy_helper)

    def shapes(m):
        return build_shape_table(m, initializers, input_maps(m.graph, set(initializers))[0])

    rewritten = fold_constants(_named(model), initializers, shapes)
    rewritten, _ = drop_unit_axes(rewritten, initializers, shapes(rewritten))
    for rewrite in (split_packed_projections, unbatch_heads, split_packed_projections):
        rewritten = rewrite(rewritten, initializers, shapes(rewritten))
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(numpy_helper.from_array(np.asarray(v), k) for k, v in initializers.items())
    x = np.random.default_rng(1).integers(-60, 60, size=(T, D), dtype=np.int8)
    np.testing.assert_array_equal(_run(rewritten, x).reshape(T, D), _run(model, x).reshape(T, D))
    assert sorted(n.op_type for n in rewritten.graph.node).count('MatMul') == 3 * H + 2 * H + 1


def test_each_head_lowers_to_one_attention_with_its_projections(tmp_path):
    ctx = lower(_mha(), tmp_path, part=PART)
    ops = [n.op_type for n in ctx.ir.logical if not n.is_folded_view]
    assert ops == ['attention'] * H + ['dense']
    assert all(has_input_role(n, 'x') for n in ctx.ir.logical if n.op_type == 'attention')
