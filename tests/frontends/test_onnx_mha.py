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


def _mha(bias: str = ''):
    """PyTorch's MHA graph, x (1, T, D) -> W_in -> Q, K, V heads -> softmax(Q Kᵀ / sqrt(HD) [+ u]) V -> W_out; u (input
    u_q) a mask-like bias, 'shared' (T, T) or 'heads' (H, T, T)."""
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
    const('w_in_q', rng.integers(-8, 8, size=(D, 3 * D), dtype=np.int8))  # mild: no near-ties in the attention
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
    nodes.append(helper.make_node('MatMul', ['q_t', 'k_t'], ['s']))
    if bias:
        inits += qparams('u', frac=2)
        dq(nodes, 'u_q', 'u', 'u')
        nodes.append(helper.make_node('Add', ['s', 'u'], ['s_b']))
    nodes.append(helper.make_node('Softmax', ['s_b' if bias else 's'], ['p_sm'], axis=-1))
    qdq(nodes, 'p_sm', 'p', 'p')
    nodes.append(helper.make_node('MatMul', ['p', 'v_t'], ['c_mm']))
    qdq(nodes, 'c_mm', 'c', 'hc')
    nodes.append(helper.make_node('Transpose', ['c'], ['c_t'], perm=[1, 0, 2]))
    nodes.append(helper.make_node('Reshape', ['c_t', const('merged', [T, 1, D])], ['c_m']))
    nodes.append(helper.make_node('MatMul', ['c_m', 'w_out'], ['y_mm']))
    qdq(nodes, 'y_mm', 'y_s', 'o')
    nodes.append(helper.make_node('Transpose', ['y_s'], ['y'], perm=[1, 0, 2]))
    u_shape = {'shared': [T, T], 'heads': [H, T, T]}.get(bias)
    inputs = [('x_q', TensorProto.INT8, [1, T, D])] + ([('u_q', TensorProto.INT8, u_shape)] if bias else [])
    return make_model(
        'mha', nodes=nodes, inputs=inputs, outputs=[('y', TensorProto.FLOAT, [1, T, D])], initializers=inits,
    )  # fmt: skip


def _run(model, x, u=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return sess.run(None, {'x_q': x.reshape(shape), **({} if u is None else {'u_q': u})})[0]


def _rewritten(model):
    """The model as the importer rewrites it, head by head, before lowering."""
    initializers = initializer_map(model.graph, numpy_helper)

    def shapes(m):
        return build_shape_table(m, initializers, input_maps(m.graph, set(initializers))[0])

    rewritten = fold_constants(_named(model), initializers, shapes)
    rewritten, _ = drop_unit_axes(rewritten, initializers, shapes(rewritten))
    rewritten = split_packed_projections(rewritten, initializers, shapes(rewritten))
    rewritten = unbatch_heads(rewritten, initializers, shapes)
    rewritten = split_packed_projections(rewritten, initializers, shapes(rewritten))
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(numpy_helper.from_array(np.asarray(v), k) for k, v in initializers.items())
    return rewritten


@pytest.mark.parametrize('bias', ['', 'shared', 'heads'])
def test_the_import_rewrite_computes_what_the_export_does(bias):
    model = _mha(bias)
    rewritten = _rewritten(model)
    rng = np.random.default_rng(1)
    x = rng.integers(-60, 60, size=(T, D), dtype=np.int8)
    u = rng.integers(-20, 20, size=(H, T, T) if bias == 'heads' else (T, T), dtype=np.int8) if bias else None
    np.testing.assert_array_equal(_run(rewritten, x, u).reshape(T, D), _run(model, x, u).reshape(T, D))
    assert sorted(n.op_type for n in rewritten.graph.node).count('MatMul') == 3 * H + 2 * H + 1


@pytest.mark.parametrize('bias', ['', 'shared', 'heads'])
def test_each_head_lowers_to_one_attention_with_its_projections(bias, tmp_path):
    """And its bias: the one every head adds, or its own, a slice of the input the host reads in rows."""
    ctx = lower(_mha(bias), tmp_path, part=PART)
    ops = [n.op_type for n in ctx.ir.logical if not n.is_folded_view]
    assert ops == ['attention'] * H + ['dense']
    heads = [n for n in ctx.ir.logical if n.op_type == 'attention']
    assert all(has_input_role(n, 'x') and has_input_role(n, 'bias') == bool(bias) for n in heads)


def _two_layers():
    """Two MHA graphs in a row adding the same U, as an encoder's layers do: the second's names prefixed, reading
    the first's output and its U."""
    first, second = _mha('heads').graph, _mha('heads').graph
    shared = {'u_q', 'u', 'u_scale', 'u_zp'}
    rename = {'x': 'y', **{n: n for n in shared}}  # the second reads the first's output, and its U
    name = lambda n: rename.get(n, f'l1_{n}') if n else n  # noqa: E731
    nodes = list(first.node)
    for node in second.node:
        if node.op_type == 'DequantizeLinear' and node.input[0] in ('x_q', 'u_q'):
            continue  # the first's
        node.input[:] = [name(n) for n in node.input]
        node.output[:] = [name(n) for n in node.output]
        node.name = f'l1_{node.name}' if node.name else ''
        nodes.append(node)
    inits = list(first.initializer) + [i for i in second.initializer if i.name not in shared]
    for init in inits[len(first.initializer) :]:
        init.name = name(init.name)
    inputs = [('x_q', TensorProto.INT8, [1, T, D]), ('u_q', TensorProto.INT8, [H, T, T])]
    return make_model(
        'mha2', nodes=nodes, inputs=inputs, outputs=[('l1_y', TensorProto.FLOAT, [1, T, D])], initializers=inits,
    )  # fmt: skip


def test_layers_adding_one_bias_each_slice_it(tmp_path):
    """Every layer's heads read their rows of the one U, as the export adds it."""
    model = _two_layers()
    rng = np.random.default_rng(1)
    x, u = rng.integers(-60, 60, size=(T, D), dtype=np.int8), rng.integers(-20, 20, size=(H, T, T), dtype=np.int8)
    np.testing.assert_array_equal(_run(_rewritten(model), x, u).reshape(T, D), _run(model, x, u).reshape(T, D))
    ctx = lower(model, tmp_path, part=PART)
    ops = [n.op_type for n in ctx.ir.logical if not n.is_folded_view]
    assert ops == (['attention'] * H + ['dense']) * 2
    assert all(has_input_role(n, 'bias') for n in ctx.ir.logical if n.op_type == 'attention')


@pytest.mark.requires_vitis
def test_x86_adds_each_heads_bias(tmp_path):
    """Each head's rows of U reach it: x86 within a code or two of the export, which swapping the heads' rows moves
    much further."""
    from aie4ml import from_onnx

    model = _mha('heads')
    rng = np.random.default_rng(2)
    x = rng.integers(-60, 60, size=(T, D), dtype=np.int8)
    u = rng.integers(-40, 40, size=(H, T, T), dtype=np.int8)
    aie_model = from_onnx(model, {'Part': PART, 'AIEConfig': {'Iterations': 1}}, output_dir=tmp_path / 'proj')
    aie_model.compile()
    feeds = {'x_q': x.reshape(1, T, D), 'u_q': u}
    got = np.asarray(aie_model.predict(feeds, simulator='x86', quantize_in=False, dequantize_out=False), np.int64)
    want, swapped = (np.rint(_run(model, x, v) * 2**3).astype(np.int64).reshape(T, D) for v in (u, u[::-1]))
    assert np.abs(got.reshape(T, D) - want).max() <= 2 < np.abs(swapped - want).max()
