"""Axes of extent 1 dropped at import: the rewritten graph computes what the exported one does, in onnxruntime."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import TensorProto, helper, make_model, numpy_helper, onnx_session

from aie4ml.frontends.onnx.shapes import build_shape_table
from aie4ml.frontends.onnx.units import drop_unit_axes
from aie4ml.frontends.onnx.utils import initializer_map, input_maps

pytest.importorskip('onnxruntime')
T, D = 8, 16


def _const(inits, name, value):
    inits.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def _rewritten(model):
    initializers = initializer_map(model.graph, numpy_helper)

    def shapes(m):
        return build_shape_table(m, initializers, input_maps(m.graph, set(initializers))[0])

    rewritten = drop_unit_axes(model, initializers, shapes)
    del rewritten.graph.initializer[:]
    rewritten.graph.initializer.extend(numpy_helper.from_array(np.asarray(v), k) for k, v in initializers.items())
    return rewritten, shapes(rewritten)


def _io(model):
    """Each graph input's and output's declared shape."""
    return {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in (*model.graph.input, *model.graph.output)
    }


def _run(model, x):
    sess = onnx_session(model)
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return sess.run(None, {model.graph.input[0].name: x.reshape(shape)})[0]


def test_a_sequence_first_block_drops_its_unit_axes():
    """Batch-first ends around sequence-first ops: a projection, a broadcast constant, LayerNorm, Softmax and a
    slice and concat of the features, each on (T, D) once rewritten."""
    rng = np.random.default_rng(0)
    inits, nodes = [], []
    w, ln_w = _const(inits, 'w', rng.normal(size=(D, D)).astype(np.float32)), _const(inits, 'g', np.ones(D, np.float32))
    shift = _const(inits, 'shift', rng.normal(size=(1, 1, D)).astype(np.float32))
    half = [_const(inits, n, [v]) for n, v in (('s0', 0), ('e0', D // 2), ('s1', D // 2), ('e1', D), ('ax', 2))]
    nodes += [
        helper.make_node('Transpose', ['x'], ['xs'], perm=[1, 0, 2]),
        helper.make_node('MatMul', ['xs', w], ['h']),
        helper.make_node('Add', ['h', shift], ['hb']),
        helper.make_node('LayerNormalization', ['hb', ln_w], ['n'], axis=-1),
        helper.make_node('Softmax', ['n'], ['p'], axis=2),
        helper.make_node('Slice', ['p', half[0], half[1], half[4]], ['lo']),
        helper.make_node('Slice', ['p', half[2], half[3], half[4]], ['hi']),
        helper.make_node('Concat', ['hi', 'lo'], ['c'], axis=2),
        helper.make_node('Transpose', ['c'], ['y'], perm=[1, 0, 2]),
    ]
    model = make_model(
        'units',
        nodes=nodes,
        inputs=[('x', TensorProto.FLOAT, [1, T, D])],
        outputs=[('y', TensorProto.FLOAT, [1, T, D])],
        initializers=inits,
    )
    rewritten, shapes = _rewritten(model)
    x = rng.normal(size=(T, D)).astype(np.float32)
    np.testing.assert_array_equal(_run(rewritten, x).reshape(T, D), _run(model, x).reshape(T, D))
    assert _io(rewritten) == {'x': (T, D), 'y': (T, D)}
    assert all(1 not in shapes[name] for node in rewritten.graph.node for name in node.output)
    assert 'Transpose' not in {node.op_type for node in rewritten.graph.node}


def test_an_nhwc_input_a_conv_reads_keeps_its_rank():
    rng = np.random.default_rng(1)
    inits = []
    w = _const(inits, 'w', rng.normal(size=(4, 1, 3, 3)).astype(np.float32))
    nodes = [
        helper.make_node('Transpose', ['x'], ['xc'], perm=[0, 3, 1, 2]),
        helper.make_node('Conv', ['xc', w], ['y'], pads=[1, 1, 1, 1]),
    ]
    model = make_model(
        'nhwc',
        nodes=nodes,
        inputs=[('x', TensorProto.FLOAT, [1, 6, 6, 1])],
        outputs=[('y', TensorProto.FLOAT, [1, 4, 6, 6])],
        initializers=inits,
    )
    rewritten, _ = _rewritten(model)
    assert _io(rewritten) == _io(model) and [n.op_type for n in rewritten.graph.node] == ['Transpose', 'Conv']
