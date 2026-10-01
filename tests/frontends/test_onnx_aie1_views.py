from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from helpers import PART, TensorProto, dq, helper, lower, make_model, numpy_helper, qdq, qparams

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'


def _input(nodes, initializers, name: str, rows: int):
    dq(nodes, f'{name}_q', name, name)
    initializers.extend(qparams(name))
    return (f'{name}_q', TensorProto.INT8, [rows, 16])


def _dense(nodes, initializers, source: str, name: str, *, seed=None):
    """16 -> 16 MatMul; all-ones weights unless a seed asks for distinct ones, which a numeric test
    needs to tell rows and features apart."""
    weight_name = f'{name}_w'
    dq(nodes, f'{weight_name}_q', weight_name, weight_name)
    nodes.append(helper.make_node('MatMul', [source, weight_name], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', f'{name}_out', name)
    weights = (
        np.ones((16, 16), dtype=np.int8)
        if seed is None
        else np.random.default_rng(seed).integers(-3, 4, size=(16, 16), dtype=np.int8)
    )
    initializers.extend([*qparams(weight_name), *qparams(name), numpy_helper.from_array(weights, f'{weight_name}_q')])
    return f'{name}_out'


def _parallelism(cas_num: int):
    return {'parallelism': {'cas_num': cas_num, 'cas_length': 1, 'contract': 'outer'}}


def _split_model(*, graph_output: bool = False):
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root')
    split_sizes = numpy_helper.from_array(np.asarray([4, 4], dtype=np.int64), 'split_sizes')
    initializers.append(split_sizes)
    nodes.append(helper.make_node('Split', [root, 'split_sizes'], ['upper', 'lower'], name='split', axis=0))

    if graph_output:
        outputs = [('upper', TensorProto.FLOAT, [4, 16])]
    else:
        left = _dense(nodes, initializers, 'upper', 'left')
        right = _dense(nodes, initializers, 'lower', 'right')
        outputs = [(left, TensorProto.FLOAT, [4, 16]), (right, TensorProto.FLOAT, [4, 16])]

    return make_model(
        'aie1_split_views',
        nodes=nodes,
        inputs=[graph_input],
        outputs=outputs,
        initializers=initializers,
    )


def _chained_view_model():
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root')
    initializers.append(numpy_helper.from_array(np.asarray([4, 4], dtype=np.int64), 'split_sizes'))
    nodes.append(helper.make_node('Split', [root, 'split_sizes'], ['upper', 'lower'], name='split', axis=0))
    nodes.append(helper.make_node('Concat', ['upper', 'lower'], ['joined'], name='concat', axis=0))
    output = _dense(nodes, initializers, 'joined', 'tail')
    return make_model(
        'aie1_chained_views',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [8, 16])],
        initializers=initializers,
    )


def _slice_model(start: int, end: int, *, source_is_graph_input: bool = False):
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    source = 'x' if source_is_graph_input else _dense(nodes, initializers, 'x', 'root')
    for name, value in (('starts', [start]), ('ends', [end]), ('axes', [0])):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
    nodes.append(helper.make_node('Slice', [source, 'starts', 'ends', 'axes'], ['slice_out'], name='slice'))
    output = _dense(nodes, initializers, 'slice_out', 'tail')
    return make_model(
        'aie1_slice_view',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [end - start, 16])],
        initializers=initializers,
    )


def _concat_model(first_rows: int, second_rows: int, *, graph_output: bool = False):
    nodes, initializers = [], []
    first_input = _input(nodes, initializers, 'a', first_rows)
    second_input = _input(nodes, initializers, 'b', second_rows)
    first = _dense(nodes, initializers, 'a', 'first')
    second = _dense(nodes, initializers, 'b', 'second')
    nodes.append(helper.make_node('Concat', [first, second], ['joined'], name='concat', axis=0))

    if graph_output:
        outputs = [('joined', TensorProto.FLOAT, [first_rows + second_rows, 16])]
    else:
        output = _dense(nodes, initializers, 'joined', 'tail')
        outputs = [(output, TensorProto.FLOAT, [first_rows + second_rows, 16])]

    return make_model(
        'aie1_concat_view',
        nodes=nodes,
        inputs=[first_input, second_input],
        outputs=outputs,
        initializers=initializers,
    )


def test_port_aligned_split_maps_disjoint_producer_ports_directly(tmp_path):
    ctx = lower(
        _split_model(),
        tmp_path,
        part=AIE1_PART,
        directives={'root': _parallelism(2), 'left': _parallelism(1), 'right': _parallelism(1)},
    )
    edges = {(edge['source'], edge['target']) for edge in ctx.ir.physical.plan['direct_edges']}

    assert ('root_aie.out1[0]', 'left_aie.in1[0]') in edges
    assert ('root_aie.out1[1]', 'right_aie.in1[0]') in edges
    assert ctx.ir.physical.plan['buffers'] == []


def test_slice_crossing_producer_port_requires_relay(tmp_path):
    with pytest.raises(
        ConfigRefused,
        match=r'slice range \[2, 6\) crosses producer port 0 range \[0, 4\); packed slice/relay',
    ):
        lower(
            _slice_model(2, 6),
            tmp_path,
            part=AIE1_PART,
            directives={'root': _parallelism(2), 'tail': _parallelism(1)},
        )


def test_port_aligned_concat_maps_each_source_to_one_consumer_port(tmp_path):
    ctx = lower(
        _concat_model(4, 4),
        tmp_path,
        part=AIE1_PART,
        directives={'first': _parallelism(1), 'second': _parallelism(1), 'tail': _parallelism(2)},
    )
    edges = {(edge['source'], edge['target']) for edge in ctx.ir.physical.plan['direct_edges']}

    assert ('first_aie.out1[0]', 'tail_aie.in1[0]') in edges
    assert ('second_aie.out1[0]', 'tail_aie.in1[1]') in edges
    assert ctx.ir.physical.plan['buffers'] == []


def test_concat_port_spanning_two_sources_requires_relay(tmp_path):
    with pytest.raises(
        ConfigRefused,
        match=r'concat consumer port 0.*range \[0, 4\).*packed concat/relay',
    ):
        lower(
            _concat_model(3, 5),
            tmp_path,
            part=AIE1_PART,
            directives={'first': _parallelism(1), 'second': _parallelism(1), 'tail': _parallelism(2)},
        )


@pytest.mark.parametrize(
    ('model', 'message'),
    [
        (_split_model(graph_output=True), r'upper: split-backed graph outputs are not implemented'),
        (_concat_model(4, 4, graph_output=True), r'joined: concat-backed graph outputs are not implemented'),
    ],
    ids=['split', 'concat'],
)
def test_view_backed_graph_outputs_fail_explicitly(model, message, tmp_path):
    with pytest.raises(NotImplementedError, match=message):
        lower(model, tmp_path, part=AIE1_PART)


def test_view_backed_graph_input_fails_explicitly(tmp_path):
    with pytest.raises(NotImplementedError, match=r'slice input .* is a graph input'):
        lower(
            _slice_model(0, 4, source_is_graph_input=True),
            tmp_path,
            part=AIE1_PART,
            directives={'tail': _parallelism(1)},
        )


def test_chained_views_fail_explicitly(tmp_path):
    with pytest.raises(NotImplementedError, match=r'chained view transport through split'):
        lower(
            _chained_view_model(),
            tmp_path,
            part=AIE1_PART,
            directives={'root': _parallelism(2), 'tail': _parallelism(2)},
        )


# --------------------------------------------------------------------------- #
# a view op on a transposed value: its axis is the canonical tensor's
# --------------------------------------------------------------------------- #


def _transposed(nodes, source: str, name: str) -> str:
    nodes.append(helper.make_node('Transpose', [source], [name], perm=[1, 0], name=name))
    return name


def _transposed_slice_model(*, seed=None):
    """root [8, 16] viewed as [16, 8]; Slice(axis=1, 0:4) takes its first four ROWS, not features."""
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root', seed=seed)
    for name, value in (('starts', [0]), ('ends', [4]), ('axes', [1])):
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
    nodes.append(
        helper.make_node('Slice', [_transposed(nodes, root, 'root_t'), 'starts', 'ends', 'axes'], ['cut'], name='slice')
    )
    output = _dense(
        nodes, initializers, _transposed(nodes, 'cut', 'cut_t'), 'tail', seed=None if seed is None else seed + 1
    )
    return make_model(
        'aie1_transposed_slice',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [4, 16])],
        initializers=initializers,
    )


def _transposed_split_model(*, seed=None):
    """root [8, 16] viewed as [16, 8]; Split(axis=1) halves its ROWS, each half feeding its own dense."""
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 8)
    root = _dense(nodes, initializers, 'x', 'root', seed=seed)
    initializers.append(numpy_helper.from_array(np.asarray([4, 4], dtype=np.int64), 'split_sizes'))
    nodes.append(
        helper.make_node(
            'Split', [_transposed(nodes, root, 'root_t'), 'split_sizes'], ['up', 'down'], name='split', axis=1
        )
    )
    left = _dense(
        nodes, initializers, _transposed(nodes, 'up', 'up_t'), 'left', seed=None if seed is None else seed + 1
    )
    right = _dense(
        nodes, initializers, _transposed(nodes, 'down', 'down_t'), 'right', seed=None if seed is None else seed + 2
    )
    return make_model(
        'aie1_transposed_split',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(left, TensorProto.FLOAT, [4, 16]), (right, TensorProto.FLOAT, [4, 16])],
        initializers=initializers,
    )


def _transposed_concat_model(*, second_transposed: bool = True):
    """[4, 16] and [4, 16] viewed as [16, 4] each; Concat(axis=1) stacks their ROWS. With the second
    input left untransposed ([16, 16]) the two views disagree on what axis 1 is."""
    nodes, initializers = [], []
    first_input = _input(nodes, initializers, 'a', 4)
    second_input = _input(nodes, initializers, 'b', 4 if second_transposed else 16)
    first = _transposed(nodes, _dense(nodes, initializers, 'a', 'first'), 'first_t')
    second = _dense(nodes, initializers, 'b', 'second')
    second = _transposed(nodes, second, 'second_t') if second_transposed else second
    nodes.append(helper.make_node('Concat', [first, second], ['joined'], name='concat', axis=1))
    output = _dense(nodes, initializers, _transposed(nodes, 'joined', 'joined_t'), 'tail')
    rows = 8 if second_transposed else 20
    return make_model(
        'aie1_transposed_concat',
        nodes=nodes,
        inputs=[first_input, second_input],
        outputs=[(output, TensorProto.FLOAT, [rows, 16])],
        initializers=initializers,
    )


def _transposed_dense_model(*, seed=None):
    """root [16, 16] read transposed by tail: tail's rows are root's features."""
    nodes, initializers = [], []
    graph_input = _input(nodes, initializers, 'x', 16)
    root = _dense(nodes, initializers, 'x', 'root', seed=seed)
    output = _dense(
        nodes, initializers, _transposed(nodes, root, 'root_t'), 'tail', seed=None if seed is None else seed + 1
    )
    return make_model(
        'transposed_dense',
        nodes=nodes,
        inputs=[graph_input],
        outputs=[(output, TensorProto.FLOAT, [16, 16])],
        initializers=initializers,
    )


# AIE1 takes the input on a stream (a PLIO re-tiled by DMA needs more BDs than a tile has), and 4x8x4 mmuls, whose
# output blocks hold the columns of a transposed 4x8 input microtile.
TRANSPOSED_DENSE = {
    'aie-ml': (PART, {}),
    'aie-mlv2': ('xc2ve3858-ssva2112-2MP-e-S', {}),
    'aie1': (
        AIE1_PART,
        {
            'root': {
                'ports': {'inputs': 'stream'},
                'microtiling': {'microtile_m': 4, 'microtile_k': 8, 'microtile_n': 4},
            },
            'tail': {'microtiling': {'microtile_m': 4, 'microtile_k': 8, 'microtile_n': 4}},
        },
    ),
}


@pytest.mark.parametrize('case', TRANSPOSED_DENSE)
def test_a_dense_reads_a_transposed_input_in_shared_memory(case, tmp_path):
    part, directives = TRANSPOSED_DENSE[case]
    ctx = lower(_transposed_dense_model(), tmp_path, directives, part=part)
    edges = [e for e in ctx.ir.physical.plan['direct_edges'] if e['source'].startswith('root_aie')]
    assert [(e['target'], e['realization']) for e in edges] == [('tail_aie.in1[0]', 'shared_memory')]


@pytest.mark.requires_vitis
@pytest.mark.parametrize('case', TRANSPOSED_DENSE)
def test_a_dense_reading_a_transposed_input_matches_onnx(case, tmp_path):
    from helpers import assert_x86_matches_onnx

    part, directives = TRANSPOSED_DENSE[case]
    feeds = {'x_q': np.random.default_rng(5).integers(-8, 8, size=(16, 16), dtype=np.int8)}
    assert_x86_matches_onnx(_transposed_dense_model(seed=21), feeds, directives, tmp_path, max_code_diff=0, part=part)


def _view(ctx, op_type):
    node = next(n for n in ctx.ir.logical if n.op_type == op_type)
    return node.traits['concat_view' if op_type == 'concat' else 'slice_view'].data


def test_slice_of_a_transposed_value_cuts_the_canonical_axis(tmp_path):
    """The ONNX axis names an axis of the transposed view; the slice is recorded on the canonical tensor,
    so it cuts root's rows -- the first of its two row-band ports -- not its features."""
    ctx = lower(
        _transposed_slice_model(),
        tmp_path,
        part=AIE1_PART,
        directives={'root': _parallelism(2), 'tail': _parallelism(1)},
    )
    assert _view(ctx, 'slice')['axis'] == 0
    assert ('root_aie.out1[0]', 'tail_aie.in1[0]') in {
        (e['source'], e['target']) for e in ctx.ir.physical.plan['direct_edges']
    }


def test_split_of_a_transposed_value_cuts_the_canonical_axis(tmp_path):
    ctx = lower(
        _transposed_split_model(),
        tmp_path,
        part=AIE1_PART,
        directives={'root': _parallelism(2), 'left': _parallelism(1), 'right': _parallelism(1)},
    )
    assert _view(ctx, 'split')['axis'] == 0
    edges = {(e['source'], e['target']) for e in ctx.ir.physical.plan['direct_edges']}
    assert {('root_aie.out1[0]', 'left_aie.in1[0]'), ('root_aie.out1[1]', 'right_aie.in1[0]')} <= edges


def test_concat_of_transposed_values_joins_the_canonical_axis(tmp_path):
    ctx = lower(
        _transposed_concat_model(),
        tmp_path,
        part=AIE1_PART,
        directives={'first': _parallelism(1), 'second': _parallelism(1), 'tail': _parallelism(2)},
    )
    assert _view(ctx, 'concat')['axis'] == 0
    edges = {(e['source'], e['target']) for e in ctx.ir.physical.plan['direct_edges']}
    assert {('first_aie.out1[0]', 'tail_aie.in1[0]'), ('second_aie.out1[0]', 'tail_aie.in1[1]')} <= edges


def test_concat_of_differently_viewed_values_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r'different axis orders .* canonical axis \[0, 1\]'):
        lower(_transposed_concat_model(second_transposed=False), tmp_path, part=AIE1_PART)


def test_a_transpose_never_folds_under_a_view_op():
    """Folding a transpose out from under a view op would leave its axis naming the other order: the pass
    refuses rather than slice the wrong axis."""
    from types import SimpleNamespace

    from aie4ml.ir.context import CONTEXT_ATTR
    from aie4ml.ir.graph import LogicalIR, OpNode, TensorVar
    from aie4ml.passes.fold_views import FoldViewOps

    graph = LogicalIR()
    x, view, cut = (TensorVar(n, s) for n, s in (('x', (8, 16)), ('x_t', (16, 8)), ('cut', (16, 4))))
    for tensor in (x, view, cut):
        graph.add_tensor(tensor)
    graph.mark_graph_input('x')
    for name, op, inputs, outputs, meta in (
        ('t', 'transpose', [x], [view], {'perm': [1, 0], 'data_format': 'channels_last'}),
        ('slice', 'slice', [view], [cut], {'axis': 1, 'slices': [{'start': 0, 'extent': 4}]}),
    ):
        node = OpNode(name, op, inputs=inputs, outputs=outputs, metadata=meta)
        for tensor in inputs:
            tensor.consumers.append(node)
        for tensor in outputs:
            tensor.producer = node
        graph.add_node(node)
    holder = SimpleNamespace()
    setattr(holder, CONTEXT_ATTR, SimpleNamespace(ir=SimpleNamespace(logical=graph)))
    with pytest.raises(NotImplementedError, match='view op'):
        FoldViewOps().transform(holder)


@pytest.mark.requires_vitis
def test_split_of_a_transposed_value_matches_onnx(tmp_path):
    """Distinct weights, so halving features instead of rows would change every number."""
    from helpers import assert_x86_matches_onnx

    feeds = {'x_q': np.random.default_rng(3).integers(-8, 8, size=(8, 16), dtype=np.int8)}
    assert_x86_matches_onnx(
        _transposed_split_model(seed=11),
        feeds,
        {'root': _parallelism(2), 'left': _parallelism(1), 'right': _parallelism(1)},
        tmp_path,
        max_code_diff=0,
        part=AIE1_PART,
    )
