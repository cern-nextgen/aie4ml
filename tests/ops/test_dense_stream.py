"""Dense on core stream ports."""

from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.writer import AIEProjectEmitter
from helpers import PART, TensorProto, assert_x86_matches_onnx, helper, lower, make_model, numpy_helper, qdq

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'
ROWS, IN_FEAT, HIDDEN, OUT_FEAT = 16, 20, 24, 20
FRAC = 4


def _qparams(prefix: str, *, frac: int, elem_type: int = TensorProto.INT8) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _dense(nodes: list, inits: list, x: str, out: str, name: str, n_in: int, n_out: int, *, relu: bool, seed: int):
    """Gemm(x, W, b) [-> Relu] -> Q -> DQ with int8 weights and an int32 bias in the accumulator scale."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-8, 8, size=(n_in, n_out), dtype=np.int8)
    b = rng.integers(-64, 64, size=(n_out,), dtype=np.int32)
    inits += [
        numpy_helper.from_array(w, f'{name}_w_q'),
        numpy_helper.from_array(b, f'{name}_b_q'),
        *_qparams(f'{name}_w', frac=FRAC),
        *_qparams(f'{name}_b', frac=2 * FRAC, elem_type=TensorProto.INT32),
        *_qparams(f'{name}o', frac=FRAC),
    ]
    nodes.append(
        helper.make_node('DequantizeLinear', [f'{name}_w_q', f'{name}_w_scale', f'{name}_w_zp'], [f'{name}_w'])
    )
    nodes.append(
        helper.make_node('DequantizeLinear', [f'{name}_b_q', f'{name}_b_scale', f'{name}_b_zp'], [f'{name}_b'])
    )
    nodes.append(helper.make_node('Gemm', [x, f'{name}_w', f'{name}_b'], [f'{name}_mm'], name=name))
    pre_q = f'{name}_mm'
    if relu:
        nodes.append(helper.make_node('Relu', [pre_q], [f'{name}_relu'], name=f'{name}_relu'))
        pre_q = f'{name}_relu'
    qdq(nodes, pre_q, out, f'{name}o')


@pytest.fixture
def stream_dense_model():
    nodes: list = []
    inits: list = [*_qparams('x', frac=FRAC)]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'))
    # Chain: inner (2 chains, N split) -> inner (cascade over the producer's 2 slices) -> output.
    _dense(nodes, inits, 'x', 'h', 'l1', IN_FEAT, HIDDEN, relu=True, seed=1)
    _dense(nodes, inits, 'h', 'y_chain', 'l2', HIDDEN, OUT_FEAT, relu=False, seed=2)
    # Row-split branch straight from the graph input.
    _dense(nodes, inits, 'x', 'y_rows', 'lo', IN_FEAT, HIDDEN, relu=True, seed=3)
    return make_model(
        'stream_dense',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [ROWS, IN_FEAT])],
        outputs=[('y_chain', TensorProto.FLOAT, [ROWS, OUT_FEAT]), ('y_rows', TensorProto.FLOAT, [ROWS, HIDDEN])],
        initializers=inits,
    )


def _stream(cas_num: int, *, cas_length: int | None = None, contract: str | None = None) -> dict:
    parallelism: dict = {'cas_num': cas_num}
    if cas_length is not None:
        parallelism['cas_length'] = cas_length
    if contract is not None:
        parallelism['contract'] = contract
    return {'ports': 'stream', 'parallelism': parallelism}


DIRECTIVES = {'l1': _stream(2), 'l2': _stream(1), 'lo': _stream(2, contract='outer')}


def _feed() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.integers(-40, 40, size=(ROWS, IN_FEAT), dtype=np.int8)


# --------------------------------------------------------------------------- #
# contract: variants, ports, transport plan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_stream_dense_plan_has_no_buffers_or_dma(stream_dense_model, tmp_path, part):
    ctx = lower(stream_dense_model, tmp_path, DIRECTIVES, part=part)
    execution = ctx.ir.execution
    plan = ctx.ir.physical.plan

    assert execution.get('l1_aie').variant.variant_id == 'dense.b.r.stream.v1'
    assert execution.get('lo_aie').variant.variant_id == 'dense.b.r.row.stream.v1'
    l2 = execution.get('l2_aie')
    assert l2.variant.variant_id == 'dense.b.r.stream.v1'
    assert (l2.config.parallelism.cas_length, l2.config.parallelism.cas_num) == (2, 1)
    for name in ('l1_aie', 'l2_aie', 'lo_aie'):
        inst = execution.get(name)
        assert {binding.kind for binding in (*inst.ports.inputs.values(), *inst.ports.outputs.values())} == {'stream'}
        assert inst.variant.buffer_locations(inst.node, inst.config, 0) == ()

    assert plan['buffers'] == []
    assert plan['kernel_read_accesses'] == []
    assert plan['kernel_write_accesses'] == []
    assert {(edge['source'], edge['target']) for edge in plan['direct_edges']} == {
        ('ifm[0]', 'l1_aie.in1[0]'),
        ('ifm[1]', 'lo_aie.in1[0]'),
        ('ifm[2]', 'lo_aie.in1[1]'),
        ('l1_aie.out1[0]', 'l2_aie.in1[0]'),
        ('l1_aie.out1[1]', 'l2_aie.in1[1]'),
        ('l2_aie.out1[0]', 'ofm[0]'),
        ('lo_aie.out1[0]', 'ofm[1]'),
        ('lo_aie.out1[1]', 'ofm[2]'),
    }
    # A kernel-to-kernel stream edge has no buffer to share or copy.
    assert {edge['realization'] for edge in plan['direct_edges'] if 'realization' in edge} == {'stream'}


def test_stream_dense_io_ports_carry_padded_tiles(stream_dense_model, tmp_path):
    ctx = lower(stream_dense_model, tmp_path, DIRECTIVES, part=AIE1_PART)
    ports = {(item['direction'], item['port']): item['staging'] for item in ctx.ir.physical.plan['io_ports']}

    whole_input = ports[('input', 0)]
    assert whole_input['storage_layout'] == 'linear'
    assert whole_input['tiling_dimension'] == [32, ROWS]  # IN_FEAT padded to 2*K per column
    assert whole_input['io_tiling_dimension'] == [IN_FEAT, ROWS]
    assert whole_input['offset'] == [0, 0]

    second_row_slice = ports[('input', 2)]
    assert second_row_slice['tiling_dimension'] == [32, ROWS // 2]
    assert second_row_slice['io_tiling_dimension'] == [IN_FEAT, ROWS // 2]
    assert second_row_slice['offset'] == [0, ROWS // 2]

    chain_output = ports[('output', 0)]
    assert chain_output['tiling_dimension'] == [32, ROWS]
    assert chain_output['io_tiling_dimension'] == [OUT_FEAT, ROWS]

    # The internal stream edge: producer N slices equal consumer K slices, padded and raw alike.
    l1, l2 = ctx.ir.execution.get('l1_aie'), ctx.ir.execution.get('l2_aie')
    out_desc = l1.variant.describe_output_staging(l1.node, l1.config, l1.node.outputs[0].name, 1)
    in_desc = l2.variant.describe_input_staging(l2.node, l2.config, l2.node.inputs[0].name, 1)
    assert out_desc['storage_layout'] == in_desc['storage_layout'] == 'linear'
    assert out_desc['tiling_dimension'] == in_desc['tiling_dimension'] == [16, ROWS]
    assert out_desc['io_tiling_dimension'] == in_desc['io_tiling_dimension'] == [HIDDEN // 2, ROWS]
    assert out_desc['offset'] == in_desc['offset'] == [16, 0]


def test_stream_dense_emits_stream_graph(stream_dense_model, tmp_path):
    ctx = lower(stream_dense_model, tmp_path, DIRECTIVES, part=AIE1_PART)
    AIEProjectEmitter().emit(ctx)
    out = ctx.project_config.output_dir / 'src'

    parameters = (out / 'parameters.h').read_text()
    assert parameters.count('STREAM_IO = true') == 3
    assert 'BUFFER_LOCATIONS' not in parameters

    graph_plan = (out / 'graph_plan.h').read_text()
    assert 'read_access' not in graph_plan and 'write_access' not in graph_plan
    assert 'dimensions(' not in graph_plan
    assert 'connect<>(self.l1_aie.out1[1], self.l2_aie.in1[1]);' in graph_plan


# --------------------------------------------------------------------------- #
# contract: what a stream port refuses
# --------------------------------------------------------------------------- #


def test_stream_dense_refuses_buffer_consumer(stream_dense_model, tmp_path):
    directives = dict(DIRECTIVES, l2={'parallelism': {'cas_num': 1}})
    with pytest.raises(ConfigRefused, match=r'l1_aie\.out1 is a stream port but consumer l2_aie\.in1 is a buffer port'):
        lower(stream_dense_model, tmp_path, directives, part=AIE1_PART)


def test_stream_dense_refuses_memtile_route(stream_dense_model, tmp_path):
    directives = dict(DIRECTIVES, l2=dict(_stream(1), io_route={'inputs': {'l1_relu': 'memtile'}}))
    with pytest.raises(ConfigRefused, match=r'io_route=memtile requested on a stream port'):
        lower(stream_dense_model, tmp_path, directives, part=PART)


def test_stream_dense_refuses_unstageable_microtile(stream_dense_model, tmp_path):
    directives = dict(
        DIRECTIVES, l1=dict(_stream(2), microtiling={'microtile_m': 1, 'microtile_k': 16, 'microtile_n': 8})
    )
    with pytest.raises(ValueError, match=r'microtile N=8 .* cannot be staged from a stream'):
        lower(stream_dense_model, tmp_path, directives, part=AIE1_PART)


# --------------------------------------------------------------------------- #
# numerics -- needs Vitis
# --------------------------------------------------------------------------- #


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_stream_dense_matches_onnx(stream_dense_model, tmp_path, part):
    """Both flavours of stream staging (2x8x8 zipped rows on AIE, 4x8x8 on AIE-ML) match onnxruntime."""
    assert_x86_matches_onnx(
        stream_dense_model,
        {'x_q': _feed()},
        DIRECTIVES,
        tmp_path,
        project='stream_dense',
        part=part,
        max_code_diff=1,
    )
