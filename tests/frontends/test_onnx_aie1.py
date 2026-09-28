from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aie4ml.device_catalog import resolve_device
from aie4ml.errors import ConfigRefused
from aie4ml.frontends.onnx import from_onnx, lower_onnx_model
from aie4ml.ir import TraitInstance
from aie4ml.op_impls.common_types import PortBinding, kernel_endpoints, to_plain
from aie4ml.op_impls.families.matmul.common import select_generation_key
from aie4ml.op_impls.families.matmul.matmul import MatmulOpImplVariant, MatmulRowWiseOpImplVariant
from aie4ml.op_impls.utils.precision import infer_accumulator_tag
from aie4ml.passes import PackKernelArtifacts, Resolve
from aie4ml.writer import AIEProjectEmitter
from helpers import TensorProto, helper, make_model, numpy_helper
from jinja2 import Environment, FileSystemLoader

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'


@pytest.mark.parametrize(
    ('part', 'expected_width'),
    [(AIE1_PART, 32), ('xilinx_vek280_base_202610_1', 32), ('vek385_base', 64)],
)
def test_device_stream_switch_width(part, expected_width):
    device, _ = resolve_device(part, {})

    assert device.stream_switch_width_bits == expected_width


def test_matmul_ports_name_their_kernel_endpoints():
    node = SimpleNamespace(
        inputs=[SimpleNamespace(name='a'), SimpleNamespace(name='b')],
        outputs=[SimpleNamespace(name='c')],
        roles={'a': 'lhs', 'b': 'rhs'},
    )
    inner = SimpleNamespace(parallelism=SimpleNamespace(cas_length=2, cas_num=3, contract='inner'))
    outer = SimpleNamespace(parallelism=SimpleNamespace(cas_length=2, cas_num=3, contract='outer'))

    ports = MatmulOpImplVariant().build_ports(node, inner)
    assert ports.inputs['a'].endpoints[1] == ('kk[1].in[0]', 'kk[3].in[0]', 'kk[5].in[0]')
    ports = MatmulRowWiseOpImplVariant().build_ports(node, outer)
    assert ports.inputs['a'].endpoints[4] == ('kk[4].in[0]',)
    assert ports.inputs['b'].endpoints[4] == ('kk[4].in[1]',)
    assert ports.outputs['c'].endpoints[2] == ('kk[5].out[0]',)


def _qparams(prefix: str, elem_type: int, *, frac: int = 4) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [2.0**-frac]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _dense_model(
    lhs_type: int = TensorProto.INT8,
    rhs_type: int = TensorProto.INT8,
    out_features: int = 8,
    in_features: int = 16,
):
    np_type = {TensorProto.INT8: np.int8, TensorProto.INT16: np.int16}
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w'], name='w_dq'),
        helper.make_node('MatMul', ['x', 'w'], ['mm'], name='dense'),
        helper.make_node('QuantizeLinear', ['mm', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    return make_model(
        'aie1_dense',
        nodes=nodes,
        inputs=[('x_q', lhs_type, [8, in_features])],
        outputs=[('y', TensorProto.FLOAT, [8, out_features])],
        initializers=[
            *_qparams('x', lhs_type),
            *_qparams('w', rhs_type),
            *_qparams('y', TensorProto.INT8),
            numpy_helper.from_array(np.ones((in_features, out_features), dtype=np_type[rhs_type]), 'w_q'),
        ],
    )


def _resolve_dense(model, tmp_path, *, part=AIE1_PART, directives=None):
    layer_directives = {'dense': dict(directives)} if directives else {}
    ctx = lower_onnx_model(
        model,
        {
            'Part': part,
            'AIEConfig': {'BatchSize': 8, 'Iterations': 1},
            'LayerDirectives': layer_directives,
        },
        output_dir=tmp_path,
        project_name='aie1_dense',
    )
    Resolve().transform(ctx)
    return ctx, ctx.ir.execution.get('dense_aie').config


def _run_pipeline(model, tmp_path, *, part=AIE1_PART, directives=None, project='aie1_dense'):
    aie_model = from_onnx(
        model,
        {
            'Part': part,
            'AIEConfig': {'BatchSize': 8, 'Iterations': 1},
            'LayerDirectives': dict(directives or {}),
        },
        output_dir=tmp_path / project,
        project_name=project,
    )
    aie_model.run_pipeline()
    return aie_model


def _fanout_dense_model():
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w0_q', 'w0_scale', 'w0_zp'], ['w0'], name='w0_dq'),
        helper.make_node('MatMul', ['x', 'w0'], ['root_mm'], name='root'),
        helper.make_node('QuantizeLinear', ['root_mm', 'root_scale', 'root_zp'], ['root_q'], name='root_q'),
        helper.make_node('DequantizeLinear', ['root_q', 'root_scale', 'root_zp'], ['root_out'], name='root_dq'),
        helper.make_node('DequantizeLinear', ['w1_q', 'w1_scale', 'w1_zp'], ['w1'], name='w1_dq'),
        helper.make_node('MatMul', ['root_out', 'w1'], ['left_mm'], name='left'),
        helper.make_node('QuantizeLinear', ['left_mm', 'left_scale', 'left_zp'], ['left_q'], name='left_q'),
        helper.make_node('DequantizeLinear', ['left_q', 'left_scale', 'left_zp'], ['left_y'], name='left_dq'),
        helper.make_node('DequantizeLinear', ['w2_q', 'w2_scale', 'w2_zp'], ['w2'], name='w2_dq'),
        helper.make_node('MatMul', ['root_out', 'w2'], ['right_mm'], name='right'),
        helper.make_node('QuantizeLinear', ['right_mm', 'right_scale', 'right_zp'], ['right_q'], name='right_q'),
        helper.make_node('DequantizeLinear', ['right_q', 'right_scale', 'right_zp'], ['right_y'], name='right_dq'),
    ]
    initializers = [
        *_qparams('x', TensorProto.INT8),
        *_qparams('w0', TensorProto.INT8),
        *_qparams('root', TensorProto.INT8),
        *_qparams('w1', TensorProto.INT8),
        *_qparams('left', TensorProto.INT8),
        *_qparams('w2', TensorProto.INT8),
        *_qparams('right', TensorProto.INT8),
        numpy_helper.from_array(np.ones((16, 16), dtype=np.int8), 'w0_q'),
        numpy_helper.from_array(np.ones((16, 16), dtype=np.int8), 'w1_q'),
        numpy_helper.from_array(np.ones((16, 16), dtype=np.int8), 'w2_q'),
    ]
    return make_model(
        'aie1_dense_fanout',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [8, 16])],
        outputs=[('left_y', TensorProto.FLOAT, [8, 16]), ('right_y', TensorProto.FLOAT, [8, 16])],
        initializers=initializers,
    )


def _dense_stack_model():
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w0_q', 'w0_scale', 'w0_zp'], ['w0'], name='w0_dq'),
        helper.make_node('MatMul', ['x', 'w0'], ['hidden'], name='dense0'),
        helper.make_node('QuantizeLinear', ['hidden', 'hidden_scale', 'hidden_zp'], ['hidden_q'], name='hidden_q'),
        helper.make_node(
            'DequantizeLinear', ['hidden_q', 'hidden_scale', 'hidden_zp'], ['hidden_dq'], name='hidden_dq'
        ),
        helper.make_node('DequantizeLinear', ['w1_q', 'w1_scale', 'w1_zp'], ['w1'], name='w1_dq'),
        helper.make_node('MatMul', ['hidden_dq', 'w1'], ['output'], name='dense1'),
        helper.make_node('QuantizeLinear', ['output', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    return make_model(
        'aie1_dense_stack',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [8, 128])],
        outputs=[('y', TensorProto.FLOAT, [8, 64])],
        initializers=[
            *_qparams('x', TensorProto.INT8),
            *_qparams('w0', TensorProto.INT8),
            *_qparams('hidden', TensorProto.INT8),
            *_qparams('w1', TensorProto.INT8),
            *_qparams('y', TensorProto.INT8),
            numpy_helper.from_array(np.ones((128, 256), dtype=np.int8), 'w0_q'),
            numpy_helper.from_array(np.ones((256, 64), dtype=np.int8), 'w1_q'),
        ],
    )


def _add_model():
    nodes = [
        helper.make_node('DequantizeLinear', ['lhs_q', 'lhs_scale', 'lhs_zp'], ['lhs'], name='lhs_dq'),
        helper.make_node('DequantizeLinear', ['rhs_q', 'rhs_scale', 'rhs_zp'], ['rhs'], name='rhs_dq'),
        helper.make_node('Add', ['lhs', 'rhs'], ['sum'], name='add'),
        helper.make_node('QuantizeLinear', ['sum', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    return make_model(
        'aie1_add',
        nodes=nodes,
        inputs=[
            ('lhs_q', TensorProto.INT8, [8, 16]),
            ('rhs_q', TensorProto.INT8, [8, 16]),
        ],
        outputs=[('y', TensorProto.FLOAT, [8, 16])],
        initializers=[
            *_qparams('lhs', TensorProto.INT8),
            *_qparams('rhs', TensorProto.INT8),
            *_qparams('y', TensorProto.INT8),
        ],
    )


def _normalization_chain_model():
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w'], name='w_dq'),
        helper.make_node('MatMul', ['x', 'w'], ['mm'], name='dense'),
        helper.make_node('QuantizeLinear', ['mm', 'dense_scale', 'dense_zp'], ['dense_q'], name='dense_q'),
        helper.make_node('DequantizeLinear', ['dense_q', 'dense_scale', 'dense_zp'], ['dense_out'], name='dense_dq'),
        helper.make_node('DequantizeLinear', ['skip_q', 'skip_scale', 'skip_zp'], ['skip'], name='skip_dq'),
        helper.make_node('MatMul', ['skip', 'w'], ['skip_mm'], name='skip_dense'),
        helper.make_node(
            'QuantizeLinear', ['skip_mm', 'dense_scale', 'dense_zp'], ['skip_dense_q'], name='skip_dense_q'
        ),
        helper.make_node(
            'DequantizeLinear', ['skip_dense_q', 'dense_scale', 'dense_zp'], ['skip_dense_out'], name='skip_dense_dq'
        ),
        helper.make_node('Add', ['dense_out', 'skip_dense_out'], ['sum'], name='add'),
        helper.make_node('QuantizeLinear', ['sum', 'add_scale', 'add_zp'], ['add_q'], name='add_q'),
        helper.make_node('DequantizeLinear', ['add_q', 'add_scale', 'add_zp'], ['add_out'], name='add_dq'),
        helper.make_node(
            'LayerNormalization', ['add_out', 'gamma', 'beta'], ['normalized'], name='layernorm', epsilon=2.0**-8
        ),
        helper.make_node('QuantizeLinear', ['normalized', 'norm_scale', 'norm_zp'], ['norm_q'], name='norm_q'),
        helper.make_node('DequantizeLinear', ['norm_q', 'norm_scale', 'norm_zp'], ['norm_out'], name='norm_dq'),
        helper.make_node('Softmax', ['norm_out'], ['probabilities'], name='softmax', axis=-1),
        helper.make_node('QuantizeLinear', ['probabilities', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    return make_model(
        'aie1_normalization_chain',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [8, 32]), ('skip_q', TensorProto.INT8, [8, 32])],
        outputs=[('y', TensorProto.FLOAT, [8, 32])],
        initializers=[
            *_qparams('x', TensorProto.INT8),
            *_qparams('w', TensorProto.INT8),
            *_qparams('dense', TensorProto.INT8),
            *_qparams('skip', TensorProto.INT8),
            *_qparams('add', TensorProto.INT8),
            *_qparams('norm', TensorProto.INT8),
            *_qparams('y', TensorProto.UINT8, frac=8),
            numpy_helper.from_array(np.eye(32, dtype=np.int8), 'w_q'),
            numpy_helper.from_array(np.ones(32, dtype=np.float32), 'gamma'),
            numpy_helper.from_array(np.zeros(32, dtype=np.float32), 'beta'),
        ],
    )


def test_aie1_catalog_capabilities_and_raw_part_target(tmp_path):
    device, _ = resolve_device(AIE1_PART, {})

    assert device.generation == 'AIE'
    assert device.columns == 59
    assert (device.column_start, device.preferred_column_start) == (1, 6)
    assert device.rows == 8
    assert device.plio_width_bits == 128
    assert device.core_stream_inputs == 2
    assert device.core_stream_outputs == 2
    assert device.stream_switch_width_bits == 32
    assert device.cascade_width_bits == 384
    assert device.has_memtile is False
    assert device.bank_count == 4
    assert device.bank_mem_bytes == 8 * 1024
    assert device.tile_mem_bytes == 32 * 1024
    assert device.vector_bytes == 32
    assert device.cascade_layout == 'alternating_horizontal'
    assert device.aie_compiler_target == 'part'

    template_root = Path(AIEProjectEmitter()._template_root) / 'firmware'
    env = Environment(loader=FileSystemLoader(str(template_root)), trim_blocks=True, lstrip_blocks=True)
    ctx = SimpleNamespace(
        device=device,
        project_config=SimpleNamespace(project_name='probe', stamp=None),
    )
    AIEProjectEmitter()._render_makefile(tmp_path, ctx, env, None)
    makefile = (tmp_path / 'Makefile').read_text()
    assert '--target=hw $(AIE_FLAGS_HW) $(AIE_JOBS)' in makefile  # make -jN reaches the kernel compile
    assert f'AIE_PART      ?= {AIE1_PART}' in makefile
    assert 'AIE_TARGET    := --part=$(AIE_PART)' in makefile
    assert '--platform=$(PLATFORM) $(APP_NAME).cpp' not in makefile
    assert makefile.count('set -o pipefail; v++ --compile') == 2

    AIEProjectEmitter()._render_makefile(tmp_path, ctx, env, {'kernels': []})
    hardware_makefile = (tmp_path / 'Makefile').read_text()
    xpfm_error = f'Hardware system linking requires a Vitis .xpfm platform; {AIE1_PART} is a raw AIE part target.'
    assert 'XPFM_GUARD := guard-XPFM' in hardware_makefile
    assert f'guard-XPFM:\n\t$(error {xpfm_error})' in hardware_makefile
    assert 'hw:     $(XPFM_GUARD) aiecom kernels xsa_hw     host package_hw' in hardware_makefile
    assert f'\n$(error {xpfm_error})' not in hardware_makefile

    (tmp_path / 'src' / 'kernels').mkdir(parents=True)
    (tmp_path / 'app.cpp').touch()
    (tmp_path / 'aie.cfg').touch()
    make_env = {**os.environ, 'VITIS_HOME': '/tmp/vitis'}
    x86com = subprocess.run(['make', '-n', 'x86com'], cwd=tmp_path, env=make_env, capture_output=True, text=True)
    assert x86com.returncode == 0, x86com.stderr
    assert f'--part={AIE1_PART}' in x86com.stdout

    hardware = subprocess.run(['make', '-n', 'hw'], cwd=tmp_path, env=make_env, capture_output=True, text=True)
    assert hardware.returncode != 0
    assert xpfm_error in hardware.stderr


@pytest.mark.parametrize('generation', ['', 'AIE-ML3', 'not-a-device'])
def test_unknown_generation_is_rejected(generation):
    with pytest.raises(ValueError, match='Unknown AIE generation'):
        select_generation_key(generation)


def test_aie1_onnx_int8_dense_prefers_direct_compatible_microtile(tmp_path):
    _ctx, config = _resolve_dense(_dense_model(), tmp_path)

    got = (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n)
    assert got == (2, 8, 8)
    assert config.accumulator_tag == 'acc48'
    assert config.alternating_horizontal is True


@pytest.mark.parametrize('microtile', [(4, 8, 4), (1, 16, 8)])
def test_aie1_onnx_int8_dense_accepts_other_native_microtiles(tmp_path, microtile):
    m, k, n = microtile
    _ctx, config = _resolve_dense(
        _dense_model(),
        tmp_path,
        directives={'microtiling': {'microtile_m': m, 'microtile_k': k, 'microtile_n': n}},
    )

    assert (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n) == microtile
    assert config.accumulator_tag == 'acc48'


def test_aie1_onnx_int16_int8_dense_uses_compile_proven_shape_and_acc48(tmp_path):
    _ctx, config = _resolve_dense(_dense_model(TensorProto.INT16, TensorProto.INT8), tmp_path)

    assert (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n) == (4, 4, 4)
    assert config.accumulator_tag == 'acc48'


def test_aie_mlv2_int16_dense_takes_a_dense_acc64_shape(tmp_path):
    """MLv2 builds int16 x int16 in acc64 only at K=4: its 8x2x8 is acc32-only and its K=8 shapes are sparse."""
    model = _dense_model(TensorProto.INT16, TensorProto.INT16)
    _ctx, config = _resolve_dense(model, tmp_path, part='xc2ve3858-ssva2112-2mp-e-s')
    assert (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n) == (4, 4, 8)
    assert config.accumulator_tag == 'acc64'
    with pytest.raises(ValueError, match=r'microtiling \(4, 8, 8\) not supported'):
        _resolve_dense(
            model,
            tmp_path,
            part='xc2ve3858-ssva2112-2mp-e-s',
            directives={'microtiling': {'microtile_m': 4, 'microtile_k': 8, 'microtile_n': 8}},
        )


@pytest.mark.parametrize(
    'part', ['xcve2802-vsvh1760-2mp-e-s', 'xc2ve3858-ssva2112-2mp-e-s'], ids=['aie-ml', 'aie-mlv2']
)
def test_a_graph_output_off_a_memory_tile_moves_whole_words(tmp_path, part):
    """10 int8 features end mid-word: the memory tile reads each row as 12 and the host trims."""
    plan = _run_pipeline(_dense_model(out_features=10), tmp_path, part=part).context.ir.physical.plan
    (out,) = [p for p in plan['io_ports'] if p['direction'] == 'output']
    assert out['descriptor']['io_boundary_dimension'][0] == 10
    assert out['descriptor']['tiling_dimension'][0] == out['descriptor']['boundary_dimension'][0] == 12
    assert out['staging']['tiling_dimension'][0] == 12


@pytest.mark.parametrize(
    'part', ['xcve2802-vsvh1760-2mp-e-s', 'xc2ve3858-ssva2112-2mp-e-s'], ids=['aie-ml', 'aie-mlv2']
)
def test_every_output_port_reads_its_slice_inside_the_dma_boundary(tmp_path, part):
    """A memory tile's DMA zero-fills past its boundary, a buffer coordinate: the second of two output chains starts
    mid-buffer, and its slice must still lie inside."""
    plan = _run_pipeline(
        _dense_model(out_features=64), tmp_path, part=part, directives={'dense': {'parallelism': {'cas_num': 2}}}
    ).context.ir.physical.plan
    outs = [p['descriptor'] for p in plan['io_ports'] if p['direction'] == 'output']
    assert len(outs) == 2
    for d in outs:
        for offset, tile, io_end, end in zip(
            d['offset'], d['io_tiling_dimension'], d['io_boundary_dimension'], d['boundary_dimension']
        ):
            assert end >= min(offset + tile, io_end), d


def test_aie1_onnx_int8_int16_dense_is_rejected_before_shift_resolution(tmp_path):
    with pytest.raises(RuntimeError, match=r'unsupported int8 x int16.*output shift'):
        _resolve_dense(_dense_model(TensorProto.INT8, TensorProto.INT16), tmp_path)


def test_aie1_acc80_vocabulary_covers_32_bit_integer_path():
    device, _ = resolve_device(AIE1_PART, {})

    from aie4ml.aie_types import AIEDataType

    assert infer_accumulator_tag(device, AIEDataType('int32'), AIEDataType('int32'), None) == 'acc80'


def test_aie1_onnx_dense_rejects_emulated_int8_microtile(tmp_path):
    with pytest.raises(ValueError, match=r'microtiling \(4, 16, 4\) not supported'):
        _resolve_dense(
            _dense_model(),
            tmp_path,
            directives={'microtiling': {'microtile_m': 4, 'microtile_k': 16, 'microtile_n': 4}},
        )


def test_aie1_onnx_dense_rejects_unenabled_int16_int16(tmp_path):
    with pytest.raises(ConfigRefused, match=r'no dense variant matches.*generation=.AIE.'):
        _resolve_dense(_dense_model(TensorProto.INT16, TensorProto.INT16), tmp_path)


def test_a_misspelled_directive_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown directive.*'paralellism'"):
        _resolve_dense(_dense_model(), tmp_path, directives={'paralellism': {'cas_num': 2}})


def test_a_directive_the_variant_does_not_read_is_refused(tmp_path):
    with pytest.raises(NotImplementedError, match=r"does not implement the directive\(s\) \['layout'\]"):
        _resolve_dense(_dense_model(), tmp_path, directives={'layout': 'tiled'})


@pytest.mark.parametrize(
    ('view', 'match'),
    [({'kind': 'flatten_2d'}, 'does not write the output view'), ({'kind': 'flatten_2d', 'shape': (8, 8)}, 'exactly')],
)
def test_an_output_view_the_family_cannot_write_is_refused(tmp_path, view, match):
    ctx = lower_onnx_model(
        _dense_model(),
        {'Part': AIE1_PART, 'AIEConfig': {'BatchSize': 8, 'Iterations': 1}},
        output_dir=tmp_path,
        project_name='aie1_dense',
    )
    next(node for node in ctx.ir.logical if node.name == 'dense_aie').add_trait(TraitInstance('output_view', view))
    with pytest.raises(ValueError, match=match):
        Resolve().transform(ctx)


def test_resolving_again_repacks_the_current_weights(tmp_path):
    ctx = _run_pipeline(_dense_model(), tmp_path).context
    first = ctx.ir.execution.get('dense_aie').artifacts['packed_weights'].copy()
    weights = next(t for t in ctx.ir.execution.get('dense_aie').node.inputs if t.is_parameter)
    weights.data = -weights.data
    Resolve().transform(ctx)
    PackKernelArtifacts().transform(ctx)
    assert np.array_equal(ctx.ir.execution.get('dense_aie').artifacts['packed_weights'], -first)


def test_replanning_a_context_matches_planning_a_fresh_one(tmp_path):
    """An optimizer replans one context per candidate: neither a larger split before nor a candidate
    that failed during placement may leave anything behind."""

    def split(dense0, dense1, **extra):
        return {'dense0': {'parallelism': dense0, **extra}, 'dense1': {'parallelism': dense1}}

    def replan(model, directives):
        for node in model.context.ir.logical:
            node.directives = directives.get(node.name.removesuffix('_aie'), node.directives)
        model.run_pipeline()

    target = split({'cas_num': 2, 'cas_length': 2}, {'cas_num': 1})
    fresh = _run_pipeline(_dense_stack_model(), tmp_path, directives=target, project='fresh')
    reused = _run_pipeline(
        _dense_stack_model(), tmp_path, directives=split({'cas_num': 4, 'cas_length': 2}, {'cas_num': 2})
    )
    with pytest.raises(ValueError, match='Invalid fixed anchor'):
        replan(reused, split({'cas_num': 2, 'cas_length': 2}, {'cas_num': 1}, placement={'col': 999, 'row': 0}))
    replan(reused, target)
    assert to_plain(reused.context.ir.physical.to_dict()) == to_plain(fresh.context.ir.physical.to_dict())


def test_a_tensor_read_as_two_operands_is_refused(tmp_path):
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('Add', ['x', 'x'], ['sum'], name='add'),
        helper.make_node('QuantizeLinear', ['sum', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    model = make_model(
        'self_add',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [8, 16])],
        outputs=[('y', TensorProto.FLOAT, [8, 16])],
        initializers=[*_qparams('x', TensorProto.INT8), *_qparams('y', TensorProto.INT8)],
    )
    with pytest.raises(NotImplementedError, match='as more than one operand'):
        _run_pipeline(model, tmp_path)


@pytest.mark.parametrize(
    ('part', 'expected_microtile'),
    [('xilinx_vek280_base_202610_1', (4, 8, 8)), ('vek385_base', (8, 8, 8))],
)
def test_existing_ml_generation_default_resolution_is_unchanged(tmp_path, part, expected_microtile):
    _ctx, config = _resolve_dense(_dense_model(), tmp_path, part=part)
    got = (config.microtiling.microtile_m, config.microtiling.microtile_k, config.microtiling.microtile_n)
    assert got == expected_microtile
    assert config.alternating_horizontal is False


def test_port_binding_default_is_an_explicit_buffer_in_serialization():
    implicit = PortBinding('in1', 2, endpoints=kernel_endpoints(2, 'in[0]'))

    assert implicit == PortBinding('in1', 2, 'buffer', (('kk[0].in[0]',), ('kk[1].in[0]',)))
    assert to_plain(implicit) == {
        'group': 'in1',
        'count': 2,
        'kind': 'buffer',
        'endpoints': [['kk[0].in[0]'], ['kk[1].in[0]']],
    }
    with pytest.raises(ValueError, match='expected kernel endpoints'):
        PortBinding('in1', 2)


def test_aie1_direct_boundaries_publish_linear_io_and_dma_accesses(tmp_path):
    aie_model = _run_pipeline(_dense_model(out_features=16), tmp_path)
    plan = aie_model.context.ir.physical.plan

    assert plan['buffers'] == []
    assert plan['graph_input_count'] == 1
    assert plan['graph_output_count'] == 1
    assert {(edge['source'], edge['target']) for edge in plan['direct_edges']} == {
        ('ifm[0]', 'dense_aie.in1[0]'),
        ('dense_aie.out1[0]', 'ofm[0]'),
    }
    assert [(port['direction'], port['tensor'], port['port']) for port in plan['io_ports']] == [
        ('input', 'x_q', 0),
        ('output', 'y', 0),
    ]
    assert [item['endpoint'] for item in plan['kernel_write_accesses']] == ['dense_aie.kk[0].in[0]']
    assert [item['endpoint'] for item in plan['kernel_read_accesses']] == ['dense_aie.kk[0].out[0]']
    for access in (plan['kernel_write_accesses'][0], plan['kernel_read_accesses'][0]):
        descriptor = access['descriptor']
        assert descriptor['storage_layout'] == 'microtiled'
        assert descriptor['buffer_dimension'] == [16, 2, 4]
        assert descriptor['tiling_dimension'] == [8, 1, 1]
        assert descriptor['tile_traversal'] == [
            {'dimension': 1, 'stride': 1, 'wrap': 2},
            {'dimension': 0, 'stride': 8, 'wrap': 2},
            {'dimension': 2, 'stride': 1, 'wrap': 4},
        ]

    AIEProjectEmitter().emit(aie_model.context)
    graph_plan = (aie_model.context.project_config.output_dir / 'src' / 'graph_plan.h').read_text()
    assert 'write_access(self.dense_aie.kk[0].in[0])' in graph_plan
    assert 'read_access(self.dense_aie.kk[0].out[0])' in graph_plan
    assert 'connect<>(self.ifm[0], self.dense_aie.in1[0]);' in graph_plan
    assert 'connect<>(self.dense_aie.out1[0], self.ofm[0]);' in graph_plan

    parameters = (aie_model.context.project_config.output_dir / 'src' / 'parameters.h').read_text()
    assert 'KERNEL_LOCATIONS' not in parameters
    assert '{ -1, 0, 2, 0, 3 }' in parameters


def test_aie1_linear_add_keeps_concrete_dma_accesses(tmp_path):
    aie_model = _run_pipeline(_add_model(), tmp_path, project='aie1_add')
    inst = aie_model.context.ir.execution.get('add_aie')
    plan = aie_model.context.ir.physical.plan

    assert inst.config.accumulator_tag == 'acc48'
    assert plan['buffers'] == []
    assert [item['endpoint'] for item in plan['kernel_write_accesses']] == [
        'add_aie.kk[0].in[0]',
        'add_aie.kk[0].in[1]',
    ]
    assert [item['endpoint'] for item in plan['kernel_read_accesses']] == ['add_aie.kk[0].out[0]']
    assert all(
        item['descriptor']['storage_layout'] == 'linear'
        for item in (*plan['kernel_write_accesses'], *plan['kernel_read_accesses'])
    )


def test_aie1_tiled_normalization_chain_uses_direct_acc48_kernels(tmp_path):
    directives = {
        'layernorm': {'layout': 'tiled', 'parallelism': {'cas_num': 1}},
        'softmax': {'parallelism': {'cas_num': 1}},
    }
    aie_model = _run_pipeline(
        _normalization_chain_model(), tmp_path, directives=directives, project='aie1_normalization_chain'
    )
    execution = aie_model.context.ir.execution
    plan = aie_model.context.ir.physical.plan

    # One-tile rows flow left to right on AIE too: on an odd row the core reaches east, so its input
    # lives in its own tile and its output in the east neighbour's; on an even row, west and own.
    for name in ('add_aie', 'layernorm_aie', 'softmax_aie'):
        inst = execution.get(name)
        assert inst.config.alternating_horizontal is True
        for row, (input_col, output_col) in ((0, (-1, 0)), (1, (0, 1))):
            locations = inst.variant.buffer_locations(inst.node, inst.config, anchor_row=row)
            assert {loc.rel_col for loc in locations if loc.port_group == 'in1'} == {input_col}
            assert {loc.rel_col for loc in locations if loc.port_group == 'out1'} == {output_col}
    assert execution.get('layernorm_aie').config.accumulator_tag == 'acc48'
    assert execution.get('softmax_aie').config.accumulator_tag == 'acc48'
    assert execution.get('layernorm_aie').variant.variant_id == 'layer_norm.i8.tiled.v1'
    assert execution.get('softmax_aie').variant.variant_id == 'softmax.exp.i8.tiled.v1'
    assert plan['buffers'] == []
    assert ('dense_aie.out1[0]', 'add_aie.in1[0]') in {
        (edge['source'], edge['target']) for edge in plan['direct_edges']
    }
    assert ('skip_dense_aie.out1[0]', 'add_aie.in2[0]') in {
        (edge['source'], edge['target']) for edge in plan['direct_edges']
    }
    assert ('add_aie.out1[0]', 'layernorm_aie.in1[0]') in {
        (edge['source'], edge['target']) for edge in plan['direct_edges']
    }
    assert ('layernorm_aie.out1[0]', 'softmax_aie.in1[0]') in {
        (edge['source'], edge['target']) for edge in plan['direct_edges']
    }
    assert all('.kk[' in item['endpoint'] for item in plan['kernel_write_accesses'])
    assert all('.kk[' in item['endpoint'] for item in plan['kernel_read_accesses'])

    AIEProjectEmitter().emit(aie_model.context)
    kernel_dir = aie_model.context.project_config.output_dir / 'src' / 'kernels'
    for family, graph_name in (
        ('elementwise_add', 'elementwise_add_graph.h'),
        ('layer_norm', 'layer_norm_graph.h'),
        ('softmax', 'softmax_graph.h'),
    ):
        graph = (kernel_dir / family / graph_name).read_text()
        assert 'adf::bank(tileCol, tileRow, 1)' in graph


def test_aie1_outer_parallel_dense_uses_direct_boundary_ports(tmp_path):
    aie_model = _run_pipeline(
        _dense_model(out_features=16),
        tmp_path,
        directives={'dense': {'parallelism': {'contract': 'outer', 'cas_num': 2, 'cas_length': 1}}},
    )
    plan = aie_model.context.ir.physical.plan

    assert plan['buffers'] == []
    assert plan['graph_input_count'] == 2
    assert plan['graph_output_count'] == 2
    assert [item['endpoint'] for item in plan['kernel_write_accesses']] == [
        'dense_aie.kk[0].in[0]',
        'dense_aie.kk[1].in[0]',
    ]
    assert [item['endpoint'] for item in plan['kernel_read_accesses']] == [
        'dense_aie.kk[0].out[0]',
        'dense_aie.kk[1].out[0]',
    ]


def test_aie1_dense_cascade_ports_follow_logical_snake_order(tmp_path):
    aie_model = _run_pipeline(
        _dense_model(out_features=32),
        tmp_path,
        directives={'dense': {'parallelism': {'contract': 'inner', 'cas_num': 2, 'cas_length': 2}}},
        project='aie1_cascade',
    )
    plan = aie_model.context.ir.physical.plan
    inst = aie_model.context.ir.execution.get('dense_aie')
    anchor_row = aie_model.context.ir.physical.placements['dense_aie']['row']
    locations = inst.variant.buffer_locations(inst.node, inst.config, anchor_row)

    assert [item['endpoint'] for item in plan['kernel_write_accesses']] == [
        'dense_aie.kk[0].in[0]',
        'dense_aie.kk[2].in[0]',
        'dense_aie.kk[1].in[0]',
        'dense_aie.kk[3].in[0]',
    ]
    assert [item['endpoint'] for item in plan['kernel_read_accesses']] == [
        'dense_aie.kk[1].out[0]',
        'dense_aie.kk[3].out[0]',
    ]
    assert [
        (location.rel_col, location.rel_row, location.banks)
        for location in locations
        if location.port_group == 'in1' and location.port == 0
    ] == [
        (-1, 0, (0, 3)),
        (2, 1, (0, 3)),
    ]
    assert [
        (location.port, location.rel_col, location.rel_row, location.banks)
        for location in locations
        if location.port_group == 'out1'
    ] == [(0, 1, 0, (0, 3)), (1, 0, 1, (0, 3))]

    AIEProjectEmitter().emit(aie_model.context)
    graph = (
        aie_model.context.project_config.output_dir / 'src' / 'kernels' / 'dense_bias_relu' / 'dense_bias_relu_graph.h'
    ).read_text()
    assert 'ConfigT::ALTERNATING_HORIZONTAL' in graph
    assert 'ConfigT::IN1_BUFFER_LOCATIONS[idx]' in graph


def test_aie1_dense_stack_inherits_direct_producer_partition(tmp_path):
    """dense0 held to four output chains hands dense1 four slices, which dense1 takes as a four-stage cascade."""
    directives = {'dense0': {'parallelism': {'cas_num': 4, 'cas_length': 1}}}
    aie_model = _run_pipeline(_dense_stack_model(), tmp_path, directives=directives, project='aie1_dense_stack')
    first = aie_model.context.ir.execution.get('dense0_aie').config
    second = aie_model.context.ir.execution.get('dense1_aie').config

    assert (first.parallelism.cas_num, first.parallelism.cas_length) == (4, 1)
    assert (second.parallelism.cas_num, second.parallelism.cas_length) == (1, 4)
    assert first.microtiling == second.microtiling

    internal = [
        edge
        for edge in aie_model.context.ir.physical.plan['direct_edges']
        if edge['source'].startswith('dense0_aie.') and edge['target'].startswith('dense1_aie.')
    ]
    assert [(edge['source'], edge['target']) for edge in internal] == [
        (f'dense0_aie.out1[{port}]', f'dense1_aie.in1[{port}]') for port in range(4)
    ]


def test_aie1_direct_buffer_fanout_keeps_each_compatible_leg(tmp_path):
    aie_model = _run_pipeline(_fanout_dense_model(), tmp_path, project='aie1_fanout')
    plan = aie_model.context.ir.physical.plan
    edges = {(edge['source'], edge['target']) for edge in plan['direct_edges']}

    assert plan['buffers'] == []
    assert ('root_aie.out1[0]', 'left_aie.in1[0]') in edges
    assert ('root_aie.out1[0]', 'right_aie.in1[0]') in edges
    assert sum(edge['source'] == 'root_aie.out1[0]' for edge in plan['direct_edges']) == 2


def test_aie1_staging_mismatch_requires_an_explicit_relayout(tmp_path):
    with pytest.raises(ConfigRefused, match=r'no supported microtiling accepts producer output microtile'):
        _run_pipeline(
            _fanout_dense_model(),
            tmp_path,
            directives={
                'root': {'microtiling': {'microtile_m': 4, 'microtile_k': 8, 'microtile_n': 4}},
            },
            project='aie1_mismatch',
        )


def test_memtile_device_keeps_default_boundaries_and_publishes_io_ports(tmp_path):
    aie_model = _run_pipeline(
        _dense_model(),
        tmp_path,
        part='xilinx_vek280_base_202610_1',
        project='aieml_dense',
    )
    plan = aie_model.context.ir.physical.plan

    assert len(plan['buffers']) == 2
    assert [(port['direction'], port['port']) for port in plan['io_ports']] == [('input', 0), ('output', 0)]


def test_direct_padded_output_uses_dma_projection(tmp_path):
    aie_model = _run_pipeline(_dense_model(), tmp_path, project='aie1_padded_output')
    plan = aie_model.context.ir.physical.plan

    assert plan['buffers'] == []
    assert [item['endpoint'] for item in plan['kernel_read_accesses']] == ['dense_aie.kk[0].out[0]']
    descriptor = plan['kernel_read_accesses'][0]['descriptor']
    assert descriptor['storage_layout'] == 'microtiled'
    assert descriptor['buffer_dimension'] == [16, 2, 4]
    assert descriptor['tiling_dimension'] == [8, 1, 1]
    assert descriptor['tile_traversal'] == [
        {'dimension': 1, 'stride': 1, 'wrap': 1},
        {'dimension': 0, 'stride': 8, 'wrap': 2},
        {'dimension': 2, 'stride': 1, 'wrap': 4},
    ]


def _small_batch_dense_model(rows):
    """int16 rows against distinct int8 weights, so a row from the wrong inference changes every output."""
    nodes = [
        helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'),
        helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w'], name='w_dq'),
        helper.make_node('MatMul', ['x', 'w'], ['mm'], name='dense'),
        helper.make_node('QuantizeLinear', ['mm', 'y_scale', 'y_zp'], ['y_q'], name='y_q'),
        helper.make_node('DequantizeLinear', ['y_q', 'y_scale', 'y_zp'], ['y'], name='y_dq'),
    ]
    weights = np.random.default_rng(1).integers(-4, 5, size=(32, 16)).astype(np.int8)
    return make_model(
        'aie1_small_batch',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT16, [rows, 32])],
        outputs=[('y', TensorProto.FLOAT, [rows, 16])],
        initializers=[
            *_qparams('x', TensorProto.INT16),
            *_qparams('w', TensorProto.INT8),
            *_qparams('y', TensorProto.INT8),
            numpy_helper.from_array(weights, 'w_q'),
        ],
        opset=21,  # int16 QDQ
    )


def test_a_batch_below_the_padded_tile_moves_only_its_rows(tmp_path):
    """AIE1 pads 4 int16 rows to a tile of 8 and has no memory tile to fill the rest, so each boundary DMA
    walks only the rows the host moves; walking the whole tile, the input would wait for rows never sent."""
    aie_model = from_onnx(
        _small_batch_dense_model(4),
        {'Part': AIE1_PART, 'AIEConfig': {'BatchSize': 4, 'Iterations': 2}},
        output_dir=tmp_path,
        project_name='small_batch',
    )
    aie_model.run_pipeline()
    for io in aie_model.context.ir.physical.plan['io_ports']:
        walks = {step['dimension']: step['wrap'] for step in io['descriptor']['tile_traversal']}
        assert io['staging']['tiling_dimension'][1] == 4 and walks[2] == 1, io['direction']


@pytest.mark.requires_vitis
def test_a_batch_below_the_padded_tile_matches_onnx(tmp_path):
    from helpers import assert_x86_matches_onnx

    feeds = {'x_q': np.random.default_rng(3).integers(-200, 200, size=(2, 4, 32)).astype(np.int16)}
    assert_x86_matches_onnx(
        _small_batch_dense_model(4),
        feeds,
        {},
        tmp_path,
        batch=4,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=2,
        per_iteration=True,
    )
