"""hls4ml/QKeras frontend: one producer feeding several consumers (fan-out routing)."""

import os
from pathlib import Path

import pytest

pytest.importorskip('keras')  # at collection, so the conftest seeds it for every test

BATCH = 1
ITERATIONS = 5


def _require_vitis():
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')


def _imports():
    np = pytest.importorskip('numpy')
    hls4ml = pytest.importorskip('hls4ml')
    keras = pytest.importorskip('keras')
    qkeras = pytest.importorskip('qkeras')
    return np, hls4ml, keras, qkeras


def _par_summary(layers):
    return '_'.join(f"{k}_c{v['cas_num']}x{v['cas_length']}" for k, v in layers.items())


def _build_qkeras_fanout(qkeras, input_shape, bits_in, bits_w, hidden, mode):
    from keras.layers import Input
    from keras.models import Model
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    q_in = quantized_bits(bits_in, 2)
    q_w = quantized_bits(bits_w, 2, alpha=1)
    q_b = quantized_bits(bits_w, 2, alpha=1)

    x0 = Input(shape=input_shape, name='input_layer')
    x = QActivation(q_in, name='input_quant')(x0)
    if mode == 'io':
        b1 = QDense(
            hidden, name='qfc1_branch1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(x)
        b2 = QDense(
            hidden, name='qfc1_branch2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(x)

    elif mode == 'internal':
        s = QDense(
            hidden,
            name='shared_internal_dense',
            kernel_quantizer=q_w,
            bias_quantizer=q_b,
            bias_initializer='random_uniform',
        )(x)
        s = QActivation(quantized_relu(bits_in, 0), name='shared_internal_quant')(s)
        b1 = QDense(
            hidden, name='qfc2_branch1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(s)
        b2 = QDense(
            hidden, name='qfc2_branch2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(s)
    else:
        raise ValueError(f'Unknown fanout mode "{mode}"')

    y1 = QActivation(quantized_relu(bits_in, 6), name='output_quant_1')(b1)
    y2 = QActivation(quantized_relu(bits_in, 6), name='output_quant_2')(b2)

    model = Model(inputs=x0, outputs=[y1, y2])
    model.compile(optimizer='adam', loss='mse')
    return model


def _build_qkeras_fanout3(qkeras, input_shape, bits_in, bits_w, hidden, mode):
    from keras.layers import Input
    from keras.models import Model
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    q_in = quantized_bits(bits_in, 2)
    q_w = quantized_bits(bits_w, 2, alpha=1)
    q_b = quantized_bits(bits_w, 2, alpha=1)

    x0 = Input(shape=input_shape, name='input_layer')
    x = QActivation(q_in, name='input_quant')(x0)
    if mode == 'io':
        b1 = QDense(
            hidden, name='qfc1_branch1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(x)
        b2 = QDense(
            hidden, name='qfc1_branch2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(x)
        b3 = QDense(
            hidden, name='qfc1_branch3', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(x)
    elif mode == 'internal':
        s = QDense(
            hidden,
            name='shared_internal_dense',
            kernel_quantizer=q_w,
            bias_quantizer=q_b,
            bias_initializer='random_uniform',
        )(x)
        s = QActivation(quantized_relu(bits_in, 0), name='shared_internal_quant')(s)
        b1 = QDense(
            hidden, name='qfc2_branch1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(s)
        b2 = QDense(
            hidden, name='qfc2_branch2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(s)
        b3 = QDense(
            hidden, name='qfc2_branch3', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
        )(s)
    else:
        raise ValueError(f'Unknown fanout mode "{mode}"')

    y1 = QActivation(quantized_relu(bits_in, 6), name='output_quant_1')(b1)
    y2 = QActivation(quantized_relu(bits_in, 6), name='output_quant_2')(b2)
    y3 = QActivation(quantized_relu(bits_in, 6), name='output_quant_3')(b3)

    model = Model(inputs=x0, outputs=[y1, y2, y3])
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_cfg(hls4ml, model, layer_parallelism):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg.setdefault('LayerName', {})
    for lname, params in layer_parallelism.items():
        cfg['LayerName'].setdefault(lname, {})
        cfg['LayerName'][lname]['cas_num'] = int(params['cas_num'])
        cfg['LayerName'][lname]['cas_length'] = int(params['cas_length'])
    return cfg


def _match_output(aie_out, idx):
    if isinstance(aie_out, dict):
        keys = list(aie_out.keys())
        if idx < len(keys):
            return aie_out[keys[idx]]
        raise ValueError(f'Missing AIE output index {idx} in {keys}.')
    return aie_out[idx]


def _make_aie_model(tmp_path, input_shape, cfg_layers, bits_in, bits_w, hidden, mode):
    _np, hls4ml, _keras, qkeras = _imports()
    qmodel = _build_qkeras_fanout(
        qkeras, input_shape=input_shape, bits_in=bits_in, bits_w=bits_w, hidden=hidden, mode=mode
    )
    cfg = _make_cfg(hls4ml, qmodel, cfg_layers)
    tag = '1d' if len(input_shape) == 1 else f'nd{len(input_shape)}'
    outdir = tmp_path / f'aie_fanout_{mode}_{tag}_h{hidden}_{_par_summary(cfg_layers)}'
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(outdir),
        backend='aie',
        project_name='proj_aie',
        batch_size=BATCH,
        iterations=ITERATIONS,
    )
    return qmodel, aie_model


def _make_aie_model_fanout3(tmp_path, input_shape, cfg_layers, bits_in, bits_w, hidden, mode):
    _np, hls4ml, _keras, qkeras = _imports()
    qmodel = _build_qkeras_fanout3(
        qkeras, input_shape=input_shape, bits_in=bits_in, bits_w=bits_w, hidden=hidden, mode=mode
    )
    cfg = _make_cfg(hls4ml, qmodel, cfg_layers)
    tag = '1d' if len(input_shape) == 1 else f'nd{len(input_shape)}'
    outdir = tmp_path / f'aie_fanout3_{mode}_{tag}_h{hidden}_{_par_summary(cfg_layers)}'
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(outdir),
        backend='aie',
        project_name='proj_aie',
        batch_size=BATCH,
        iterations=ITERATIONS,
    )
    return qmodel, aie_model


FANOUT_CASES = [
    {
        'name': 'io_int8',
        'mode': 'io',
        'hidden': 256,
        'cfg_layers': {
            'qfc1_branch1': {'cas_num': 2, 'cas_length': 4},
            'qfc1_branch2': {'cas_num': 2, 'cas_length': 4},
        },
        'nd_cases': [((384,), 8, 8)],
    },
    {
        'name': 'internal_int8',
        'mode': 'internal',
        'hidden': 256,
        'cfg_layers': {
            'shared_internal_dense': {'cas_num': 2, 'cas_length': 4},
            'qfc2_branch1': {'cas_num': 2, 'cas_length': 4},
            'qfc2_branch2': {'cas_num': 2, 'cas_length': 4},
        },
        'nd_cases': [((384,), 8, 8)],
    },
    {
        'name': 'io_int16_small',
        'mode': 'io',
        'hidden': 16,
        'cfg_layers': {
            'qfc1_branch1': {'cas_num': 1, 'cas_length': 2},
            'qfc1_branch2': {'cas_num': 1, 'cas_length': 2},
        },
        'nd_cases': [((8, 64), 16, 16)],
    },
    {
        'name': 'internal_int16_small',
        'mode': 'internal',
        'hidden': 16,
        'cfg_layers': {
            'shared_internal_dense': {'cas_num': 1, 'cas_length': 2},
            'qfc2_branch1': {'cas_num': 2, 'cas_length': 1},
            'qfc2_branch2': {'cas_num': 2, 'cas_length': 1},
        },
        'nd_cases': [((80, 32), 16, 16)],
    },
]

CONV_CASES = [(fc['name'], nd_case) for fc in FANOUT_CASES for nd_case in fc['nd_cases']]
SIM_CASES = CONV_CASES[: min(5, len(CONV_CASES))]


@pytest.mark.aie_ir
@pytest.mark.parametrize('case_name,nd_case', CONV_CASES)
def test_aie_fanout_conversion_only(tmp_path: Path, case_name, nd_case):
    fc = next(c for c in FANOUT_CASES if c['name'] == case_name)
    input_shape, bits_in, bits_w = nd_case
    qmodel, aie_model = _make_aie_model(
        tmp_path, input_shape, fc['cfg_layers'], bits_in=bits_in, bits_w=bits_w, hidden=fc['hidden'], mode=fc['mode']
    )
    assert qmodel is not None
    assert aie_model is not None


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('case_name,nd_case', SIM_CASES)
def test_aie_fanout_compile_x86_sim(tmp_path: Path, case_name, nd_case):
    _require_vitis()
    np, _hls4ml, _keras, _qkeras = _imports()
    fc = next(c for c in FANOUT_CASES if c['name'] == case_name)
    input_shape, bits_in, bits_w = nd_case

    qmodel, aie_model = _make_aie_model(
        tmp_path, input_shape, fc['cfg_layers'], bits_in=bits_in, bits_w=bits_w, hidden=fc['hidden'], mode=fc['mode']
    )
    aie_model.compile()

    batch = BATCH
    x = (np.random.random((batch, *input_shape)).astype('float32') * 2.0) - 1.0
    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')

    y_ref_0 = y_ref[0]
    y_ref_1 = y_ref[1]
    y_aie_0 = _match_output(y_aie, 0)[:batch]
    y_aie_1 = _match_output(y_aie, 1)[:batch]

    assert y_ref_0.shape == y_aie_0.shape
    assert y_ref_1.shape == y_aie_1.shape

    if len(input_shape) == 1:
        np.testing.assert_equal(y_ref_0, y_aie_0)
        np.testing.assert_equal(y_ref_1, y_aie_1)


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    ('case_name', 'input_shape', 'bits_in', 'bits_w', 'hidden', 'cfg_layers'),
    [
        (
            'io_3way',
            (384,),
            8,
            8,
            256,
            {
                'qfc1_branch1': {'cas_num': 2, 'cas_length': 4},
                'qfc1_branch2': {'cas_num': 2, 'cas_length': 4},
                'qfc1_branch3': {'cas_num': 2, 'cas_length': 4},
            },
        ),
        (
            'internal_3way',
            (384,),
            8,
            8,
            256,
            {
                'shared_internal_dense': {'cas_num': 2, 'cas_length': 4},
                'qfc2_branch1': {'cas_num': 2, 'cas_length': 4},
                'qfc2_branch2': {'cas_num': 2, 'cas_length': 4},
                'qfc2_branch3': {'cas_num': 2, 'cas_length': 4},
            },
        ),
    ],
)
def test_aie_fanout3_compile_x86_sim(tmp_path: Path, case_name, input_shape, bits_in, bits_w, hidden, cfg_layers):
    _require_vitis()
    np, _hls4ml, _keras, _qkeras = _imports()

    mode = 'io' if case_name == 'io_3way' else 'internal'
    qmodel, aie_model = _make_aie_model_fanout3(
        tmp_path, input_shape, cfg_layers, bits_in=bits_in, bits_w=bits_w, hidden=hidden, mode=mode
    )
    aie_model.compile()

    x = (np.random.random((BATCH, *input_shape)).astype('float32') * 2.0) - 1.0
    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')

    for idx in range(3):
        y_ref_i = y_ref[idx]
        y_aie_i = _match_output(y_aie, idx)[:BATCH]
        assert y_ref_i.shape == y_aie_i.shape
        np.testing.assert_equal(y_ref_i, y_aie_i)


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_aie_fanout3_compile_x86_sim_nd_input(tmp_path: Path):
    _require_vitis()
    np, _hls4ml, _keras, _qkeras = _imports()

    input_shape = (8, 16, 384)
    cfg_layers = {
        'qfc1_branch1': {'cas_num': 2, 'cas_length': 4},
        'qfc1_branch2': {'cas_num': 2, 'cas_length': 4},
        'qfc1_branch3': {'cas_num': 2, 'cas_length': 4},
    }
    qmodel, aie_model = _make_aie_model_fanout3(
        tmp_path, input_shape, cfg_layers, bits_in=8, bits_w=8, hidden=256, mode='io'
    )
    aie_model.compile()

    x = (np.random.random((BATCH, *input_shape)).astype('float32') * 2.0) - 1.0
    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')

    for idx in range(3):
        y_ref_i = y_ref[idx]
        y_aie_i = _match_output(y_aie, idx)[:BATCH]
        assert y_ref_i.shape == y_aie_i.shape
        np.testing.assert_equal(y_ref_i, y_aie_i)
