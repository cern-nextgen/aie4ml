"""hls4ml/QKeras frontend: a QDense stack converted through the AIE backend."""

import os
import random
from pathlib import Path

import numpy as np
import pytest
from aie4ml.ir import get_backend_context

keras = pytest.importorskip('keras')

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)


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


def _build_qkeras_mlp(qkeras, input_shape, in_features, hidden1, hidden2, out_features, bits):
    from keras.models import Sequential
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    assert int(input_shape[-1]) == int(in_features)

    INT_BITS = 3

    q_in = quantized_bits(bits, INT_BITS)
    q_w = quantized_bits(bits, 2, alpha=1)
    q_b = quantized_bits(bits, 2, alpha=1)

    model = Sequential(
        [
            keras.Input(shape=input_shape),
            QActivation(q_in, name='input_quant'),
            QDense(hidden1, name='dense0', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'),
            QActivation(quantized_relu(bits, 0), name='act0'),
            QDense(hidden2, name='dense1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'),
            QActivation(quantized_relu(bits, 0), name='act1'),
            QDense(
                out_features, name='dense2', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer='random_uniform'
            ),
        ]
    )
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_cfg(hls4ml, model, layer_parallelism, io_route=None):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    cfg.setdefault('LayerName', {})

    for lname, params in layer_parallelism.items():
        cfg['LayerName'].setdefault(lname, {})
        cfg['LayerName'][lname]['cas_num'] = int(params['cas_num'])
        cfg['LayerName'][lname]['cas_length'] = int(params['cas_length'])
    for lname, route in (io_route or {}).items():
        cfg['LayerName'][lname]['io_route'] = route

    return cfg


def _make_aie_model(tmp_path, cfg_dict, input_shape):
    np, hls4ml, _keras, qkeras = _imports()

    qmodel = _build_qkeras_mlp(
        qkeras,
        input_shape=input_shape,
        in_features=384,
        hidden1=256,
        hidden2=200,
        out_features=10,
        bits=cfg_dict['bits'],
    )

    cfg = _make_cfg(hls4ml, qmodel, cfg_dict['layers'], cfg_dict.get('io_route'))

    tag = '1d' if len(input_shape) == 1 else f'nd{len(input_shape)}'
    outdir = tmp_path / (f"aie_mlp_b{cfg_dict['bits']}_bs{cfg_dict['batch']}_{_par_summary(cfg_dict['layers'])}_{tag}")

    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(outdir),
        backend='aie',
        project_name='proj_aie',
        batch_size=cfg_dict['batch'],
        iterations=5,
    )

    return qmodel, aie_model


CFG_LIST = [
    {
        'bits': 8,
        'batch': 1,
        'layers': {
            'dense0': {'cas_num': 2, 'cas_length': 4},
            'dense1': {'cas_num': 1, 'cas_length': 4},
            'dense2': {'cas_num': 1, 'cas_length': 1},
        },
        # through a memory tile that zero-fills each reader's padded rows: both BD pools exactly full
        'io_route': {'dense0': {'inputs': {'layer2_out': 'memtile'}}},
    },
    {
        'bits': 8,
        'batch': 10,  # 10 of 16 rows: each memory tile between the layers zero-fills its readers' padded rows
        'layers': {
            'dense0': {'cas_num': 1, 'cas_length': 8},
            'dense1': {'cas_num': 2, 'cas_length': 2},
            'dense2': {'cas_num': 1, 'cas_length': 2},
        },
    },
]


@pytest.mark.aie_ir
@pytest.mark.parametrize('cfg', CFG_LIST)
def test_aie_backend_conversion_only(tmp_path, cfg):
    """
    Conversion + lowering test.
    No Vitis required.
    """
    qmodel, aie_model = _make_aie_model(tmp_path, cfg, input_shape=(384,))
    assert aie_model is not None


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('cfg', CFG_LIST)
def test_aie_compile_x86_sim(tmp_path: Path, cfg: dict):
    """
    AIE backend + Vitis compile + x86 functional simulation.
    """
    _require_vitis()

    np, _hls4ml, _keras, _qkeras = _imports()
    qmodel, aie_model = _make_aie_model(tmp_path, cfg, input_shape=(384,))

    aie_model.compile()

    rng = np.random.default_rng(SEED)
    x = (rng.random((cfg['batch'], 384), dtype=np.float32) * 2.0) - 1.0

    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')

    y_aie = y_aie[: cfg['batch']]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_equal(y_ref, y_aie)


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('input_shape', [(16, 384), (8, 16, 384)])
def test_aie_compile_x86_sim_nd_input(tmp_path: Path, input_shape):
    _require_vitis()
    np, hls4ml, keras, qkeras = _imports()

    from keras.models import Sequential
    from qkeras import QActivation, QDense, quantized_bits

    bits = 8
    B = 1
    N = 10

    q_w = quantized_bits(bits, 2, alpha=1)
    q_b = quantized_bits(bits, 2, alpha=1)

    qmodel = Sequential(
        [
            keras.Input(shape=input_shape),
            QActivation(quantized_bits(bits, 3), name='input_quant'),
            QDense(
                N,
                name='dense0',
                kernel_quantizer=q_w,
                bias_quantizer=q_b,
                bias_initializer='random_uniform',
            ),
        ]
    )
    qmodel.compile(optimizer='adam', loss='mse')

    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')

    shape_tag = 'x'.join(str(d) for d in input_shape)
    outdir = tmp_path / f'aie_dense_nd_single_layer_{shape_tag}'
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(outdir),
        backend='aie',
        project_name='proj_aie',
        batch_size=B,
        iterations=5,
    )

    aie_model.compile()

    x = (np.random.random((B, *input_shape)).astype('float32') * 2.0) - 1.0
    y_ref = qmodel.predict(x, verbose=0)
    y_aie = aie_model.predict(x, simulator='x86')
    y_aie = y_aie[:B]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_equal(y_ref, y_aie)


# Future TODO
#
# @pytest.mark.aie_ir
# @pytest.mark.requires_vitis
# @pytest.mark.slow
# def test_aie_build_and_hw_sim(...):
#     aie_model.build()
#


@pytest.mark.aie_ir
def test_a_permute_between_dense_layers_converts(tmp_path):
    """hls4ml leaves a Keras Permute's data_format unset: its permutation is over the batchless axes, the
    channels-last order the AIE folds a transpose in."""
    _np, hls4ml, keras, qkeras = _imports()
    q8 = qkeras.quantized_bits(8, 0, alpha=1)
    model = keras.Sequential(
        [
            keras.layers.Input((16, 32)),
            qkeras.QActivation(qkeras.quantized_bits(8, 2), name='in_q'),
            qkeras.QDense(16, kernel_quantizer=q8, bias_quantizer=q8, name='fc1'),
            qkeras.QActivation(qkeras.quantized_relu(8), name='relu1'),
            keras.layers.Permute((2, 1), name='permute'),
            qkeras.QDense(8, kernel_quantizer=q8, bias_quantizer=q8, name='fc2'),
        ]
    )
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    aie_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=str(tmp_path), project_name='permute', backend='aie', batch_size=1
    )
    fc2 = get_backend_context(aie_model).ir.execution.get('fc2_aie')
    assert [view.perm for view in fc2.port_views.values() if view.perm is not None] == [(0, 2, 1)]


@pytest.mark.aie_ir
def test_a_conversion_lowers_one_sample_unless_asked_for_more(tmp_path):
    """Keras tensors carry no batch axis: the conversion's batch_size (default 1) is the leading axis the AIE sees."""
    _np, hls4ml, keras, qkeras = _imports()
    q8 = qkeras.quantized_bits(8, 0, alpha=1)
    model = keras.Sequential(
        [
            keras.layers.Input((32,)),
            qkeras.QActivation(qkeras.quantized_bits(8, 2), name='in_q'),
            qkeras.QDense(16, kernel_quantizer=q8, bias_quantizer=q8, name='fc'),
        ]
    )
    cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')
    for batch, asked in ((1, {}), (4, {'batch_size': 4})):
        aie_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=cfg, output_dir=str(tmp_path / str(batch)), project_name='b', backend='aie', **asked
        )
        logical = get_backend_context(aie_model).ir.logical
        assert logical.tensors[logical.input_tensor_names[0]].shape == (batch, 32)
