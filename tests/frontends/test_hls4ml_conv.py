"""hls4ml/QKeras frontend: a QConv2D stack lowered to the canonical conv2d contract.

The point of this test is the frontend boundary: hls4ml is channels-last, so it reaches the same
IR as the ONNX path without any op_impls change -- only the attribute names and, for a depthwise
layer, the weight arrangement differ.
"""

import numpy as np
import pytest

keras = pytest.importorskip('keras')

PART = 'xcvp2802-vsva5601-2MHP-e-S'
H, W, CIN, COUT, CLASSES, BITS = 8, 8, 8, 8, 8, 8


@pytest.fixture
def lowered(tmp_path):
    hls4ml = pytest.importorskip('hls4ml')
    pytest.importorskip('qkeras')
    from keras.models import Sequential
    from qkeras import QActivation, QConv2D, QDense, QDepthwiseConv2D, quantized_bits, quantized_relu

    keras.utils.set_random_seed(7)
    q_w = quantized_bits(BITS, 2, alpha=1)
    model = Sequential(
        [
            keras.Input(shape=(H, W, CIN)),
            QConv2D(COUT, (3, 3), padding='same', kernel_quantizer=q_w, bias_quantizer=q_w, name='conv'),
            QActivation(quantized_relu(BITS, 2), name='relu'),
            QDepthwiseConv2D((3, 3), padding='same', depthwise_quantizer=q_w, bias_quantizer=q_w, name='dw'),
            QActivation(quantized_relu(BITS, 2), name='dwrelu'),
            keras.layers.Flatten(name='flatten'),
            QDense(CLASSES, kernel_quantizer=q_w, bias_quantizer=q_w, name='fc'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = f'ap_fixed<{BITS},3>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend='AIE',
        io_type='io_parallel',
        output_dir=str(tmp_path / 'proj'),
        part=PART,
        hls_config=config,
        project_name='proj',
        batch_size=1,  # conv2d.b.r.v1 runs one sample per call
    )
    hls_model.write()
    from aie4ml.ir import get_backend_context

    return get_backend_context(hls_model)


def test_hls4ml_conv_reaches_the_canonical_contract(lowered):
    conv = next(node for node in lowered.ir.logical if node.op_type == 'conv2d')
    activation, weight = conv.inputs[0], conv.inputs[1]
    assert len(activation.shape) == 4 and tuple(activation.shape)[1:] == (H, W, CIN)  # NHWC
    assert tuple(weight.shape) == (3, 3, CIN, COUT)  # [kh, kw, Cin/groups, Cout]
    assert conv.metadata['kernel_shape'] == (3, 3)
    assert conv.metadata['strides'] == (1, 1) and conv.metadata['dilations'] == (1, 1)
    assert conv.metadata['pads'] == (1, 1, 1, 1) and conv.metadata['groups'] == 1
    assert conv.roles[weight.name] == 'rhs' and 'bias' in conv.roles.values()

    inst = lowered.ir.execution.get(conv.name)
    assert inst.variant.variant_id == 'conv2d.b.r.v1'
    assert 'fused_activation' in conv.traits  # the QActivation folded in


def test_hls4ml_depthwise_reaches_the_compact_group_contract(lowered):
    """Keras keeps a filter per input channel; the canonical form is one group per channel."""
    dw = next(node for node in lowered.ir.logical if node.metadata.get('groups', 1) > 1)
    assert tuple(dw.inputs[1].shape) == (3, 3, 1, COUT)  # [kh, kw, Cin/groups, Cout]
    assert dw.metadata['groups'] == COUT
    tiles = lowered.ir.execution.get(dw.name).artifacts['packed_weights']
    blocks = COUT // 8
    padded = blocks if blocks == 1 else blocks + blocks % 2  # pairs, except a lone block
    grid = tiles.reshape(9, blocks, padded, 8, 8)
    assert np.count_nonzero(grid[0, 0, 0]) == np.count_nonzero(np.diag(grid[0, 0, 0]))


def test_layers_without_bias_lower_with_only_their_operands(tmp_path):
    """hls4ml gives each layer without bias a zero bias weight for its own templates; aie4ml lowers none."""
    hls4ml = pytest.importorskip('hls4ml')
    pytest.importorskip('qkeras')
    import aie4ml
    from keras.models import Sequential
    from qkeras import QActivation, QConv2D, QDense, QDepthwiseConv2D, quantized_bits, quantized_relu

    q_w = quantized_bits(BITS, 2, alpha=1)
    model = Sequential(
        [
            keras.Input(shape=(H, W, CIN)),
            QConv2D(COUT, (3, 3), padding='same', kernel_quantizer=q_w, use_bias=False, name='conv'),
            QActivation(quantized_relu(BITS, 2), name='relu'),
            QDepthwiseConv2D((3, 3), padding='same', depthwise_quantizer=q_w, use_bias=False, name='dw'),
            QActivation(quantized_relu(BITS, 2), name='dwrelu'),
            keras.layers.Flatten(name='flatten'),
            QDense(CLASSES, kernel_quantizer=q_w, use_bias=False, name='fc'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = f'ap_fixed<{BITS},3>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend='AIE',
        io_type='io_parallel',
        output_dir=str(tmp_path / 'proj'),
        part=PART,
        hls_config=config,
        project_name='proj',
        batch_size=1,
    )
    weighted = [n for n in aie4ml.from_hls4ml(hls_model).context.ir.logical if n.op_type in ('conv2d', 'dense')]
    assert [n.op_type for n in weighted] == ['conv2d', 'conv2d', 'dense']
    for node in weighted:
        assert sorted(node.roles.values()) == ['lhs', 'rhs'] and len(node.inputs) == 2, node.name


def _batchnorm_model(tmp_path, part):
    """QConv2DBatchnorm with trained-looking statistics: hls4ml folds them into its quantized weights and bias."""
    hls4ml = pytest.importorskip('hls4ml')
    qkeras = pytest.importorskip('qkeras')
    from keras.models import Sequential

    keras.utils.set_random_seed(7)
    q_w = qkeras.quantized_bits(BITS, 2, alpha=1)
    conv = qkeras.QConv2DBatchnorm(COUT, (3, 3), padding='same', kernel_quantizer=q_w, bias_quantizer=q_w, name='conv')
    model = Sequential([keras.Input(shape=(H, W, CIN)), conv, qkeras.QActivation(qkeras.quantized_relu(BITS, 2))])
    rng = np.random.default_rng(3)
    stats = {'gamma': (0.5, 1.5), 'beta': (-0.5, 0.5), 'moving_mean': (-0.5, 0.5), 'moving_variance': (0.5, 2)}
    for weight in conv.weights:
        if (name := weight.path.rsplit('/', 1)[1]) in stats:
            weight.assign(rng.uniform(*stats[name], COUT).astype(np.float32))
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = f'ap_fixed<{BITS},3>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend='AIE',
        io_type='io_parallel',
        output_dir=str(tmp_path / 'proj'),
        part=part,
        hls_config=config,
        project_name='proj',
        batch_size=1,
        iterations=2,
    )
    return model, hls_model


def test_hls4ml_folded_batchnorm_lowers_to_a_conv(tmp_path):
    import aie4ml

    _, hls_model = _batchnorm_model(tmp_path, PART)
    (conv,) = [n for n in aie4ml.from_hls4ml(hls_model).context.ir.logical if n.op_type == 'conv2d']
    folded = next(layer for layer in hls_model.get_layers() if layer.class_name == 'Conv2DBatchnorm')
    np.testing.assert_array_equal(conv.inputs[1].data, folded.weights['weight'].data)


def test_hls4ml_refuses_a_batchnorm_it_did_not_fold(tmp_path):
    """After a quantized conv hls4ml keeps the batchnorm: folding it would change the quantized weights."""
    hls4ml = pytest.importorskip('hls4ml')
    qkeras = pytest.importorskip('qkeras')
    from keras.models import Sequential

    q_w = qkeras.quantized_bits(BITS, 2, alpha=1)
    model = Sequential(
        [
            keras.Input(shape=(H, W, CIN)),
            qkeras.QConv2D(COUT, (3, 3), padding='same', kernel_quantizer=q_w, bias_quantizer=q_w, name='conv'),
            keras.layers.BatchNormalization(name='bn'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = f'ap_fixed<{BITS},3>'
    with pytest.raises(NotImplementedError, match='did not fold'):
        hls4ml.converters.convert_from_keras_model(
            model, backend='AIE', output_dir=str(tmp_path / 'proj'), part=PART, hls_config=config, project_name='proj'
        )


@pytest.mark.requires_vitis
def test_hls4ml_folded_batchnorm_matches_qkeras(tmp_path):
    model, hls_model = _batchnorm_model(tmp_path, 'xcve2802-vsvh1760-2mp-e-s')
    hls_model.compile()
    x = np.random.default_rng(5).integers(-128, 128, size=(2, H, W, CIN)).astype(np.float32) / 32
    want = model.predict(x, verbose=0)
    got = hls_model.predict(x.reshape(2, 1, H, W, CIN), simulator='x86')
    np.testing.assert_equal(np.asarray(got).reshape(want.shape), want)


PARTS = {
    'aie1': 'xcvp2802-vsva5601-2MHP-e-S',
    'aie-ml': 'xcve2802-vsvh1760-2mp-e-s',
    'aie-mlv2': 'xc2ve3858-ssva2112-2mp-e-s',
}


def _pooled_model(tmp_path, part, relu_after_pool=False):
    """QConv2D -> quantized ReLU -> MaxPooling2D -> Flatten -> QDense, the ReLU on either side of the pool."""
    hls4ml = pytest.importorskip('hls4ml')
    pytest.importorskip('qkeras')
    from keras.models import Sequential
    from qkeras import QActivation, QConv2D, QDense, quantized_bits, quantized_relu

    keras.utils.set_random_seed(7)
    q_w = quantized_bits(BITS, 2, alpha=1)
    relu = QActivation(quantized_relu(BITS, 2), name='relu')
    pool = keras.layers.MaxPooling2D((2, 2), name='pool')
    model = Sequential(
        [
            keras.Input(shape=(H, W, CIN)),
            QConv2D(16, (3, 3), padding='same', kernel_quantizer=q_w, bias_quantizer=q_w, name='conv'),
            *([pool, relu] if relu_after_pool else [relu, pool]),
            keras.layers.Flatten(name='flatten'),
            QDense(CLASSES, kernel_quantizer=q_w, bias_quantizer=q_w, name='fc'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = f'ap_fixed<{BITS},3>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend='AIE',
        io_type='io_parallel',
        output_dir=str(tmp_path / 'proj'),
        part=part,
        hls_config=config,
        project_name='proj',
        batch_size=1,
        iterations=2,
    )
    return model, hls_model


@pytest.mark.parametrize('relu_after_pool', [False, True], ids=['relu-pool', 'pool-relu'])
def test_hls4ml_max_pool_fuses_into_the_conv(tmp_path, relu_after_pool):
    """The same canonical pool2d the ONNX MaxPool lowers to, so the conv fuses it and the ReLU either side."""
    import aie4ml

    _, hls_model = _pooled_model(tmp_path, PART, relu_after_pool)
    logical = aie4ml.from_hls4ml(hls_model).context.ir.logical
    (conv,) = [n for n in logical if n.op_type == 'conv2d']
    assert not [n for n in logical if n.op_type == 'pool2d']
    assert {'fused_pool', 'fused_activation', 'output_view'} <= set(conv.traits)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
def test_hls4ml_max_pool_matches_qkeras(tmp_path, part):
    model, hls_model = _pooled_model(tmp_path, part)
    hls_model.compile()
    x = np.random.default_rng(5).integers(-128, 128, size=(2, H, W, CIN)).astype(np.float32) / 32  # input grid
    want = model.predict(x, verbose=0)  # two distinct inferences: the output is refilled every call
    got = hls_model.predict(x.reshape(2, 1, H, W, CIN), simulator='x86')  # (iterations, batch, ...)
    np.testing.assert_equal(np.asarray(got).reshape(want.shape), want)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [PARTS['aie-ml'], PARTS['aie-mlv2']], ids=['aie-ml', 'aie-mlv2'])
def test_hls4ml_int16_conv_matches_qkeras(tmp_path, part):
    """16-bit activations against 8-bit weights: QConv2D -> 16-bit ReLU -> MaxPooling2D -> Flatten -> QDense."""
    hls4ml = pytest.importorskip('hls4ml')
    qkeras = pytest.importorskip('qkeras')
    from keras.models import Sequential

    keras.utils.set_random_seed(7)
    q_w = qkeras.quantized_bits(BITS, 2, alpha=1)
    model = Sequential(
        [
            keras.Input(shape=(H, W, CIN)),
            qkeras.QConv2D(16, (3, 3), padding='same', kernel_quantizer=q_w, bias_quantizer=q_w, name='conv'),
            qkeras.QActivation(qkeras.quantized_relu(16, 6), name='relu'),
            keras.layers.MaxPooling2D((2, 2), name='pool'),
            keras.layers.Flatten(name='flatten'),
            qkeras.QDense(CLASSES, kernel_quantizer=q_w, bias_quantizer=q_w, name='fc'),
            qkeras.QActivation(qkeras.quantized_relu(BITS, 3), name='out'),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend='AIE',
        io_type='io_parallel',
        output_dir=str(tmp_path / 'proj'),
        part=part,
        hls_config=config,
        project_name='proj',
        batch_size=1,
        iterations=2,
    )
    hls_model.compile()
    x = np.random.default_rng(5).integers(-2048, 2048, size=(2, H, W, CIN)).astype(np.float32) / 1024
    want = model.predict(x, verbose=0)
    got = hls_model.predict(x.reshape(2, 1, H, W, CIN), simulator='x86')
    np.testing.assert_equal(np.asarray(got).reshape(want.shape), want)
