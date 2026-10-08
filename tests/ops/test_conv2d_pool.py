"""A 2x2 max pool of stride 2 fused into the Conv2D epilogue: the conv writes the pooled tensor."""

from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls import get_family_resolver_registry
from helpers import TensorProto, assert_x86_matches_onnx, helper, lower, make_model, qdq, qparams
from ops.test_conv2d import CIN, CLASSES, FRAC, _conv, _head, _start

PARTS = {
    'aie1': 'xcvp2802-vsva5601-2MHP-e-S',
    'aie-ml': 'xcve2802-vsvh1760-2mp-e-s',
    'aie-mlv2': 'xc2ve3858-ssva2112-2mp-e-s',
}
H = W = 8
ROW_SPLIT = {'c1': {'parallelism': {'contract': 'outer', 'cas_num': 2}}}


def _pool(nodes, inits, src, out, *, kernel=2, stride=2, pads=0, ceil_mode=0, unsigned=False):
    """MaxPool over the dequantized NCHW value, requantized with its input's scale."""
    inits += qparams(f'{out}o', frac=FRAC, unsigned=unsigned)
    nodes.append(
        helper.make_node(
            'MaxPool',
            [src],
            [f'{out}_pool'],
            name=f'{out}_pool',
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pads] * 4,
            ceil_mode=ceil_mode,
        )
    )
    qdq(nodes, f'{out}_pool', out, f'{out}o')


def _model(name, *, second_conv=False, relu_after_pool=False, unsigned=False, export_unpooled=False, **pool):
    """conv(3x3, same) -> [relu] -> maxpool -> [relu] -> [conv(1x1)] -> flatten -> Gemm; the conv's output unsigned,
    or also a graph output, when asked."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, 24, 3, pad=1, relu=not relu_after_pool, seed=1, unsigned=unsigned)
    _pool(nodes, inits, 'a1', 'p1', unsigned=unsigned, **pool)
    kernel, stride, pads = pool.get('kernel', 2), pool.get('stride', 2), pool.get('pads', 0)
    side = -(-(H + 2 * pads - kernel) // stride) + 1 if pool.get('ceil_mode') else (H + 2 * pads - kernel) // stride + 1
    src, rows = 'p1', side * side * 24
    if relu_after_pool:
        inits += qparams('p1ro', frac=FRAC)
        nodes.append(helper.make_node('Relu', ['p1'], ['p1_relu'], name='p1_relu'))
        qdq(nodes, 'p1_relu', 'p1r', 'p1ro')
        src = 'p1r'
    if second_conv:
        _conv(nodes, inits, src, 'a2', 'c2', 24, 16, 1, pad=0, relu=True, seed=2)
        src, rows = 'a2', side * side * 16
    _head(nodes, inits, src, rows, seed=3)
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, CLASSES])]
        + ([('a1', TensorProto.FLOAT, [1, 24, H, W])] if export_unpooled else []),
        initializers=inits,
    )


def _row_split_model():
    """conv(3x3, same, one channel block) -> maxpool -> NHWC output, which the host reassembles from row slices."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, 8, 3, pad=1, relu=True, seed=1)
    _pool(nodes, inits, 'a1', 'p1')
    nodes.append(helper.make_node('Transpose', ['p1'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    return make_model(
        'pool_row_split',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, H // 2, W // 2, 8])],
        initializers=inits,
    )


def _cascade_model():
    """conv(3x3) -> conv(3x3, 24 input channels) -> maxpool -> flatten -> Gemm: only the cascade's last kernel pools."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a0', 'c0', CIN, 24, 3, pad=1, relu=True, seed=4)
    _conv(nodes, inits, 'a0', 'a1', 'c1', 24, 16, 3, pad=1, relu=True, seed=1)
    _pool(nodes, inits, 'a1', 'p1')
    _head(nodes, inits, 'p1', (H // 2) * (W // 2) * 16, seed=3)
    return make_model(
        'pool_cascade',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, CLASSES])],
        initializers=inits,
    )


CASCADE = {'c0': {'parallelism': {'cas_num': 3}}, 'c1': {'parallelism': {'cas_length': 3}}}


def _pooled(ctx):
    (conv,) = [inst for inst in ctx.ir.execution if inst.node.name == 'c1_aie']
    return conv


@pytest.mark.parametrize(
    'directives',
    [{}, {'c1': {'parallelism': {'cas_num': 3}}}, {'c1': {'parallelism': {'cas_num': 1}}}],
    ids=['whole', 'channel-split', 'paired-blocks'],
)
def test_the_conv_writes_the_pooled_tensor_and_no_pool_remains(tmp_path, directives):
    ctx = lower(_model('pooled', second_conv=True), tmp_path, directives, part=PARTS['aie1'])
    assert {inst.node.op_type for inst in ctx.ir.execution} == {'frame_fold', 'conv2d', 'dense'}  # 3 channels fold
    conv = _pooled(ctx)
    assert conv.config.pool is not None and conv.config.pool.window.kernel == (2, 2)
    (out,) = conv.node.outputs
    assert tuple(out.shape) == (1, H // 2, W // 2, 24)  # the pooled tensor, and its frame is the pooled one
    assert tuple(conv.config.io_views[out.name].logical) == (1, H // 2, W // 2, 24)


def test_row_slices_write_their_pooled_rows(tmp_path):
    """Each row slice computes its conv rows and writes half as many pooled rows, at its own offset."""
    ctx = lower(_row_split_model(), tmp_path, ROW_SPLIT, part=PARTS['aie1'])
    conv = _pooled(ctx)
    assert conv.config.parallelism.contract == 'outer' and conv.config.pool is not None
    (out,) = conv.node.outputs
    ports = [conv.variant.describe_output_staging(conv.node, conv.config, out.name, p) for p in (0, 1)]
    assert [d['offset'][2] for d in ports] == [0, H // 4]  # the second slice's pooled rows start 2 rows in
    params = conv.variant.build_template_params(conv.node, conv.config, {'row': 0, 'col': 0})
    assert (params['out_h'], params['out_rows']) == (H // 2, H // 4)


def test_row_slices_that_split_a_pool_window_are_refused(tmp_path):
    with pytest.raises(ConfigRefused, match='whole pool windows'):
        lower(
            _row_split_model(),
            tmp_path,
            {'c1': {'parallelism': {'contract': 'outer', 'cas_num': 8}}},
            part=PARTS['aie1'],
        )


def test_only_the_last_kernel_of_a_cascade_pools(tmp_path):
    ctx = lower(_cascade_model(), tmp_path, CASCADE, part=PARTS['aie1'])
    conv = _pooled(ctx)
    assert conv.config.parallelism.cas_length == 3 and conv.config.pool is not None
    assert tuple(conv.node.outputs[0].shape) == (1, (H // 2) * (W // 2) * 16)


def test_a_pool_straight_into_a_flatten_stays_one_zero_copy_row(tmp_path):
    ctx = lower(_model('pooled_flat'), tmp_path, part=PARTS['aie1'])
    conv = _pooled(ctx)
    assert conv.config.flags.emit_flattened and conv.config.pool is not None
    assert tuple(conv.node.outputs[0].shape) == (1, (H // 2) * (W // 2) * 24)


def test_a_relu_after_the_pool_fuses_too(tmp_path):
    ctx = lower(_model('pool_relu', relu_after_pool=True), tmp_path, part=PARTS['aie1'])
    conv = _pooled(ctx)
    assert conv.config.flags.use_relu and conv.config.pool is not None


def test_a_pool_whose_input_is_read_elsewhere_is_refused(tmp_path):
    """The conv would write only the pooled tensor, so the unpooled one it also exports would be lost."""
    with pytest.raises(NotImplementedError, match='a graph output'):
        lower(_model('pool_exported', export_unpooled=True), tmp_path, part=PARTS['aie1'])


def test_directives_for_a_fused_pool_are_refused(tmp_path):
    with pytest.raises(ValueError, match='direct that layer instead'):
        lower(_model('pool_directed'), tmp_path, {'p1_pool': {'parallelism': {'cas_num': 1}}}, part=PARTS['aie1'])


def test_a_malformed_fused_pool_trait_is_refused(tmp_path):
    """The conv checks the trait's exact contract, so a key its kernel would ignore cannot slip through."""
    ctx = lower(_model('pooled'), tmp_path, part=PARTS['aie1'])
    conv = _pooled(ctx).node
    conv.traits['fused_pool'].data['ceil_mode'] = 1
    with pytest.raises(ValueError, match='holds its kind and window'):
        get_family_resolver_registry().get('conv2d').resolve(conv, ctx.device, dict(conv.directives), {})


@pytest.mark.parametrize(
    'pool, reason',
    [
        ({'kernel': 3, 'stride': 2}, 'fuses a 2x2 max pool of stride 2'),
        ({'pads': 1}, 'fuses a 2x2 max pool of stride 2'),
        ({'kernel': 3, 'stride': 2, 'ceil_mode': 1}, 'ceil_mode adds a partial window'),
    ],
)
def test_pools_the_kernel_does_not_implement_are_refused(tmp_path, pool, reason):
    with pytest.raises(NotImplementedError, match=reason):
        lower(_model('refused', **pool), tmp_path, part=PARTS['aie1'])


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
@pytest.mark.parametrize(
    'model, directives',
    [
        (lambda: _model('pool_flat'), {}),
        (lambda: _model('pool_conv', second_conv=True), {'c1': {'parallelism': {'cas_num': 1}}}),
        (
            lambda: _model('pool_relu_after', relu_after_pool=True, second_conv=True),
            {'c1': {'parallelism': {'cas_num': 3}}},
        ),
        (_row_split_model, ROW_SPLIT),
        (_cascade_model, CASCADE),
        (lambda: _model('pool_unsigned', unsigned=True, second_conv=True), {}),
    ],
    ids=['flat', 'conv', 'relu_after', 'row_split', 'cascade', 'unsigned'],
)
def test_fused_pool_matches_onnx(tmp_path, part, model, directives):
    """Two distinct inferences: the output is refilled every call, so a stale one would show."""
    feeds = np.random.default_rng(5).integers(-60, 60, size=(2, 1, H, W, CIN), dtype=np.int8)
    assert_x86_matches_onnx(
        model(),
        {'x_q': feeds},
        directives,
        tmp_path,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=2,
        per_iteration=True,
    )
