"""Conv2D: channel-blocked NHWC frames on buffer ports, conv -> depthwise -> 1x1 -> flatten -> dense."""

from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.ir.graph import shard_major_order
from helpers import (
    PART,
    TensorProto,
    assert_aie_matches_onnx,
    assert_x86_matches_onnx,
    direct_edges,
    helper,
    lower,
    make_model,
    numpy_helper,
    output_staging,
    qdq,
)

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'
MLV2_PART = 'xc2ve3858-ssva2112-2mp-e-s'
# 24 channels = 3 blocks, so one model covers a 3-chain split and the first/middle/last cascade;
# 12 classes because a memtile shard needs whole 32-bit words.
H, W, CIN, C1, C2, C3, CLASSES = 8, 8, 3, 24, 24, 8, 12
FRAC = 4


def _qparams(prefix: str, *, frac: int, elem_type: int = TensorProto.INT8) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _conv(nodes, inits, x, out, name, cin, cout, k, *, pad, groups=1, relu, seed, stride=1, unsigned=False, bits=8):
    """Conv(x, W, b) [-> Relu] -> Q -> DQ with int8 weights, an int32 bias in the accumulator scale and a `bits`-bit
    output."""
    rng = np.random.default_rng(seed)
    w = rng.integers(-6, 6, size=(cout, cin // groups, k, k), dtype=np.int8)
    b = rng.integers(-64, 64, size=(cout,), dtype=np.int32)
    inits += [
        numpy_helper.from_array(w, f'{name}_w_q'),
        numpy_helper.from_array(b, f'{name}_b_q'),
        *_qparams(f'{name}_w', frac=FRAC),
        *_qparams(f'{name}_b', frac=2 * FRAC, elem_type=TensorProto.INT32),
        *_qparams(
            f'{name}o',
            frac=FRAC,
            elem_type=TensorProto.INT16 if bits == 16 else TensorProto.UINT8 if unsigned else TensorProto.INT8,
        ),
    ]
    for tag in ('w', 'b'):
        nodes.append(
            helper.make_node(
                'DequantizeLinear', [f'{name}_{tag}_q', f'{name}_{tag}_scale', f'{name}_{tag}_zp'], [f'{name}_{tag}']
            )
        )
    nodes.append(
        helper.make_node(
            'Conv',
            [x, f'{name}_w', f'{name}_b'],
            [f'{name}_conv'],
            name=name,
            kernel_shape=[k, k],
            pads=[pad, pad, pad, pad],
            strides=list(stride) if isinstance(stride, tuple) else [stride, stride],
            group=groups,
        )
    )
    pre_q = f'{name}_conv'
    if relu:
        nodes.append(helper.make_node('Relu', [pre_q], [f'{name}_relu'], name=f'{name}_relu'))
        pre_q = f'{name}_relu'
    qdq(nodes, pre_q, out, f'{name}o')


def _head(nodes, inits, src, rows, seed, *, name='fc', out='y'):
    """Flatten the ONNX NCHW view and classify it, so the Gemm rows follow the canonical order."""
    nodes.append(helper.make_node('Flatten', [src], [f'{name}_flat'], axis=1, name=f'{name}_flat'))
    w = np.random.default_rng(seed).integers(-4, 4, size=(rows, CLASSES), dtype=np.int8)
    inits += [numpy_helper.from_array(w, f'{name}_w_q'), *_qparams(f'{name}_w', frac=FRAC)]
    inits += _qparams(f'{name}o', frac=FRAC)
    nodes.append(
        helper.make_node('DequantizeLinear', [f'{name}_w_q', f'{name}_w_scale', f'{name}_w_zp'], [f'{name}_w'])
    )
    nodes.append(helper.make_node('Gemm', [f'{name}_flat', f'{name}_w'], [f'{name}_mm'], name=name))
    qdq(nodes, f'{name}_mm', out, f'{name}o')


def _start(nodes, inits):
    inits += [*_qparams('x', frac=FRAC)]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'))
    nodes.append(helper.make_node('Transpose', ['x'], ['x_nchw'], perm=[0, 3, 1, 2], name='to_nchw'))


def _model(name, nodes, inits):
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, CLASSES])],
        initializers=inits,
    )


# c1 splits its 3 output channel blocks across 3 chains; c2 then reads those 3 slices along one
# cascade (first + middle + last), and its own output stays whole for the pointwise stage.
DIRECTIVES = {'c1': {'parallelism': {'cas_num': 3}}, 'c2': {'parallelism': {'cas_num': 1}}}


@pytest.fixture
def conv_model():
    """NHWC input viewed as ONNX NCHW, conv -> depthwise -> pointwise, then flatten -> Gemm."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, C1, 3, pad=1, relu=True, seed=1)
    _conv(nodes, inits, 'a1', 'a2', 'c2', C1, C2, 3, pad=1, groups=C1, relu=True, seed=2)
    _conv(nodes, inits, 'a2', 'a3', 'c3', C2, C3, 1, pad=0, relu=False, seed=3)
    _head(nodes, inits, 'a3', H * W * C3, seed=4)
    return _model('conv2d_chain', nodes, inits)


def _valid_model(k: int):
    """One 'valid' conv (no padding): its output is narrower than the register block it computes."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'c', CIN, C3, k, pad=0, relu=True, seed=7)
    _head(nodes, inits, 'a', (H - k + 1) ** 2 * C3, seed=8)
    return _model(f'conv2d_valid_k{k}', nodes, inits)


def _frame_model(channels_in=CIN, channels_out=C3, name='conv_frame'):
    """A same-padded conv straight from the graph input, writing its frame to the boundary.

    Nothing gathers row slices back together, so a chain split by rows ends here; a streamed conv also
    has to start here, because its frame must arrive with the border already in it.
    """
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'b', channels_in, channels_out, 3, pad=1, relu=True, seed=21)
    nodes.append(helper.make_node('Transpose', ['a'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, channels_in])],
        outputs=[('y', TensorProto.FLOAT, [1, H, W, channels_out])],
        initializers=inits,
    )


def _strided_model(stride=2, k=3, channels_in=8, channels_out=8, size=16, name='conv_strided', pad=0):
    """A strided conv reading the graph boundary.

    A strided window reads every stride-th column, so the frame keeps its columns grouped by that
    residue. The boundary carries the tensor in plain order and a retiler builds the frame.
    """
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'b', channels_in, channels_out, k, pad=pad, relu=True, seed=41, stride=stride)
    nodes.append(helper.make_node('Transpose', ['a'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    out_h, out_w = (size + 2 * pad - k) // stride_h + 1, (size + 2 * pad - k) // stride_w + 1
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, size, size, channels_in])],
        outputs=[('y', TensorProto.FLOAT, [1, out_h, out_w, channels_out])],
        initializers=inits,
    )


def _beat_carry_model():
    """6x6 with Cin=4 and Cout=12, where a band is not a whole number of stream beats.

    An 8-wide image never exercises that: four rows of eight pixels are a whole number of beats
    whatever the channel count, so the wire's carry between bands stays unused. Six is.
    """
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'b', 4, 12, 3, pad=1, relu=True, seed=31)
    nodes.append(helper.make_node('Transpose', ['a'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    return make_model(
        'conv_beat_carry',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, 6, 6, 4])],
        outputs=[('y', TensorProto.FLOAT, [1, 6, 6, 12])],
        initializers=inits,
    )


def _nchw_output_model():
    """The graph output keeps ONNX's own NCHW order, which the conv's NHWC frame does not have."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'y', 'b', CIN, C1, 3, pad=1, relu=True, seed=21)
    return make_model(
        'conv_nchw_out',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, C1, H, W])],
        initializers=inits,
    )


def _row_split_model():
    return _frame_model(name='conv_row_split')


def _stream_model():
    """Several channel blocks, so the wire order and the blocked frame really differ -- and more
    than one block crosses the boundary, which a DMA-fed frame cannot do."""
    return _frame_model(channels_in=C1, channels_out=C2, name='conv_stream')


ROW_SPLIT = {'b': {'parallelism': {'contract': 'outer', 'cas_num': 2}}}
STREAM_DIRECTIVES = {'b': {'ports': 'stream'}}


def _padded_pair_model(size=H, pad=1):
    """conv(3x3, same) -> conv(3x3, `pad`) -> NHWC output: the first conv stores into the second's frame, bordered
    when the second is same-padded."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'b', CIN, C1, 3, pad=1, relu=True, seed=41)
    _conv(nodes, inits, 'a', 'c', 'd', C1, 8, 3, pad=pad, relu=True, seed=42)
    nodes.append(helper.make_node('Transpose', ['c'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    out = size - 2 + 2 * pad
    return make_model(
        'conv_padded_pair',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, size, size, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, out, out, 8])],
        initializers=inits,
    )


def _feed() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.integers(-40, 40, size=(1, H, W, CIN), dtype=np.int8)


# --------------------------------------------------------------------------- #
# contract: layouts, frames, transport plan
# --------------------------------------------------------------------------- #


def test_conv_chain_lowers_to_blocked_frames(conv_model, tmp_path):
    ctx = lower(conv_model, tmp_path, part=AIE1_PART)
    execution = ctx.ir.execution
    plan = ctx.ir.physical.plan

    c1, c2, c3, fc = (execution.get(f'{n}_aie') for n in ('c1', 'c2', 'c3', 'fc'))
    assert {inst.variant.variant_id for inst in (c1, c2, c3)} == {'conv2d.b.r.v1'}
    assert fc.variant.variant_id == 'dense.b.r.v1'
    assert 'fused_activation' in c1.node.traits and 'bias' in c1.node.roles.values()
    assert c3.node.traits['output_view'].data == {'kind': 'flatten_2d'} and c3.config.flags.emit_flattened
    assert (c2.config.spatial.kernel, c2.config.spatial.pads, c2.config.groups) == ((3, 3), (1, 1, 1, 1), C2)

    # The graph input's frame: 3 channels padded to one 8-block, a 1-pixel zero border, and the
    # image at column 2 so every stored register tile stays aligned.
    x_view = c1.config.io_views[c1.node.inputs[0].name]
    assert x_view.full == (1, H + 2, 12, 8) and x_view.origin == (0, 1, 2, 0)
    # A tensor's padded frame follows from the tensor alone, so producer and consumer agree on it.
    for producer, consumer in ((c1, c2), (c2, c3)):
        tensor = producer.node.outputs[0].name
        assert producer.config.io_views[tensor] == consumer.config.io_views[tensor]
    assert c3.config.io_views[c2.node.outputs[0].name].origin == (0, 0, 0, 0)  # 1x1 needs no border

    # Every internal edge is a direct whole-frame copy; the flattened output feeds Dense's LHS.
    assert {(e['source'], e['target']) for e in plan['direct_edges']} == {
        ('ifm[0]', 'c1_aie.in1[0]'),
        ('c1_aie.out1[0]', 'c2_aie.in1[0]'),
        ('c2_aie.out1[0]', 'c3_aie.in1[0]'),
        ('c3_aie.out1[0]', 'fc_aie.in1[0]'),
        ('fc_aie.out1[0]', 'ofm[0]'),
    }
    assert plan['buffers'] == []

    graph_input = next(p['staging'] for p in plan['io_ports'] if p['direction'] == 'input')
    assert graph_input['storage_layout'] == 'linear'
    assert graph_input['tiling_dimension'] == [8, 12, H + 2, 1]  # the PLIO carries the whole frame
    assert graph_input['io_boundary_dimension'] == [CIN, W, H, 1]
    assert graph_input['logical_origin'] == [0, -2, -1, 0]  # the window opens on the border


@pytest.mark.parametrize('k', [3, 7])
def test_valid_conv_frame_covers_the_computed_width(k, tmp_path):
    """The kernel computes whole register tiles, so the frame must hold every column it reads."""
    ctx = lower(_valid_model(k), tmp_path, part=AIE1_PART)
    conv = ctx.ir.execution.get('c_aie')
    params = conv.variant.build_template_params(conv.node, conv.config, {'row': 0, 'col': 0})
    out_w = H - k + 1
    assert params['out_w'] == out_w
    assert params['out_w_computed'] == -(-out_w // 4) * 4  # 2 register tiles x M=2 on AIE1
    assert params['in_origin_c'] - conv.config.spatial.pads[1] + params['out_w_computed'] + k - 1 <= params['in_cols']


def test_conv_weights_pack_compact_groups_into_dense_tiles(conv_model, tmp_path):
    """AIE1 has no channelwise core: a depthwise conv runs the mmul core as block-diagonal tiles."""
    ctx = lower(conv_model, tmp_path, part=AIE1_PART)
    c2 = ctx.ir.execution.get('c2_aie')
    assert not c2.variant.uses_depthwise_core(c2.node, c2.config)
    # The IR keeps the compact per-group form; the variant expands the groups when it packs.
    assert tuple(c2.node.inputs[1].shape) == (3, 3, 1, C2)
    packed = c2.artifacts['packed_weights']
    blocks = C2 // 8
    padded = blocks if blocks == 1 else blocks + blocks % 2  # output blocks in pairs, except a lone block
    assert packed.shape == (1, 1, 9 * (C1 // 8) * padded * 64)
    tiles = packed.reshape(9, C1 // 8, padded, 8, 8)  # (tap, cin block, cout block, 8, 8)
    # Depthwise: a tap's (cin, cout) plane is block-diagonal, and each diagonal holds the one
    # coefficient that channel has -- this is the compact-weight contract the packer expands.
    compact = np.asarray(c2.node.inputs[1].data).reshape(3, 3, 1, C2)
    for tap, (ky, kx) in enumerate([(ky, kx) for ky in range(3) for kx in range(3)]):
        for cb in range(C1 // 8):
            for nb in range(padded):
                block = tiles[tap, cb, nb]
                if cb != nb:
                    assert not block.any()
                else:
                    assert np.array_equal(np.diag(block), np.rint(compact[ky, kx, 0, cb * 8 : cb * 8 + 8] * 16))
                    assert np.count_nonzero(block) == np.count_nonzero(np.diag(block))


@pytest.mark.parametrize('part', [PART, MLV2_PART], ids=['aie-ml', 'aie-mlv2'])
def test_depthwise_core_packs_only_the_channel_taps(conv_model, tmp_path, part):
    """On one AIE-ML or AIE-MLv2 tile a depthwise conv runs the channelwise core, which reads each channel's own taps:
    per block and kernel row, the taps in groups of four x 8 channels."""
    c2 = lower(conv_model, tmp_path, part=part).ir.execution.get('c2_aie')
    assert c2.variant.uses_depthwise_core(c2.node, c2.config)
    compact = np.rint(np.asarray(c2.node.inputs[1].data).reshape(3, 3, C2) * 16)  # ky, kx, channel
    want = np.zeros((C2 // 8, 3, 4, 8))  # block, ky, tap, channel
    want[:, :, :3, :] = compact.reshape(3, 3, C2 // 8, 8).transpose(2, 0, 1, 3)
    assert np.array_equal(c2.artifacts['packed_weights'].reshape(want.shape), want)


def test_conv_partitions_channel_blocks_across_tiles(conv_model, tmp_path):
    """cas_num splits the output channel blocks; the consumer reads back exactly those slices."""
    ctx = lower(conv_model, tmp_path, DIRECTIVES, part=AIE1_PART)
    c1, c2 = ctx.ir.execution.get('c1_aie'), ctx.ir.execution.get('c2_aie')
    assert (c1.config.parallelism.cas_num, c1.config.parallelism.cas_length) == (3, 1)
    assert (c2.config.parallelism.cas_num, c2.config.parallelism.cas_length) == (1, 3)
    assert c2.variant.footprint(c2.node, c2.config).width == 3  # a three-tile cascade

    # Each of c1's ports carries one channel block of the shared frame, and c2 reads them back.
    tensor = c1.node.outputs[0].name
    write = [c1.variant.describe_output_staging(c1.node, c1.config, tensor, port) for port in range(3)]
    read = [c2.variant.describe_input_staging(c2.node, c2.config, tensor, port) for port in range(3)]
    assert [d['tiling_dimension'][0] for d in write] == [8, 8, 8]
    assert [d['offset'][0] for d in write] == [0, 8, 16]
    assert all(w['tile_traversal'] == r['tile_traversal'] and w['offset'] == r['offset'] for w, r in zip(write, read))

    # Weights are cut per (chain, column) tile; the bias belongs to the tile that stores, one per
    # chain, and must survive the cut -- a zeroed bias here is the cascade-bias regression.
    assert c1.artifacts['packed_weights'].shape[:2] == (3, 1)
    assert c2.artifacts['packed_weights'].shape[:2] == (1, 3)
    assert c2.artifacts['packed_bias'].shape[0] == 1 and c2.artifacts['packed_bias'].any()
    assert all(row.any() for row in c1.artifacts['packed_bias'])
    plan = ctx.ir.physical.plan
    edges = {(e['source'], e['target']) for e in plan['direct_edges']}
    assert {('c1_aie.out1[2]', 'c2_aie.in1[2]'), ('c1_aie.out1[0]', 'c2_aie.in1[0]')} <= edges


def _sharded_heads_model():
    """Two heads whose conv chains each flatten their own output channels: `ca` (4 blocks, 2 per chain, with a fused
    2x2 max pool) into `fa`, and `cb` (2 blocks, 1 per chain) into `fb`."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'ca', CIN, 32, 3, pad=1, relu=True, seed=31)
    inits += _qparams('pao', frac=FRAC)
    nodes.append(helper.make_node('MaxPool', ['a'], ['pa_pool'], name='pa', kernel_shape=[2, 2], strides=[2, 2]))
    qdq(nodes, 'pa_pool', 'pa', 'pao')
    _head(nodes, inits, 'pa', H // 2 * W // 2 * 32, seed=32, name='fa', out='ya')
    _conv(nodes, inits, 'x_nchw', 'b', 'cb', CIN, 16, 3, pad=1, relu=True, seed=33)
    _head(nodes, inits, 'b', H * W * 16, seed=34, name='fb', out='yb')
    return make_model(
        'conv_sharded_heads',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('ya', TensorProto.FLOAT, [1, CLASSES]), ('yb', TensorProto.FLOAT, [1, CLASSES])],
        initializers=inits,
    )


# One cascade stage per chain for `fa`, which takes each chain's slice directly; four for `fb`, which cuts the same
# row differently, so a memory tile re-slices it.
SHARDED = {
    'ca': {'parallelism': {'cas_num': 2}},
    'fa': {'parallelism': {'cas_num': 1, 'cas_length': 2}},
    'cb': {'parallelism': {'cas_num': 2}},
    'fb': {'parallelism': {'cas_num': 1, 'cas_length': 4}},
}


def test_channel_chains_of_a_flattened_conv_feed_the_dense_reduction(tmp_path):
    """Each chain flattens its own output channels, so the row is stored chain by chain and the Dense adopts that
    order into its weight rows: a cascade stage per chain takes its slice directly, and any other cut of the row
    goes through a memory tile."""
    assert shard_major_order(8, 2, 2).tolist() == [0, 1, 4, 5, 2, 3, 6, 7]
    ctx = lower(_sharded_heads_model(), tmp_path, SHARDED)
    execution = ctx.ir.execution
    assert execution.tensor_contracts['fa_flat'].inner_shards == (2, 16)
    assert execution.tensor_contracts['fb_flat'].inner_shards == (2, 8)
    assert [execution.get(name).config.lhs_inner_shards for name in ('fa_aie', 'fb_aie')] == [(2, 16), (2, 8)]
    assert {('ca_aie', 'fa_aie')} <= direct_edges(ctx) and ('cb_aie', 'fb_aie') not in direct_edges(ctx)
    assert [(b['name'], len(b['writers']), len(b['readers'])) for b in ctx.ir.physical.plan['buffers']] == [
        ('buffer_fb_flat', 2, 4)
    ]


def test_a_row_stored_chain_by_chain_is_no_graph_output(tmp_path):
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'c', CIN, 16, 3, pad=1, relu=True, seed=1)
    nodes.append(helper.make_node('Flatten', ['a'], ['y'], axis=1, name='flat'))
    model = make_model(
        'conv_flat_out',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, H * W * 16])],
        initializers=inits,
    )
    with pytest.raises(ConfigRefused, match='stores its inner axis shard by shard, but a graph output leaves'):
        lower(model, tmp_path, {'c': {'parallelism': {'cas_num': 2}}})


def test_conv_rejects_partitions_it_cannot_cut(conv_model, tmp_path):
    with pytest.raises(ConfigRefused, match='cas_num=5 does not split'):
        lower(conv_model, tmp_path, {'c1': {'parallelism': {'cas_num': 5}}}, part=AIE1_PART)
    # c1 feeds a 3x3 conv, so its output frame carries a border no row slice can own.
    with pytest.raises(ConfigRefused, match='output split by rows'):
        lower(conv_model, tmp_path, {'c1': {'parallelism': {'contract': 'outer', 'cas_num': 2}}}, part=AIE1_PART)


def test_frame_refuses_consumers_that_read_different_windows(tmp_path):
    """One padded frame serves one window: a fanout whose branches pad differently needs a
    per-consumer view, which transport does not materialize, so it is refused up front."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'c', CIN, C3, 3, pad=1, relu=True, seed=7)
    _conv(nodes, inits, 'a', 'b1', 'w3', C3, C3, 3, pad=1, relu=True, seed=8)  # needs a 1-pixel border
    _conv(nodes, inits, 'a', 'b2', 'w1', C3, C3, 1, pad=0, relu=True, seed=9)  # needs none
    _conv(nodes, inits, 'b1', 'b3', 'p1', C3, C3, 1, pad=0, relu=False, seed=10)
    _conv(nodes, inits, 'b2', 'b4', 'p2', C3, C3, 1, pad=0, relu=False, seed=11)
    nodes.append(helper.make_node('Add', ['b3', 'b4'], ['sum'], name='sum'))
    inits += _qparams('sumo', frac=FRAC)
    qdq(nodes, 'sum', 'a2', 'sumo')
    _conv(nodes, inits, 'a2', 'a3', 'p3', C3, C3, 1, pad=0, relu=False, seed=12)
    _head(nodes, inits, 'a3', H * W * C3, seed=13)
    with pytest.raises(NotImplementedError, match='read different windows'):
        lower(_model('conv_fanout', nodes, inits), tmp_path, part=AIE1_PART)


def test_frame_rows_hold_whole_register_tiles(tmp_path):
    """A producer stores whole register tiles, so every frame row starts on one: 64 bytes on AIE-MLv2, where
    32-byte rows put every other row's stores out of alignment."""
    ctx = lower(_padded_pair_model(), tmp_path, part=MLV2_PART)
    for inst in ctx.ir.execution:
        m = inst.config.microtiling.microtile_m
        assert all(view.full[2] % m == 0 for view in inst.config.io_views.values() if len(view.full) == 4)


def test_outer_splits_rows_into_overlapping_slices(tmp_path):
    """Row slices overlap by the window span, and a slice's window may open before the image: the host
    clips it against the tensor and zero-fills the rest, so no kernel has to own that border."""
    ctx = lower(_row_split_model(), tmp_path, ROW_SPLIT, part=AIE1_PART)
    conv = ctx.ir.execution.get('b_aie')
    assert conv.config.parallelism.contract == 'outer' and conv.config.parallelism.cas_num == 2

    view = conv.config.io_views[conv.node.inputs[0].name]
    assert view.full[1] == H + 2 and view.tile[1] == H // 2 + 2  # a slice of 4 rows plus its halo
    ports = [conv.variant.describe_input_staging(conv.node, conv.config, conv.node.inputs[0].name, p) for p in (0, 1)]
    assert [d['offset'][2] for d in ports] == [0, 4]  # the second slice starts 4 frame rows in
    assert [d['logical_origin'][2] for d in ports] == [-1, 3]  # slice 0 opens on the top border
    assert all(d['tiling_dimension'][2] == H // 2 + 2 for d in ports)

    params = conv.variant.build_template_params(conv.node, conv.config, {'row': 0, 'col': 0})
    assert (params['out_h'], params['in_rows']) == (H // 2, H // 2 + 2)
    assert not params['fills_border']  # the host delivers each slice's window, border included

    # Each slice writes its own rows of the output, which the host reassembles.
    out_ports = [
        conv.variant.describe_output_staging(conv.node, conv.config, conv.node.outputs[0].name, p) for p in (0, 1)
    ]
    assert [d['logical_origin'][2] for d in out_ports] == [0, H // 2]
    assert all(d['tiling_dimension'][2] == H // 2 for d in out_ports)

    plan = ctx.ir.physical.plan
    edges = {(e['source'], e['target']) for e in plan['direct_edges']}
    assert {('ifm[0]', 'b_aie.in1[0]'), ('ifm[1]', 'b_aie.in1[1]')} <= edges
    assert {('b_aie.out1[0]', 'ofm[0]'), ('b_aie.out1[1]', 'ofm[1]')} <= edges


@pytest.mark.parametrize('pad', [1, 0], ids=['bordered', 'overlapping'])
def test_row_slices_feed_no_window_past_their_rows(tmp_path, pad):
    """A row slice writes its own rows once, so it cannot fill the consumer's zero border (same) nor the rows its
    neighbour's window also reads (valid)."""
    with pytest.raises(ConfigRefused, match='rows past its slice'):
        lower(_padded_pair_model(pad=pad), tmp_path, ROW_SPLIT, part=AIE1_PART)


def test_stream_conv_carries_the_logical_tensor(tmp_path):
    """A stream carries a wire order, not a memory layout, and the wire is the tensor itself: no
    border, no padded channels, no computed-width tail. The kernel owns all of that."""
    ctx = lower(_stream_model(), tmp_path, STREAM_DIRECTIVES, part=AIE1_PART)
    conv = ctx.ir.execution.get('b_aie')
    assert conv.variant.variant_id == 'conv2d.s.r.v1'
    assert int(conv.config.io_views[conv.node.inputs[0].name].full[-1]) // 8 == 3  # three blocks
    assert {b.kind for b in (*conv.ports.inputs.values(), *conv.ports.outputs.values())} == {'stream'}

    logical = [C2, W, H, 1]  # buffer order: channels, columns, rows, batch
    for staging in (
        conv.variant.describe_input_staging(conv.node, conv.config, conv.node.inputs[0].name, 0),
        conv.variant.describe_output_staging(conv.node, conv.config, conv.node.outputs[0].name, 0),
    ):
        assert staging['storage_layout'] == 'linear'
        assert staging['tiling_dimension'] == staging['buffer_dimension'] == logical
        assert staging['logical_origin'] == [0, 0, 0, 0]  # the window is the tensor

    plan = ctx.ir.physical.plan
    assert plan['buffers'] == [] and plan['kernel_write_accesses'] == []
    assert ('ifm[0]', 'b_aie.in1[0]') in {(e['source'], e['target']) for e in plan['direct_edges']}


def test_conv_refuses_dilation(tmp_path):
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    inits += [
        numpy_helper.from_array(np.zeros((C3, CIN, 3, 3), np.int8), 'w_q'),
        *_qparams('w', frac=FRAC),
        *_qparams('co', frac=FRAC),
    ]
    nodes.append(helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w']))
    nodes.append(helper.make_node('Conv', ['x_nchw', 'w'], ['cv'], kernel_shape=[3, 3], dilations=[2, 2], name='conv'))
    qdq(nodes, 'cv', 'a', 'co')
    _head(nodes, inits, 'a', 4 * 4 * C3, seed=9)
    # A dilated window skips input pixels the way a strided one skips outputs, but it does not
    # group into polyphase classes the same way, so it is still refused rather than misread.
    with pytest.raises(NotImplementedError, match='dilations'):
        lower(_model('conv_dilation', nodes, inits), tmp_path, part=AIE1_PART)


def test_conv_refuses_an_unfolded_batchnorm(tmp_path):
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    inits += [numpy_helper.from_array(np.zeros((C3, CIN, 3, 3), np.int8), 'w_q'), *_qparams('w', frac=FRAC)]
    inits += [numpy_helper.from_array(np.ones(C3, np.float32), f'bn_{name}') for name in 'gbmv']
    inits += _qparams('co', frac=FRAC)
    nodes.append(helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w']))
    nodes.append(helper.make_node('Conv', ['x_nchw', 'w'], ['cv'], pads=[1, 1, 1, 1], name='conv'))
    nodes.append(helper.make_node('BatchNormalization', ['cv', 'bn_g', 'bn_b', 'bn_m', 'bn_v'], ['bn'], name='bn'))
    qdq(nodes, 'bn', 'a', 'co')
    _head(nodes, inits, 'a', H * W * C3, seed=9)
    with pytest.raises(NotImplementedError, match='fold BatchNormalization'):
        lower(_model('conv_bn', nodes, inits), tmp_path)


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_stream_conv_matches_onnx(tmp_path, part):
    """The wire order and the blocked frame must agree, or the image lands scrambled. Two different
    inputs, because the band frame and the beat cursor outlive the call: the second inference must
    not inherit the rows and the half beat the first one left behind."""
    feeds = np.random.default_rng(12).integers(-40, 40, size=(2, 1, H, W, C1), dtype=np.int8)
    assert_x86_matches_onnx(
        _stream_model(),
        {'x_q': feeds},
        STREAM_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=2,
        per_iteration=True,
    )


def test_graph_output_keeps_the_order_onnx_declares(tmp_path):
    """A pending axis order on a graph output is realized or refused -- never quietly ignored,
    which would expose a differently shaped tensor than the ONNX graph promises."""
    with pytest.raises(NotImplementedError, match=r"transpose feeds graph output 'y'"):
        lower(_nchw_output_model(), tmp_path, part=AIE1_PART)


@pytest.mark.parametrize('part', [PART, MLV2_PART])
def test_a_graph_output_off_a_memory_tile_moves_whole_plio_beats(tmp_path, part):
    """12 int8 classes end mid-beat on a 128-bit PLIO: the memory tile and the host move 16 and the host trims."""
    nodes, inits = [], []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, 8, 3, pad=1, relu=True, seed=1)
    _head(nodes, inits, 'a1', H * W * 8, seed=3)
    directives = {'c1': {'parallelism': {'cas_num': 1}}, 'fc': {'io_route': {'outputs': {'y': 'memtile'}}}}
    ctx = lower(_model('conv_beats', nodes, inits), tmp_path, directives, part=part)
    plan = ctx.ir.physical.plan
    assert [b['tensor'] for b in plan['buffers']] == ['y']
    out = next(p for p in plan['io_ports'] if p['direction'] == 'output')
    assert out['descriptor']['io_boundary_dimension'] == [CLASSES, 1]
    assert out['descriptor']['tiling_dimension'] == out['staging']['tiling_dimension'] == [16, 1]


def test_conv2d_refuses_directives_it_does_not_implement(tmp_path):
    """The kernel fixes its own register tiling, so a microtiling request is refused, not ignored."""
    with pytest.raises(NotImplementedError, match='microtiling'):
        lower(_frame_model(), tmp_path, {'b': {'microtiling': {'microtile_m': 999}}}, part=AIE1_PART)


@pytest.mark.requires_vitis
def test_stream_conv_moves_partial_channel_blocks(tmp_path):
    """Cin=11 is one whole channel block and a tail, so the reader has to shift a block across a
    beat boundary -- the path that neither a blocked nor an all-tail input reaches."""
    feed = np.random.default_rng(3).integers(-40, 40, size=(1, H, W, 11), dtype=np.int8)
    assert_x86_matches_onnx(
        _frame_model(channels_in=11, channels_out=C1, name='conv_tail'),
        {'x_q': feed},
        STREAM_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=1,
        part=AIE1_PART,
    )


@pytest.mark.requires_vitis
def test_stream_conv_sends_partial_channel_blocks(tmp_path):
    """Cout=11 is one whole channel block and a tail, so a band's bytes do not divide into beats
    and the writer carries the remainder into the next band.

    This checks the ordering and that carry. It cannot check the alignment the path also needs:
    x86 loads unaligned addresses happily, so only a native run sees that.
    """
    feed = np.random.default_rng(4).integers(-40, 40, size=(1, H, W, C1), dtype=np.int8)
    assert_x86_matches_onnx(
        _frame_model(channels_in=C1, channels_out=11, name='conv_send_tail'),
        {'x_q': feed},
        STREAM_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=1,
        part=AIE1_PART,
    )


def _strided_chain_model(size=16, first_stride=1, name='conv_strided_chain'):
    """A same-padded conv, 8 -> 16 channels, feeding a stride-2 one: the strided conv reads another
    kernel's frame, two channel blocks of it."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a', 'first', 8, 16, 3, pad=1, relu=True, seed=5, stride=first_stride)
    _conv(nodes, inits, 'a', 'b', 'second', 16, 8, 3, pad=1, relu=True, seed=6, stride=2)
    nodes.append(helper.make_node('Transpose', ['b'], ['y'], perm=[0, 2, 3, 1], name='to_nhwc'))
    out = ((size - 1) // first_stride) // 2 + 1
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, size, size, 8])],
        outputs=[('y', TensorProto.FLOAT, [1, out, out, 8])],
        initializers=inits,
    )


def test_strided_conv_retiles_its_producers_frame(tmp_path):
    """A producer writes whole register tiles, which span every residue group, so a retiler -- a
    kernel of its own in the execution graph, not in the model -- reads the frame as written and
    hands the conv the grouped one. The execution graph alone says so; the logical graph keeps the
    model's two convs. The frame's hand-over must be shared memory, so both of its ports are pinned
    to one memory; the producer's is an ordinary direct edge."""
    from aie4ml.ir import ExecutionInput

    ctx = lower(_strided_chain_model(), tmp_path, part=AIE1_PART)
    assert [inst.name for inst in ctx.ir.execution] == ['first_aie', 'second_aie_retile', 'second_aie']
    assert [node.name for node in ctx.ir.logical if not node.is_folded_view] == ['first_aie', 'second_aie']
    logical = next(node for node in ctx.ir.logical if node.name == 'second_aie')
    assert [t.name for t in logical.inputs if t.data is None] == ['first_relu']  # untouched by lowering

    retile, second = ctx.ir.execution.get('second_aie_retile'), ctx.ir.execution.get('second_aie')
    frame = second.variant.retiled_frame(second.node)
    assert retile.inputs == (ExecutionInput('first_relu', 'lhs', shared_memory=False),)
    assert second.inputs == (ExecutionInput(frame, 'lhs', shared_memory=True),)
    assert ctx.ir.execution.values[frame].producer == 'second_aie_retile'
    assert {('first_aie', 'second_aie_retile'), ('second_aie_retile', 'second_aie')} <= direct_edges(ctx)

    # The retiler reads the producer's frame as written; only a retiler writes a column-grouped one.
    write = output_staging(ctx, 'first_aie')
    read = retile.variant.describe_input_staging(retile.node, retile.config, 'first_relu', 0)
    assert write['tile_traversal'] == read['tile_traversal'] and write['offset'] == read['offset']
    assert 'column_phases' not in write and 'transfer_bytes' not in read
    assert second.variant.describe_input_staging(second.node, second.config, frame, 0)['column_phases'] == 2
    assert not second.variant.build_template_params(second.node, second.config, {'row': 0, 'col': 0})['fills_border']

    def pinned(inst, group):
        where = ctx.ir.physical.placements[inst.name]
        return {
            (where['col'] + loc.rel_col, where['row'] + loc.rel_row, loc.banks)
            for loc in inst.variant.buffer_locations(inst.node, inst.config, where['row'])
            if loc.port_group == group
        }

    assert pinned(retile, 'out1') == pinned(second, 'in1') != set()

    # Two producer chains feed a cascade of two: one retiler kernel per chain, each reading that
    # chain's slice of the frame in place.
    ctx = lower(_strided_chain_model(), tmp_path / 'chains', {'first': {'parallelism': {'cas_num': 2}}}, part=AIE1_PART)
    retile = ctx.ir.execution.get('second_aie_retile')
    assert [w.first_channel for w in retile.config.windows] == [0, 8]
    for port in range(2):
        write = output_staging(ctx, 'first_aie', port)
        read = retile.variant.describe_input_staging(retile.node, retile.config, 'first_relu', port)
        assert write['offset'] == read['offset']
    legs = [e for e in ctx.ir.physical.plan['direct_edges'] if e['tensor'] == 'first_relu']
    assert [e['realization'] for e in legs] == ['shared_memory'] * 2


def test_shared_edge_needs_room_for_one_buffer(tmp_path):
    """Pinning the strided conv right beside its producer puts the conv's input memory in the
    producer's tile, where no buffer of theirs is shared, and leaves no tile for the retiler whose
    frame it must share: placement refuses rather than accept a DMA hop."""
    directives = {'first': {'placement': {'col': 7, 'row': 0}}, 'second': {'placement': {'col': 8, 'row': 0}}}
    with pytest.raises(ValueError, match='conflicts with another anchor'):
        lower(_strided_chain_model(), tmp_path, directives, part=AIE1_PART)


@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    'stride,k,cin,size', [(2, 3, 8, 16), (2, 7, 3, 18), (3, 3, 8, 15)], ids=['s2k3', 's2k7-lowc', 's3k3']
)
def test_strided_conv_matches_onnx(tmp_path, stride, k, cin, size):
    """The retiler groups the frame's columns by residue and the conv reads them as it reads a
    dense window, so a strided conv must be exact for every stride, kernel and channel count.

    Six different inputs, because two of these tensors are not whole 16-byte units and travel with
    padding after them: a transfer framed wrongly shifts every later inference, which repeating one
    input would hide.
    """
    feeds = np.random.default_rng(21).integers(-40, 40, size=(6, 1, size, size, cin), dtype=np.int8)
    assert_x86_matches_onnx(
        _strided_model(stride=stride, k=k, channels_in=cin, size=size),
        {'x_q': feeds},
        {},
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=6,
        per_iteration=True,
    )


@pytest.mark.requires_vitis
def test_vertical_only_stride_takes_the_plain_boundary(tmp_path):
    """A stride along rows only skips whole frame rows, which the conv does by itself: no column
    grouping, so no retiler, and the boundary is the one an unstrided conv reads."""
    model = _strided_model(stride=(2, 1), k=3, channels_in=8, size=16, name='conv_row_stride')
    entry = lower(model, tmp_path / 'lowered', part=AIE1_PART).ir.execution.get('b_aie')
    assert not entry.variant.retiles_input(entry.config)
    assert entry.ports.inputs['x_q'].endpoints == (('kk[0].in[0]',),)

    feeds = np.random.default_rng(29).integers(-40, 40, size=(3, 1, 16, 16, 8), dtype=np.int8)
    assert_x86_matches_onnx(
        model,
        {'x_q': feeds},
        {},
        tmp_path / 'numeric',
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=3,
        per_iteration=True,
    )


def _single_channel_strided_conv():
    """One input channel, 18x18, 7x7 stride 2, five filters: the conv reads 17 of the 18 rows, 306
    bytes, not whole units."""
    return _strided_model(stride=2, k=7, channels_in=1, channels_out=5, size=18, name='conv_single_channel_s2')


def test_strided_boundary_conv_is_retiled(tmp_path):
    """The boundary carries the rows the conv reads, in plain order, and a retiler builds the frame.
    Those rows are 306 bytes and one inference moves 320: the two are kept apart."""
    from aie4ml.simulation import build_io_layout

    ctx = lower(_single_channel_strided_conv(), tmp_path, part=AIE1_PART)
    retile, conv = ctx.ir.execution.get('b_aie_retile'), ctx.ir.execution.get('b_aie')
    assert retile.op_type == 'frame_retile' and conv.variant.retiles_input(conv.config)
    assert list(retile.ports.inputs) == ['x_q'] and 'x_q' not in conv.ports.inputs

    port = build_io_layout(ctx).inputs['x_q'][0]
    assert port.numpy_tile_shape == (1, 17, 18, 1)  # stride 2 never reaches row 18; nothing around them
    assert port.transfer_bytes == 320
    assert port.staging['storage_layout'] == 'linear'


def _row_split_strided_conv(channels_in=8):
    """18x18, 3x3 stride 2, padded: nine output rows, three row slices of three."""
    return _strided_model(
        stride=2, k=3, channels_in=channels_in, channels_out=8, size=18, name='conv_row_split_s2', pad=1
    )


ROW_SPLIT_3 = {'b': {'parallelism': {'contract': 'outer', 'cas_num': 3}}}
# Twelve channels in a cascade of two: an 8-channel slice and a 4-channel one that does not fill its block.
ROW_SPLIT_X_CASCADE = {'b': {'parallelism': {'contract': 'outer', 'cas_num': 3, 'cas_length': 2}}}


def test_strided_conv_splits_by_rows(tmp_path):
    """Each row slice gets a retiler kernel of its own, beside the slice's conv tile, reading only the
    rows of the tensor its window covers -- the first slice's window opens on the top pad, which that kernel
    builds -- and handing its frame over as one shared buffer on every row. Chains of output channels
    all read one retiler kernel's frame, by DMA."""
    from aie4ml.op_impls.families.conv2d.config import RetileWindow

    ctx = lower(_row_split_strided_conv(), tmp_path / 'rows', ROW_SPLIT_3, part=AIE1_PART)
    retile, conv = ctx.ir.execution.get('b_aie_retile'), ctx.ir.execution.get('b_aie')
    # Output rows 3b..3b+2 read frame rows 6b..6b+6: tensor rows 6b-1..6b+5, clipped to the tensor.
    assert retile.config.windows == (
        RetileWindow(first_row=0, rows=6, origin_row=1, first_channel=0, channels=8, transfer_bytes=864),
        RetileWindow(first_row=5, rows=7, origin_row=0, first_channel=0, channels=8, transfer_bytes=1008),
        RetileWindow(first_row=11, rows=7, origin_row=0, first_channel=0, channels=8, transfer_bytes=1008),
    )
    reads = [conv.variant.describe_input_staging(conv.node, conv.config, conv.inputs[0].tensor, b) for b in range(3)]
    writes = [retile.variant.describe_output_staging(retile.node, retile.config, '', b) for b in range(3)]
    assert [d['offset'] for d in reads] == [d['offset'] for d in writes]
    assert [d['offset'][2] for d in reads] == [0, 6, 12]  # three output rows a slice, stride 2
    edges = [e for e in ctx.ir.physical.plan['direct_edges'] if e['tensor'] == conv.inputs[0].tensor]
    assert [e['realization'] for e in edges] == ['shared_memory'] * 3

    # With a cascade, each row slice's window splits into channel slices: one retiler kernel per (row,
    # slice), in the conv's port order. The cascade reads its inputs in its own row, beyond the
    # retilers' reach, so the hand-over is not required to be shared.
    ctx = lower(_row_split_strided_conv(channels_in=12), tmp_path / 'cascade', ROW_SPLIT_X_CASCADE, part=AIE1_PART)
    retile, conv = ctx.ir.execution.get('b_aie_retile'), ctx.ir.execution.get('b_aie')
    assert [(w.first_row, w.first_channel, w.channels) for w in retile.config.windows] == [
        (row, first, count) for row in (0, 5, 11) for first, count in ((0, 8), (8, 4))
    ]
    reads = [conv.variant.describe_input_staging(conv.node, conv.config, conv.inputs[0].tensor, p) for p in range(6)]
    writes = [retile.variant.describe_output_staging(retile.node, retile.config, '', p) for p in range(6)]
    assert [d['offset'] for d in reads] == [d['offset'] for d in writes]
    assert not conv.inputs[0].shared_memory

    # Chains of output channels each read the whole frame: one retiler kernel, multicast by DMA.
    two_blocks = _strided_model(stride=2, k=3, channels_in=8, channels_out=16, size=16)
    ctx = lower(two_blocks, tmp_path / 'inner', {'b': {'parallelism': {'cas_num': 2}}}, part=AIE1_PART)
    retile, conv = ctx.ir.execution.get('b_aie_retile'), ctx.ir.execution.get('b_aie')
    assert [(w.first_channel, w.channels) for w in retile.config.windows] == [(0, 8)]
    assert [b.endpoints for b in conv.ports.inputs.values()] == [(('kk[0].in[0]', 'kk[1].in[0]'),)]
    assert not conv.inputs[0].shared_memory


def test_physical_plan_proves_each_shared_edge(tmp_path):
    """A shared edge is proven before any code exists: the verifier re-derives it from the finished
    plan and refuses one whose ports no longer name one memory, or that a DMA access pattern would
    turn into a copy."""
    from aie4ml.passes.verify_physical import verify_physical

    ctx = lower(_strided_chain_model(), tmp_path, part=AIE1_PART)
    verify_physical(ctx)
    edges = {e['tensor']: e for e in ctx.ir.physical.plan['direct_edges']}
    assert edges[ctx.ir.execution.get('second_aie').inputs[0].tensor]['realization'] == 'shared_memory'
    # Not required, but placement put the producer where its output and the retiler's input coincide.
    assert edges['first_relu']['realization'] == 'shared_memory'

    ctx.ir.physical.plan['kernel_read_accesses'].append({'endpoint': 'second_aie.kk[0].in[0]'})
    with pytest.raises(RuntimeError, match='DMA access pattern'):
        verify_physical(ctx)
    ctx.ir.physical.plan['kernel_read_accesses'].pop()

    ctx.ir.physical.placements['second_aie_retile']['row'] += 1
    with pytest.raises(RuntimeError, match='not one memory'):
        verify_physical(ctx)


def test_generated_graph_pins_the_frame(tmp_path):
    """The frame's hand-over is one buffer because both ops' graphs pin their ends of it to the same
    banks -- the retiler's output and the conv's input, banks 0 and 3 of the retiler's tile -- so the
    compiler places that one buffer or refuses. Nothing is pinned from outside the op graphs, and the
    build needs no check of its own afterwards."""
    from aie4ml import from_onnx

    config = {'Part': AIE1_PART, 'AIEConfig': {'BatchSize': 1, 'Iterations': 1}, 'LayerDirectives': {}}
    from_onnx(_strided_chain_model(), config, output_dir=tmp_path, project_name='strided').write()
    params = (tmp_path / 'src' / 'parameters.h').read_text()
    assert '{ 0, 0, 2, 0, 3 }' in params  # the retiler's frame: its own tile
    assert '{ -1, 0, 2, 0, 3 }' in params  # the conv's input: its west neighbour, the retiler
    assert 'location<buffer>' not in (tmp_path / 'src' / 'graph_plan.h').read_text()
    assert 'python3' not in (tmp_path / 'Makefile').read_text()


@pytest.mark.parametrize(
    'part, size, blocks',
    [
        ('xcve2802-vsvh1760-2mp-e-s', 16, 4),  # both fill 16 columns: the larger measured faster
        ('xcve2802-vsvh1760-2mp-e-s', 8, 2),
        (MLV2_PART, 8, 2),  # 4 blocks of 8 would compute 32 columns for 8
        (AIE1_PART, 8, 2),
    ],
)
def test_register_blocking_pads_the_output_width_least(tmp_path, part, size, blocks):
    ctx = lower(_padded_pair_model(size), tmp_path, part=part)
    assert ctx.ir.execution.get('b_aie').config.spatial_blocks == blocks


def test_convs_of_different_blockings_share_one_frame(tmp_path):
    """16 wide takes 4 blocks, the stride-2 consumer's 8 wide takes 2: the frame between them is one layout."""
    ctx = lower(_strided_chain_model(size=16), tmp_path, part='xcve2802-vsvh1760-2mp-e-s')
    assert [ctx.ir.execution.get(n).config.spatial_blocks for n in ('first_aie', 'second_aie')] == [4, 2]


def test_frame_larger_than_a_bank_is_split(tmp_path):
    """Each activation copy sits in one bank, as for Dense: AIE-MLv2's 16x16x16 frame is 19.6 KB, over its
    16 KB bank, so the layer is split -- and refused when the user holds it to one tile."""
    ctx = lower(_strided_chain_model(), tmp_path, part='vek385_base')
    split = ctx.ir.execution.get('first_aie').config.parallelism
    assert split.cas_num * split.cas_length > 1
    one = {'first': {'parallelism': {'cas_num': 1, 'cas_length': 1}}}
    with pytest.raises(ConfigRefused, match='memory bank holds'):
        lower(_strided_chain_model(), tmp_path / 'one', one, part='vek385_base')


def _compiled_buffers(project, port: str):
    """The buffers the AIE compiler gave one kernel port, from its own report."""
    import json

    report = json.loads((project / 'Work' / 'reports' / 'compiler_report.json').read_text())
    ids = [i for i, info in report['portInstances'].items() if info['qualifiedName'].endswith(f'.{port}')]
    assert len(ids) == 1, port
    return sorted(b['generatedName'] for b in report['mapping']['portInstanceMapping'][ids[0]]['bufferInfo'])


@pytest.mark.requires_vitis
def test_strided_conv_matches_onnx_on_the_core(tmp_path):
    """Two stride-2 convs through aiesim, each behind a retiler: the first reads the boundary, an
    odd-width tensor whose rows start off the vector grid and whose 1800 bytes travel framed to
    1808; the second reads the first's two-block frame. Six different inputs, exact, and every
    hand-over but the boundary's is one buffer in shared memory."""
    feeds = np.random.default_rng(23).integers(-40, 40, size=(6, 1, 15, 15, 8), dtype=np.int8)
    assert_aie_matches_onnx(
        _strided_chain_model(size=15, first_stride=2, name='conv_strided_pair'),
        {'x_q': feeds},
        {},
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=6,
        per_iteration=True,
    )
    # A regression check on the compiler, not a gate the design relies on: each hand-over is one buffer.
    for writer, reader in (
        ('first_aie_retile.kk[0].out[0]', 'first_aie.kk[0].in[0]'),
        ('second_aie_retile.kk[0].out[0]', 'second_aie.kk[0].in[0]'),
        ('first_aie.kk[0].out[0]', 'second_aie_retile.kk[0].in[0]'),  # not required; placement put them side by side
    ):
        assert _compiled_buffers(tmp_path / 'proj', writer) == _compiled_buffers(tmp_path / 'proj', reader)


@pytest.mark.requires_vitis
def test_wide_pixel_retiler_matches_onnx_on_the_core(tmp_path):
    """Sixteen channels from the boundary, which the retiler moves four pixels a group. On AIE the
    13-column image starts two pixels into its first group and fills three of its last, so both
    carry border zeros, neither may read past its row, and the odd width leaves every other row off
    the 32-byte grid. Two chains of output channels read that one frame, multicast by DMA. Six
    different inputs through aiesim, exact."""
    feeds = np.random.default_rng(31).integers(-40, 40, size=(6, 1, 13, 13, 16), dtype=np.int8)
    assert_aie_matches_onnx(
        _strided_model(stride=2, k=3, channels_in=16, channels_out=16, size=13, pad=1, name='conv_wide_pixel'),
        {'x_q': feeds},
        {'b': {'parallelism': {'cas_num': 2}}},
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=6,
        per_iteration=True,
    )
    # The host measures latency from the first input beat, after configuration and weight loading.
    from aie4ml.report import report

    measured = report(tmp_path / 'proj')
    latency = measured['latency']
    assert 0 < latency['latency_cc'] < latency['first_output_from_sim_start_ns'] * measured['aie_clock_GHz']


@pytest.mark.requires_vitis
def test_row_split_strided_conv_matches_onnx_on_the_core(tmp_path):
    """Three row slices, each a cascade of two, through aiesim over six different inputs, exact: a
    retiler kernel per row and channel slice -- the first row slice's window opening on the top pad, the
    second channel slice's 4 channels not filling their block -- handing over by DMA, and the middle row slice's
    cascade on AIE's odd row, where it runs east to west."""
    feeds = np.random.default_rng(29).integers(-40, 40, size=(6, 1, 18, 18, 12), dtype=np.int8)
    assert_aie_matches_onnx(
        _row_split_strided_conv(channels_in=12),
        {'x_q': feeds},
        ROW_SPLIT_X_CASCADE,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=6,
        per_iteration=True,
    )


@pytest.mark.requires_vitis
def test_stream_conv_matches_onnx_on_the_core(tmp_path):
    """One shape on aiesim, over six different inputs, as the smoke test for what x86 cannot see.

    x86 loads unaligned addresses happily and schedules nothing, so alignment and pipelining
    failures pass there; both have reached the benchmark from a green suite. The shape is picked to
    touch what the x86 tests miss in one build: an input and an output channel count that each
    leave a partial block, a band whose bytes do not divide into beats, and a second inference over
    whatever the previous one left behind.
    """
    feeds = np.random.default_rng(15).integers(-40, 40, size=(6, 1, 6, 6, 4), dtype=np.int8)
    assert_aie_matches_onnx(
        _beat_carry_model(),
        {'x_q': feeds},
        STREAM_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=AIE1_PART,
        iterations=6,
        per_iteration=True,
    )


@pytest.mark.requires_vitis
def test_padded_conv_pair_matches_onnx_on_aie_mlv2(tmp_path):
    assert_x86_matches_onnx(
        _padded_pair_model(), {'x_q': _feed()}, {}, tmp_path, batch=1, frac=FRAC, max_code_diff=0, part=MLV2_PART
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_outer_split_matches_onnx(tmp_path, part):
    """Same-padded conv split by rows into two slices: the halo rows and the delivered top/bottom border
    are what this checks, so any mistake in the slice windows shows up as wrong pixels."""
    assert_x86_matches_onnx(
        _row_split_model(), {'x_q': _feed()}, ROW_SPLIT, tmp_path, batch=1, frac=FRAC, max_code_diff=1, part=part
    )


DW = 24  # three channel blocks
DEPTHWISE_ROWS = {'d1': {'parallelism': {'cas_num': 2, 'contract': 'outer'}}}


def _depthwise_model():
    """Every path of the channelwise core in one graph, two outputs of one input: a layer in two row bands, which read
    the graph input whose border the host delivers and write a graph output; and a chain whose depthwise layers
    re-border the kernel frames they read -- a valid 3x3 (windows on an even frame column), a same 3x3 (odd) and a
    same 5x5 (two tap groups a row on AIE-ML), over three channel blocks."""
    nodes: list = []
    inits: list = []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'd1', 8, 8, 3, pad=1, groups=8, relu=True, seed=1)
    nodes.append(helper.make_node('Transpose', ['a1'], ['y1'], perm=[0, 2, 3, 1], name='y1_nhwc'))
    _conv(nodes, inits, 'x_nchw', 'a2', 'c2', 8, DW, 3, pad=1, relu=True, seed=2)  # one window of the input
    _conv(nodes, inits, 'a2', 'a3', 'd3', DW, DW, 3, pad=0, groups=DW, relu=True, seed=3)
    _conv(nodes, inits, 'a3', 'a4', 'd4', DW, DW, 3, pad=1, groups=DW, relu=True, seed=4)
    _conv(nodes, inits, 'a4', 'a5', 'd5', DW, DW, 5, pad=2, groups=DW, relu=True, seed=5)
    _conv(nodes, inits, 'a5', 'a6', 'c6', DW, 8, 1, pad=0, relu=False, seed=6)
    nodes.append(helper.make_node('Transpose', ['a6'], ['y2'], perm=[0, 2, 3, 1], name='y2_nhwc'))
    return make_model(
        'depthwise',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, 8])],
        outputs=[('y1', TensorProto.FLOAT, [1, H, W, 8]), ('y2', TensorProto.FLOAT, [1, H - 2, W - 2, 8])],
        initializers=inits,
    )


@pytest.mark.parametrize('part', [AIE1_PART, PART, MLV2_PART], ids=['aie1', 'aie-ml', 'aie-mlv2'])
def test_depthwise_layers_take_the_channelwise_core_where_it_exists(tmp_path, part):
    # AIE1's mmul core holds the 5x5 as block-diagonal tiles, three blocks of which overflow a bank
    split = {'d5': {'parallelism': {'cas_num': 3}}} if part == AIE1_PART else {}
    ctx = lower(_depthwise_model(), tmp_path, {**DEPTHWISE_ROWS, **split}, part=part)
    for name in ('d1', 'd3', 'd4', 'd5'):
        inst = ctx.ir.execution.get(f'{name}_aie')
        assert inst.variant.uses_depthwise_core(inst.node, inst.config) == (part != AIE1_PART), name


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [PART, MLV2_PART], ids=['aie-ml', 'aie-mlv2'])
def test_depthwise_core_matches_onnx(tmp_path, part):
    feeds = np.random.default_rng(12).integers(-40, 40, size=(6, 1, H, W, 8), dtype=np.int8)
    assert_x86_matches_onnx(
        _depthwise_model(),
        {'x_q': feeds},
        DEPTHWISE_ROWS,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=6,
        per_iteration=True,
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART], ids=['aie1', 'aie-ml'])
def test_conv_chain_matches_onnx(conv_model, tmp_path, part):
    """One aiesim build covers the single-tile conv, a 3-chain output split, a three-tile cascade
    (first + middle + last, whose last tile owns the bias), a depthwise group and flatten -> Dense.
    The biases are non-zero and differ per channel, so a chain that dropped one would show up. The
    cascade sits on row 1, where AIE runs it east to west."""
    feeds = np.random.default_rng(11).integers(-40, 40, size=(6, 1, H, W, CIN), dtype=np.int8)
    directives = {**DIRECTIVES, 'c2': {**DIRECTIVES['c2'], 'placement': {'col': 9, 'row': 1}}}
    assert_aie_matches_onnx(
        conv_model,
        {'x_q': feeds},
        directives,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=6,
        per_iteration=True,
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', [AIE1_PART, PART, MLV2_PART], ids=['aie1', 'aie-ml', 'aie-mlv2'])
def test_channel_chains_of_a_flattened_conv_match_onnx(tmp_path, part):
    """Both heads of the sharded model through the DMAs that realize them, direct and through a memory tile; AIE1
    has no memory tile, so there `fb` takes a stage per chain too."""
    directives = SHARDED if part != AIE1_PART else {**SHARDED, 'fb': {'parallelism': {'cas_num': 1, 'cas_length': 2}}}
    feeds = np.random.default_rng(5).integers(-40, 40, size=(2, 1, H, W, CIN), dtype=np.int8)
    assert_aie_matches_onnx(
        _sharded_heads_model(),
        {'x_q': feeds},
        directives,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=2,
        per_iteration=True,
    )


# --------------------------------------------------------------------------- #
# int16 activations against int8 weights (AIE-ML and AIE-MLv2)
# --------------------------------------------------------------------------- #

INT16_PARTS = {'aie-ml': PART, 'aie-mlv2': MLV2_PART}
INT16_DIRECTIVES = {
    'c0': {'parallelism': {'cas_num': 1}},
    'c1': {'parallelism': {'cas_num': 2}},
    'c2': {'parallelism': {'cas_length': 2}},
}


def _int16_model():
    """Every path an int16 frame takes: an int8 conv writes it (c0), two chains split it (c1), a two-kernel cascade
    reads it and fuses ReLU and a 2x2 max pool (c2), and a Dense reads it flattened."""
    nodes, inits = [], []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a0', 'c0', CIN, 16, 3, pad=1, relu=True, seed=1, bits=16)
    _conv(nodes, inits, 'a0', 'a1', 'c1', 16, 16, 3, pad=1, relu=True, seed=2, bits=16)
    _conv(nodes, inits, 'a1', 'a2', 'c2', 16, 16, 3, pad=1, relu=True, seed=3, bits=16)
    inits += _qparams('p2o', frac=FRAC, elem_type=TensorProto.INT16)
    nodes.append(helper.make_node('MaxPool', ['a2'], ['p2_pool'], name='p2', kernel_shape=[2, 2], strides=[2, 2]))
    qdq(nodes, 'p2_pool', 'p2', 'p2o')
    _head(nodes, inits, 'p2', H // 2 * W // 2 * 16, seed=4)
    return make_model(
        'conv_int16',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, H, W, CIN])],
        outputs=[('y', TensorProto.FLOAT, [1, CLASSES])],
        initializers=inits,
        opset=21,  # int16 QuantizeLinear
    )


@pytest.mark.parametrize('part', INT16_PARTS.values(), ids=INT16_PARTS.keys())
def test_int16_frames_take_the_int16_core(tmp_path, part):
    ctx = lower(_int16_model(), tmp_path, INT16_DIRECTIVES, part=part)
    m16 = {PART: 2, MLV2_PART: 4}[part]
    for name, m in (('c0_aie', 2 * m16), ('c1_aie', m16), ('c2_aie', m16)):
        assert ctx.ir.execution.get(name).config.microtiling.microtile_m == m, name
    assert {'fused_pool', 'fused_activation', 'output_view'} <= set(ctx.ir.execution.get('c2_aie').node.traits)


def test_int16_conv_is_refused_where_no_core_takes_it(tmp_path):
    with pytest.raises(ConfigRefused, match='no conv2d variant matches'):
        lower(_int16_model(), tmp_path, INT16_DIRECTIVES, part=AIE1_PART)


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', INT16_PARTS.values(), ids=INT16_PARTS.keys())
def test_int16_conv_matches_onnx(tmp_path, part):
    feeds = np.random.default_rng(3).integers(-60, 60, size=(2, 1, H, W, CIN), dtype=np.int8)
    assert_x86_matches_onnx(
        _int16_model(),
        {'x_q': feeds},
        INT16_DIRECTIVES,
        tmp_path,
        batch=1,
        frac=FRAC,
        max_code_diff=0,
        part=part,
        iterations=2,
        per_iteration=True,
    )


@pytest.mark.parametrize('contract', ['inner', 'outer'])
def test_a_memory_tile_serves_only_the_readers_its_buffer_descriptors_cover(tmp_path, contract):
    """A flattened row read by a Dense cascade crosses a memory tile that zero-fills each reader's padded rows: 8 BDs
    a reader, from 24-BD pools the even and the odd channels each share. Four readers fit beside the writer; eight
    would need a relay stage."""
    nodes, inits = [], []
    _start(nodes, inits)
    _conv(nodes, inits, 'x_nchw', 'a1', 'c1', CIN, 8, 3, pad=1, relu=True, seed=1)
    _head(nodes, inits, 'a1', H * W * 8, seed=3)
    model = _model('bd_head', nodes, inits)
    held = {'c1': {'parallelism': {'cas_num': 1, 'cas_length': 1}}}
    split = {'cas_num': 1, 'contract': contract}
    ctx = lower(model, tmp_path / 'four', {**held, 'fc': {'parallelism': {**split, 'cas_length': 4}}})
    assert [
        len(b['readers']) for b in ctx.ir.physical.plan['buffers'] if b['writers'][0]['source'].startswith('c1')
    ] == [4]
    with pytest.raises(ConfigRefused, match=r'1 writers and 8 readers need \[36, 32\] BDs'):
        lower(model, tmp_path / 'eight', {**held, 'fc': {'parallelism': {**split, 'cas_length': 8}}})
