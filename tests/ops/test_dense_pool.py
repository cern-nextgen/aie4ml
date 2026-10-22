"""A pool over runs of rows, fused into the Dense producing them: the pooling of a model over sets (a per-element phi
chain, a ReduceMean over each group's T rows, a rho chain), exported either as a Reshape (G*T, C) -> (G, T, C) of 2-D
rows or straight from a 3-D MatMul, the pool between two 8-bit lanes.

The Dense writes each row's codes as it would unpooled, sums a group's codes and converts the sum once, a rounded right
shift by log2(T) + f_rows - f_pool for an average (f_rows - f_pool for a sum): what onnxruntime computes, exactly.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from helpers import (
    PART,
    TensorProto,
    assert_aie_matches_onnx,
    assert_x86_matches_onnx,
    direct_edges,
    helper,
    lower,
    make_model,
    microtiling,
    numpy_helper,
    parallelism,
    qdq,
    qparams,
)

from aie4ml.errors import ConfigRefused
from aie4ml.serialization import serialize_logical_node
from aie4ml.writer import kernel_config

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'
MLV2_PART = 'xc2ve3858-ssva2112-2MP-e-S'
# AIE1 tiles cannot re-tile a graph input wider than 8 bytes a row: the first phi Dense reads it as a stream.
AIE1_INPUT = {'phi0': {'ports': {'inputs': 'stream'}}}
FEATURES, F_IN, F_W = 16, 4, 6

# (phi widths, rho widths with the classes last, rows per group)
SHAPES = {
    'small': ((32, 32), (32, 16), 16),
    'paper': ((32, 32, 32), (32, 16), 64),
    'default': ((128, 128, 128), (128, 128, 16), 64),
}


def _layer_frac(codes, w, b, f_in, relu):
    """The output fraction that puts a layer's 99th-percentile code near 100, and those codes."""
    acc = codes @ w.astype(np.int64) + b
    top = float(np.percentile(np.abs(acc), 99)) or 1.0
    shift = max(0, int(np.ceil(np.log2(top / 100.0))))
    out = np.rint(acc / 2.0**shift).astype(np.int64)
    return f_in + F_W - shift, np.clip(out, 0, 255) if relu else np.clip(out, -128, 127)


def _set_model(
    groups,
    rows,
    phi,
    rho,
    *,
    seed=0,
    form='reshape',
    reduce='ReduceMean',
    keepdims=0,
    opset=17,
    pooled_unsigned=True,
    f_pool=None,
    finer=0,
    row_lane=True,
    relu_after_pool=False,
):
    """A model over sets and an input for it: x_q int8 (G*T, FEATURES) -- (G, T, FEATURES) for form '3d' -- through
    phi (Gemm, or MatMul + Add on 3-D rows, each ReLU'd into a uint8 lane), the reduction into the pooled lane and rho
    (the last layer an int8 lane without ReLU). Returns (model, feeds, output fraction)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (groups * rows, FEATURES)).astype(np.int8)
    nodes, inits = [], [*qparams('x', frac=F_IN)]
    shape = [groups, rows, FEATURES] if form == '3d' else [groups * rows, FEATURES]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'], name='x_dq'))
    codes, f_in, src = x.astype(np.int64), F_IN, 'x'

    def dense(name, src, codes, f_in, n, relu, lane=True):
        k = codes.shape[-1]
        w = rng.integers(-64, 65, (k, n)).astype(np.int8)
        b = rng.integers(-2000, 2001, n).astype(np.int32)
        f_out, out = _layer_frac(codes, w, b, f_in, relu)
        inits.extend([*qparams(f'{name}_w', frac=F_W), numpy_helper.from_array(w, f'{name}_w_q')])
        inits.extend(
            [
                helper.make_tensor(f'{name}_b_scale', TensorProto.FLOAT, [], [float(2.0 ** -(f_in + F_W))]),
                helper.make_tensor(f'{name}_b_zp', TensorProto.INT32, [], [0]),
                numpy_helper.from_array(b, f'{name}_b_q'),
            ]
        )
        nodes.append(
            helper.make_node('DequantizeLinear', [f'{name}_w_q', f'{name}_w_scale', f'{name}_w_zp'], [f'{name}_w'])
        )
        nodes.append(
            helper.make_node('DequantizeLinear', [f'{name}_b_q', f'{name}_b_scale', f'{name}_b_zp'], [f'{name}_b'])
        )
        if form == '3d' and name.startswith('phi'):
            nodes.append(helper.make_node('MatMul', [src, f'{name}_w'], [f'{name}_mm'], name=name))
            nodes.append(helper.make_node('Add', [f'{name}_mm', f'{name}_b'], [f'{name}_acc'], name=f'{name}_bias'))
        else:
            nodes.append(helper.make_node('Gemm', [src, f'{name}_w', f'{name}_b'], [f'{name}_acc'], name=name))
        out_name = f'{name}_acc'
        if relu:
            nodes.append(helper.make_node('Relu', [out_name], [f'{name}_relu'], name=f'{name}_relu'))
            out_name = f'{name}_relu'
        if not lane:
            return out_name, out, f_out
        inits.extend(qparams(f'{name}o', frac=f_out, unsigned=relu))
        qdq(nodes, out_name, f'{name}_out', f'{name}o')
        return f'{name}_out', out, f_out

    for i, n in enumerate(phi):
        last = i == len(phi) - 1
        src, codes, f_in = dense(f'phi{i}', src, codes, f_in, n, True, lane=row_lane or not last)

    width = phi[-1]
    if form == 'reshape':
        inits.append(numpy_helper.from_array(np.array([groups, rows, width], np.int64), 'pool_shape'))
        nodes.append(helper.make_node('Reshape', [src, 'pool_shape'], ['pool_in'], name='pool_reshape'))
        src = 'pool_in'
    sums = codes.reshape(groups, rows, width).sum(axis=1)
    top = float(np.percentile(sums, 99)) or 1.0
    log2_rows = int(rows).bit_length() - 1 if reduce == 'ReduceMean' else 0
    shift = max(0, int(np.ceil(np.log2(top / 100.0))))
    f_pool = f_in + log2_rows - shift + finer if f_pool is None else f_pool
    if opset >= 18 or reduce == 'ReduceSum':  # ReduceSum names its axes as an input from opset 13
        inits.append(numpy_helper.from_array(np.array([1], np.int64), 'pool_axes'))
        nodes.append(helper.make_node(reduce, [src, 'pool_axes'], ['pool_red'], name='pool', keepdims=keepdims))
    else:
        nodes.append(helper.make_node(reduce, [src], ['pool_red'], name='pool', axes=[1], keepdims=keepdims))
    src = 'pool_red'
    if keepdims:
        inits.append(numpy_helper.from_array(np.array([groups, width], np.int64), 'pooled_shape'))
        nodes.append(helper.make_node('Reshape', [src, 'pooled_shape'], ['pool_flat'], name='pool_flat'))
        src = 'pool_flat'
    if relu_after_pool:
        nodes.append(helper.make_node('Relu', [src], ['pool_relu'], name='pool_relu'))
        src = 'pool_relu'
    inits.extend(qparams('poolo', frac=f_pool, unsigned=pooled_unsigned))
    qdq(nodes, src, 'pool_out', 'poolo')
    codes = np.clip(np.rint(sums * 2.0 ** (f_pool - f_in - log2_rows)), 0, 255)
    src, f_in = 'pool_out', f_pool

    for i, n in enumerate(rho):
        last = i == len(rho) - 1
        src, codes, f_in = dense('out' if last else f'rho{i}', src, codes.astype(np.int64), f_in, n, not last)

    model = make_model(
        'set_model',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, shape)],
        outputs=[(src, TensorProto.FLOAT, [groups, rho[-1]])],
        initializers=inits,
        opset=opset,
    )
    return model, {'x_q': x.reshape(shape)}, f_in


def _constant(text: str, name: str) -> int:
    match = re.search(rf'constexpr int {name} = (-?\d+);', text)
    assert match, f'{name} is not in the parameter header'
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# lowering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'form',
    [{}, {'form': '3d'}, {'keepdims': 1}, {'opset': 18}],
    ids=['reshape of 2-D rows', 'MatMul on 3-D rows', 'keepdims', 'axes as input'],
)
def test_the_average_fuses_into_the_last_phi_dense(tmp_path, form):
    """Every export form lowers to the same Dense: it reads the 64 rows of each of 4 groups and writes one row per
    group, the rows at their own lane's shift and the groups' sums at log2(64) + f_rows - f_pool."""
    model, _, _ = _set_model(4, 64, (32, 32, 32), (32, 16), **form)
    ctx = lower(model, tmp_path, part=MLV2_PART)
    assert {node.op_type for node in ctx.ir.logical} == {'dense'}
    pooled = ctx.ir.execution.get('phi2_aie')
    assert pooled.variant.variant_id == 'dense.b.r.pool.v1'
    pool = pooled.node.attributes['fused_pool']
    assert (pool.kind, pool.kernel_shape) == ('average', (64,))
    assert tuple(pooled.node.outputs[0].shape) == (4, 32)
    row_frac = pool.input_precision.frac
    text = kernel_config(pooled)
    assert [_constant(text, n) for n in ('POOL_ROWS', 'POOL_GROUPS', 'POOL_EXTENT', 'POOL_SHIFT')] == [
        64,
        4,
        8,
        6 + row_frac - pooled.node.outputs[0].precision.frac,
    ]
    phi1_frac = ctx.ir.execution.get('phi1_aie').node.outputs[0].precision.frac
    assert pooled.config.shift == phi1_frac + F_W - row_frac
    assert 'POOL_ROWS' not in kernel_config(ctx.ir.execution.get('rho0_aie'))
    assert ('phi2_aie', 'rho0_aie') in direct_edges(ctx)
    # the report rates the Dense by the rows it reads, not the groups it writes
    assert serialize_logical_node(pooled.node)['macs'] == 4 * 64 * 32 * 32


def test_a_sum_shifts_by_the_lanes_only(tmp_path):
    model, _, _ = _set_model(4, 16, (32, 32), (32, 16), reduce='ReduceSum')
    pooled = lower(model, tmp_path, part=MLV2_PART).ir.execution.get('phi1_aie')
    pool = pooled.node.attributes['fused_pool']
    assert pool.kind == 'sum'
    shift = pool.input_precision.frac - pooled.node.outputs[0].precision.frac
    assert _constant(kernel_config(pooled), 'POOL_SHIFT') == shift


@pytest.mark.parametrize(
    'part, directives, blocks',
    [
        (MLV2_PART, {}, (4, 8)),
        (MLV2_PART, {'phi1': microtiling(4, 8, 8)}, (2, 4)),
        (PART, {}, (2, 4)),
        (AIE1_PART, AIE1_INPUT, (2, 2)),
    ],
    ids=['AIE-MLv2', 'AIE-MLv2 4x8x8', 'AIE-ML', 'AIE1'],
)
def test_the_kernel_steps_through_whole_groups(tmp_path, part, directives, blocks):
    """64 rows a group take the 4x2 block on AIE-MLv2 (32 rows a step), two 4-row microtiles a step on AIE-ML and two
    2-row ones on AIE1, whose selector product then sums 4 code rows."""
    model, _, _ = _set_model(2, 64, (32, 32), (32, 16))
    pooled = lower(model, tmp_path, directives, part=part).ir.execution.get('phi1_aie')
    assert (pooled.config.row_blocks, pooled.config.microtiling.microtile_m) == blocks


def test_groups_of_eight_rows_take_four_row_microtiles_on_aie_mlv2(tmp_path):
    """8 rows a group are less than a step of two 8-row microtiles: the Dense takes 4x8x8, a step of 8 rows."""
    model, _, _ = _set_model(8, 8, (32, 32), (32, 16))
    pooled = lower(model, tmp_path, part=MLV2_PART).ir.execution.get('phi1_aie')
    assert (pooled.config.row_blocks, pooled.config.microtiling.microtile_m) == (2, 4)


# --------------------------------------------------------------------------- #
# refused by name
# --------------------------------------------------------------------------- #


def test_groups_that_are_not_whole_steps_are_refused(tmp_path):
    model, _, _ = _set_model(8, 8, (32, 32), (32, 16))
    with pytest.raises(ConfigRefused, match=r'phi1_aie: its fused pool sums whole kernel steps .* a step 16'):
        lower(model, tmp_path, {'phi1': microtiling(8, 8, 8)}, part=MLV2_PART)


@pytest.mark.parametrize(
    'directive, message',
    [
        (parallelism(2, contract='outer'), r'phi1_aie: no dense variant matches'),
        ({'ports': 'stream'}, r"phi1_aie: no dense variant on stream input .* implements \['average_pool'"),
    ],
    ids=['row split', 'stream ports'],
)
def test_kernels_that_split_or_stream_the_rows_are_refused(tmp_path, directive, message):
    model, _, _ = _set_model(4, 16, (32, 32), (32, 16))
    with pytest.raises(ConfigRefused, match=message):
        lower(model, tmp_path, {'phi1': directive}, part=MLV2_PART)


@pytest.mark.parametrize(
    'variant, error, message',
    [
        ({'rows': 24}, NotImplementedError, r'phi1_aie: an average of 24 rows divides by 24, which is no power of two'),
        ({'f_pool': 9}, NotImplementedError, r'phi1_aie: its pooled rows have 9 fraction bits, more than the average'),
        ({'pooled_unsigned': False}, NotImplementedError, r'phi1_aie: .* differ in sign, rounding or saturation'),
        ({'row_lane': False}, ValueError, r'phi1_prebias: missing quantization intent'),
    ],
    ids=['average of 24 rows', 'pooled lane too fine', 'signed pooled lane', 'rows without a lane'],
)
def test_a_pool_the_lanes_do_not_define_exactly_is_refused(tmp_path, variant, error, message):
    rows = variant.pop('rows', 16)
    model, _, _ = _set_model(4, rows, (32, 32), (32, 16), **variant)
    with pytest.raises(error, match=message):
        lower(model, tmp_path, part=MLV2_PART)


def test_a_relu_after_an_average_is_not_fused_before_it(tmp_path):
    """relu(mean(x)) is not mean(relu(x)): the producer's ReLU runs before its pool, so this one stays, refused."""
    model, _, _ = _set_model(4, 16, (32, 32), (32, 16), relu_after_pool=True)
    with pytest.raises(NotImplementedError, match=r'pool_relu_aie: no kernel runs a relu on its own'):
        lower(model, tmp_path, part=MLV2_PART)


def _reduce_of_input(axes, keepdims=0):
    """A reduction straight over the int8 graph input (4, 16, 32)."""
    nodes, inits = [], [*qparams('x'), *qparams('r')]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x']))
    nodes.append(helper.make_node('ReduceMean', ['x'], ['m'], name='pool', axes=axes, keepdims=keepdims))
    qdq(nodes, 'm', 'y', 'r')
    out_shape = [d for i, d in enumerate([4, 16, 32]) if i not in [a % 3 for a in axes]]
    return make_model(
        'reduce',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [4, 16, 32])],
        outputs=[('y', TensorProto.FLOAT, out_shape)],
        initializers=inits,
    )


def test_a_pool_of_the_graph_input_is_refused(tmp_path):
    with pytest.raises(NotImplementedError, match=r'pool_aie: its average pool runs fused .* but the graph input'):
        lower(_reduce_of_input([1]), tmp_path, part=MLV2_PART)


@pytest.mark.parametrize('axes', [[2], [0]], ids=['the channels', 'the groups'])
def test_a_reduction_over_other_axes_is_refused(tmp_path, axes):
    with pytest.raises(NotImplementedError, match=rf'pool: ReduceMean over axes \[{axes[0]}\] of \(4, 16, 32\)'):
        lower(_reduce_of_input(axes), tmp_path, part=MLV2_PART)


def test_regrouped_rows_no_pool_reads_are_refused(tmp_path):
    """A Reshape (G*T, C) -> (G, T, C) the Dense writes is a view only a pool it fuses reads."""
    nodes, inits = [], [*qparams('x'), *qparams('w'), *qparams('h'), *qparams('y')]
    nodes.append(helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x']))
    inits.append(numpy_helper.from_array(np.ones((32, 32), np.int8), 'w_q'))
    nodes.append(helper.make_node('DequantizeLinear', ['w_q', 'w_scale', 'w_zp'], ['w']))
    nodes.append(helper.make_node('MatMul', ['x', 'w'], ['mm'], name='fc'))
    qdq(nodes, 'mm', 'h', 'h')
    inits.append(numpy_helper.from_array(np.array([4, 16, 32], np.int64), 'shape'))
    nodes.append(helper.make_node('Reshape', ['h', 'shape'], ['hr'], name='regroup'))
    nodes.append(helper.make_node('MatMul', ['hr', 'w'], ['mm2'], name='fc2'))
    qdq(nodes, 'mm2', 'y', 'y')
    model = make_model(
        'regroup',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [64, 32])],
        outputs=[('y', TensorProto.FLOAT, [4, 16, 32])],
        initializers=inits,
    )
    with pytest.raises(NotImplementedError, match=r'fc_aie: a Dense writes its rows regrouped only into a pool'):
        lower(model, tmp_path, part=MLV2_PART)


def test_a_pool_directed_on_its_own_is_refused(tmp_path):
    model, _, _ = _set_model(4, 16, (32, 32), (32, 16))
    with pytest.raises(ValueError, match=r'pool_aie: a fused pool runs inside phi1_aie; direct that layer'):
        lower(model, tmp_path, {'pool': parallelism(2)}, part=MLV2_PART)


# --------------------------------------------------------------------------- #
# x86 simulation, exact against onnxruntime
# --------------------------------------------------------------------------- #


@pytest.mark.requires_vitis
@pytest.mark.parametrize('groups', [1, 4, 8], ids=['J1', 'J4', 'J8'])
@pytest.mark.parametrize('shape', list(SHAPES))
def test_set_pooling_matches_onnx(tmp_path, shape, groups):
    phi, rho, rows = SHAPES[shape]
    model, feeds, frac = _set_model(groups, rows, phi, rho, seed=groups)
    assert_x86_matches_onnx(model, feeds, {}, tmp_path, frac=frac, max_code_diff=0, part=MLV2_PART)


SPLITS = {
    'K cascade': parallelism(1, cas_length=2),
    'N chains': parallelism(2),
    '2x2': parallelism(2, cas_length=2),
}


@pytest.mark.requires_vitis
@pytest.mark.parametrize('split', list(SPLITS))
def test_a_split_pooled_dense_matches_onnx(tmp_path, split):
    """A K cascade sums its rows at the last stage; N chains each sum their own columns."""
    model, feeds, frac = _set_model(4, 16, (64, 64), (64, 16), seed=3)
    assert_x86_matches_onnx(model, feeds, {'phi1': SPLITS[split]}, tmp_path, frac=frac, max_code_diff=0, part=MLV2_PART)


@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    'variant',
    [{'form': '3d', 'reduce': 'ReduceSum'}, {'rows': 8}, {'finer': 2}, {'keepdims': 1}],
    ids=['ReduceSum of 3-D rows', 'groups of 8 rows', 'saturating pooled lane', 'keepdims'],
)
def test_other_forms_match_onnx(tmp_path, variant):
    rows = variant.pop('rows', 16)
    model, feeds, frac = _set_model(8, rows, (32, 32), (32, 16), seed=5, **variant)
    assert_x86_matches_onnx(model, feeds, {}, tmp_path, frac=frac, max_code_diff=0, part=MLV2_PART)


@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    'part, directives',
    [(PART, {}), (PART, {'phi1': microtiling(4, 8, 4)}), (AIE1_PART, AIE1_INPUT)],
    ids=['AIE-ML', 'AIE-ML 4x8x4', 'AIE1'],
)
def test_other_generations_match_onnx(tmp_path, part, directives):
    """A selector product sums two 4-row microtiles on AIE-ML, four columns a sum row with 4x8x4, and a step of two
    2-row microtiles and four zero rows on AIE1."""
    model, feeds, frac = _set_model(4, 16, (32, 32), (32, 16), seed=6)
    assert_x86_matches_onnx(model, feeds, directives, tmp_path, frac=frac, max_code_diff=0, part=part)


# --------------------------------------------------------------------------- #
# aiesim, which x86 cannot stand in for
# --------------------------------------------------------------------------- #


@pytest.mark.requires_vitis
@pytest.mark.parametrize(
    'part, directives', [(MLV2_PART, {}), (PART, {}), (AIE1_PART, AIE1_INPUT)], ids=['AIE-MLv2', 'AIE-ML', 'AIE1']
)
def test_the_pooled_kernel_matches_onnx_on_aiesim(tmp_path, part, directives):
    model, feeds, frac = _set_model(4, 32, (32, 32), (32, 16), seed=7)
    assert_aie_matches_onnx(model, feeds, directives, tmp_path, frac=frac, max_code_diff=0, part=part)
