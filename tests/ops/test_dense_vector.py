"""Dense on a single sample: one row microtile per step (dense_vector.cpp)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    PART,
    TensorProto,
    assert_aie_matches_onnx,
    direct_edges,
    helper,
    lower,
    make_model,
    numpy_helper,
    qdq,
)

AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'
MLV2_PART = 'xc2ve3858-ssva2112-2mp-e-s'
PARTS = {'aie1': AIE1_PART, 'aie-ml': PART, 'aie-mlv2': MLV2_PART}
K, HIDDEN, CLASSES = 96, 32, 16


def _qparams(prefix: str, frac: int, elem_type: int = TensorProto.INT8) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', elem_type, [], [0]),
    ]


def _mlp(rows: int = 1):
    """`d0` with a bias and a ReLU, `d1` with a ReLU only, `d2` with neither."""
    rng = np.random.default_rng(7)
    shapes = {'d0': (K, HIDDEN), 'd1': (HIDDEN, HIDDEN), 'd2': (HIDDEN, CLASSES)}
    inits = [*_qparams('x', 4)]
    nodes = [helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zp'], ['x'])]
    src = 'x'
    for name, (k, n) in shapes.items():
        inits += [numpy_helper.from_array(rng.integers(-8, 8, size=(k, n), dtype=np.int8), f'{name}_w_q')]
        inits += [*_qparams(f'{name}_w', 4), *_qparams(f'{name}o', 3)]
        nodes.append(
            helper.make_node('DequantizeLinear', [f'{name}_w_q', f'{name}_w_scale', f'{name}_w_zp'], [f'{name}_w'])
        )
        if name == 'd0':
            inits += [numpy_helper.from_array(rng.integers(-256, 256, size=(n,), dtype=np.int32), 'd0_b_q')]
            inits += _qparams('d0_b', 8, TensorProto.INT32)
            nodes.append(helper.make_node('DequantizeLinear', ['d0_b_q', 'd0_b_scale', 'd0_b_zp'], ['d0_b']))
            nodes.append(helper.make_node('Gemm', [src, 'd0_w', 'd0_b'], ['d0_mm'], name=name))
        else:
            nodes.append(helper.make_node('MatMul', [src, f'{name}_w'], [f'{name}_mm'], name=name))
        pre_q = f'{name}_mm'
        if name != 'd2':
            nodes.append(helper.make_node('Relu', [pre_q], [f'{name}_relu'], name=f'{name}_relu'))
            pre_q = f'{name}_relu'
        src = 'y' if name == 'd2' else f'{name}_out'
        qdq(nodes, pre_q, src, f'{name}o')
    return make_model(
        'dense_vector_mlp',
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [rows, K])],
        outputs=[('y', TensorProto.FLOAT, [rows, CLASSES])],
        initializers=inits,
    )


# Two chains of a first, a middle and a last kernel, a first and a last kernel reading one chain each, then a
# single kernel.
SPLIT = {
    'd0': {'parallelism': {'cas_num': 2, 'cas_length': 3}},
    'd1': {'parallelism': {'cas_num': 1, 'cas_length': 2}},
    'd2': {'parallelism': {'cas_num': 1, 'cas_length': 1}},
}


@pytest.mark.parametrize('part, rows', [(AIE1_PART, 2), (PART, 2), (MLV2_PART, 4)], ids=PARTS.keys())
def test_one_sample_takes_one_row_block(tmp_path, part, rows):
    """The fewest rows of a microtile a Dense of the same microtile reads directly: the chain stays direct. A batch
    keeps the 2x2 kernel."""
    ctx = lower(_mlp(), tmp_path / 'one', SPLIT, part=part)
    for name in ('d0_aie', 'd1_aie', 'd2_aie'):
        inst = ctx.ir.execution.get(name)
        assert inst.variant.variant_id == 'dense.b.r.vector.v1'
        assert inst.config.microtiling.microtile_m == rows
        assert inst.port_views[inst.inputs[0].tensor].full[0] == rows
    assert {('d0_aie', 'd1_aie'), ('d1_aie', 'd2_aie')} <= direct_edges(ctx) and ctx.ir.physical.plan['buffers'] == []

    batch = lower(_mlp(rows=8), tmp_path / 'eight', SPLIT, part=part)
    assert {inst.variant.variant_id for inst in batch.ir.execution} == {'dense.b.r.v1'}


@pytest.mark.requires_vitis
@pytest.mark.parametrize('part', PARTS.values(), ids=PARTS.keys())
def test_one_sample_matches_onnx(tmp_path, part):
    feeds = np.random.default_rng(9).integers(-60, 60, size=(4, 1, K), dtype=np.int8)
    assert_aie_matches_onnx(
        _mlp(),
        {'x_q': feeds},
        SPLIT,
        tmp_path,
        frac=3,
        max_code_diff=0,
        part=part,
        iterations=4,
        per_iteration=True,
    )
