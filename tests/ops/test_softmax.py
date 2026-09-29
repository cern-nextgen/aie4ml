"""HCCS Softmax kernels against an integer oracle.

ONNX has no HCCS op and HCCS is a clipped-linear surrogate, so the reference is the exact
integer arithmetic the kernel implements: row max, clipped-linear score, floor-divide scale.
Covers the linear and tiled layouts (which must agree bit-for-bit across the 2x8/4x8/8x8
microtiles) and row splitting across cas_num, each packed into one graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aie4ml.frontends.onnx import from_onnx
from helpers import PART, TensorProto, dq, helper, lower, make_model, qdq, qparams

ROWS, COLS = 32, 64
B, S, DMAX, INV_SHIFT = 255, 2, 64, 15


def _hccs_oracle(x_i8: np.ndarray) -> np.ndarray:
    """Exact integer HCCS -> uint8 attention codes, reducing over the last axis."""
    x = np.asarray(x_i8, dtype=np.int16)
    delta = np.minimum(x.max(axis=-1, keepdims=True) - x, DMAX)
    score = B - S * delta
    rho = (255 << INV_SHIFT) // score.sum(axis=-1, keepdims=True, dtype=np.int32)
    return np.clip((score.astype(np.int32) * rho) >> INV_SHIFT, 0, 255).astype(np.uint8)


def _feed() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.clip(np.rint(rng.normal(0, 24, (ROWS, COLS))), -96, 96).astype(np.int8)


def _hccs(cas_num: int, layout: str | None = None, microtile: tuple[int, int] | None = None) -> dict:
    d: dict = {
        'approximation': 'hccs',
        'parallelism': {'cas_num': cas_num},
        'hccs': {'param_sets': 1, 'B': [B], 'S': [S], 'Dmax': [DMAX], 'inv_shift': INV_SHIFT, 'use_clb': False},
    }
    if layout:
        d['layout'] = layout
    if microtile:
        d['microtiling'] = {'microtile_m': microtile[0], 'microtile_n': microtile[1]}
    return d


def _softmax(nodes: list, out: str, prefix: str) -> None:
    nodes.append(helper.make_node('Softmax', ['x'], [f'{prefix}_sm'], name=prefix, axis=-1))
    qdq(nodes, f'{prefix}_sm', out, f'{prefix}o')


def _compile_and_check(model, directives, feed, tmp_path, project):
    aie_model = from_onnx(
        model,
        {'Part': PART, 'AIEConfig': {'Iterations': 1}, 'LayerDirectives': dict(directives)},
        output_dir=Path(tmp_path) / project,
        project_name=project,
    )
    aie_model.compile()
    got = aie_model.predict({'x_i8': feed}, simulator='x86', quantize_in=False, dequantize_out=False)
    if not isinstance(got, dict):
        got = {model.graph.output[0].name: got}
    expected = _hccs_oracle(feed)
    for name, arr in got.items():
        have = np.asarray(arr)[:ROWS].astype(np.uint8)
        assert np.array_equal(have, expected), f'{name}: max code diff {int(np.abs(have.astype(int) - expected).max())}'


# --------------------------------------------------------------------------- #
# layouts: linear and every tiled microtile agree with the oracle (and so each other)
# --------------------------------------------------------------------------- #

VARIANTS = {
    'lin': _hccs(2, layout='linear'),
    'til48': _hccs(2, layout='tiled', microtile=(4, 8)),
    'til28': _hccs(2, layout='tiled', microtile=(2, 8)),
    'til88': _hccs(2, layout='tiled', microtile=(8, 8)),
}


@pytest.fixture
def softmax_layouts():
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    for name in VARIANTS:
        _softmax(nodes, f'y_{name}', name)
    inits = [*qparams('x')]
    for name in VARIANTS:
        inits += qparams(f'{name}o', frac=8, unsigned=True)
    return make_model(
        'softmax_layouts',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[(f'y_{name}', TensorProto.FLOAT, [ROWS, COLS]) for name in VARIANTS],
        initializers=inits,
    )


@pytest.mark.requires_vitis
def test_softmax_layouts_match_oracle(softmax_layouts, tmp_path):
    """Linear and tiled (2x8/4x8/8x8) all reproduce the integer HCCS oracle bit-for-bit."""
    _compile_and_check(softmax_layouts, VARIANTS, _feed(), tmp_path, project='softmax_layouts')


# --------------------------------------------------------------------------- #
# row splitting across cas_num
# --------------------------------------------------------------------------- #


@pytest.fixture
def boundary_softmax():
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    _softmax(nodes, 'y_lin', 'lin')
    _softmax(nodes, 'y_til', 'til')
    inits = [*qparams('x'), *qparams('lino', frac=8, unsigned=True), *qparams('tilo', frac=8, unsigned=True)]
    return make_model(
        'boundary_softmax',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[('y_lin', TensorProto.FLOAT, [ROWS, COLS]), ('y_til', TensorProto.FLOAT, [ROWS, COLS])],
        initializers=inits,
    )


@pytest.mark.parametrize('cas_num', [1, 2, 4], ids=['1-tile', '2-tiles', '4-tiles'])
@pytest.mark.requires_vitis
def test_softmax_row_split_matches_oracle(boundary_softmax, tmp_path, cas_num):
    """Splitting the rows across tiles -- both layouts stay bit-exact to the oracle."""
    directives = {'lin': _hccs(cas_num, layout='linear'), 'til': _hccs(cas_num, layout='tiled')}
    _compile_and_check(boundary_softmax, directives, _feed(), tmp_path, project=f'sm_split_{cas_num}')


def test_softmax_rejects_noncanonical_uint8_scale(tmp_path):
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    _softmax(nodes, 'y', 'sm')
    model = make_model(
        'softmax_bad_output_scale',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[('y', TensorProto.FLOAT, [ROWS, COLS])],
        initializers=[*qparams('x'), *qparams('smo', frac=4, unsigned=True)],
    )

    with pytest.raises(ValueError, match=r'emits uint8 Q8 probabilities, got output frac=4'):
        lower(model, tmp_path, {'sm': {'layout': 'linear'}})


def test_softmax_existing_ml_accumulator_is_unchanged(boundary_softmax, tmp_path):
    directives = {'lin': {'layout': 'linear'}, 'til': {'layout': 'tiled'}}
    ctx = lower(boundary_softmax, tmp_path, directives)

    assert ctx.ir.execution.get('lin_aie').config.accumulator_tag == 'acc32'
    assert ctx.ir.execution.get('til_aie').config.accumulator_tag == 'acc32'


def test_softmax_prefers_tiled_and_honors_explicit_linear_layout(boundary_softmax, tmp_path):
    directives = {'lin': {'layout': 'linear'}, 'til': {'parallelism': {'cas_num': 1}}}
    ctx = lower(boundary_softmax, tmp_path, directives)

    assert ctx.ir.execution.get('lin_aie').variant.variant_id == 'softmax.exp.i8.v1'
    assert ctx.ir.execution.get('til_aie').variant.variant_id == 'softmax.exp.i8.tiled.v1'
