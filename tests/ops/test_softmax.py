"""HCCS Softmax kernels against an integer oracle.

ONNX has no HCCS op and HCCS is a clipped-linear surrogate, so the reference is the exact integer arithmetic the
kernel implements: row max, clipped-linear score, CLB or floor-divide scale. One x86 build covers both layouts, every
tiled microtile, both scales, the int16 output and row splitting across cas_num; one small aiesim build checks what
x86 cannot see, the hardware compiler's schedule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aie4ml.frontends.onnx import from_onnx
from helpers import PART, TensorProto, dq, helper, lower, make_model, qdq, qparams

ROWS, COLS = 32, 64
B, S, DMAX, INV_SHIFT = 255, 2, 64, 15


def _hccs_oracle(x_i8: np.ndarray, *, use_clb: bool, out16: bool) -> np.ndarray:
    """Integer HCCS over the last axis: uint8 Q8 or int16 Q15 codes."""
    x = np.asarray(x_i8, dtype=np.int64)
    score = B - S * np.minimum(x.max(axis=-1, keepdims=True) - x, DMAX)
    z = score.sum(axis=-1, keepdims=True)
    k = np.floor(np.log2(z)).astype(np.int64)
    if out16:
        return np.clip(score * ((32767 >> k) if use_clb else 32767 // z), -32768, 32767)
    rho = (255 << (INV_SHIFT - k)) if use_clb else (255 << INV_SHIFT) // z
    return np.clip((score * rho) >> INV_SHIFT, 0, 255)


def _feed(iterations: int, rows: int, cols: int) -> np.ndarray:
    """Random rows, and in each iteration an all -128 row and one whose gaps reach 255."""
    rng = np.random.default_rng(7)
    x = np.clip(np.rint(rng.normal(0, 24, (iterations, rows, cols))), -96, 96)
    x[:, 0] = -128
    x[:, 1] = np.where(np.arange(cols) % 2, 127, -128)
    return x.astype(np.int8)


def _hccs(cas_num: int, layout: str, microtile: tuple[int, int] | None = None, use_clb: bool = False) -> dict:
    d: dict = {
        'approximation': 'hccs',
        'layout': layout,
        'parallelism': {'cas_num': cas_num},
        'hccs': {'param_sets': 1, 'B': [B], 'S': [S], 'Dmax': [DMAX], 'inv_shift': INV_SHIFT, 'use_clb': use_clb},
    }
    if microtile:
        d['microtiling'] = {'microtile_m': microtile[0], 'microtile_n': microtile[1]}
    return d


def _softmax(nodes: list, out: str, prefix: str) -> None:
    nodes.append(helper.make_node('Softmax', ['x'], [f'{prefix}_sm'], name=prefix, axis=-1))
    qdq(nodes, f'{prefix}_sm', out, f'{prefix}o')


def _softmax_model(name: str, variants: dict, rows: int, cols: int):
    """One Softmax per variant, all on the same input; variants map name -> (directives, use_clb, int16 output)."""
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    inits = [*qparams('x')]
    for v, (_, _, out16) in variants.items():
        _softmax(nodes, f'y_{v}', v)
        if out16:
            inits += [
                helper.make_tensor(f'{v}o_scale', TensorProto.FLOAT, [], [2.0**-15]),
                helper.make_tensor(f'{v}o_zp', TensorProto.INT16, [], [0]),
            ]
        else:
            inits += qparams(f'{v}o', frac=8, unsigned=True)
    return make_model(
        name,
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [rows, cols])],
        outputs=[(f'y_{v}', TensorProto.FLOAT, [rows, cols]) for v in variants],
        initializers=inits,
    )


def _compile_and_check(variants: dict, feed: np.ndarray, tmp_path, project: str, simulator: str) -> None:
    iterations, rows, cols = feed.shape
    aie_model = from_onnx(
        _softmax_model(project, variants, rows, cols),
        {
            'Part': PART,
            'AIEConfig': {'Iterations': iterations},
            'LayerDirectives': {v: d for v, (d, _, _) in variants.items()},
        },
        output_dir=Path(tmp_path) / project,
        project_name=project,
    )
    if simulator == 'aie':
        aie_model.build('all')
    else:
        aie_model.compile()
    got = aie_model.predict(feed, simulator=simulator, quantize_in=False, dequantize_out=False, aie_profile=False)
    for v, (_, use_clb, out16) in variants.items():
        have = np.asarray(got[f'y_{v}']).astype(np.int64).reshape(feed.shape)
        want = _hccs_oracle(feed, use_clb=use_clb, out16=out16)
        assert np.array_equal(have, want), f'{v}: max code diff {int(np.abs(have - want).max())}'


VARIANTS = {
    'lin_e2': (_hccs(2, 'linear'), False, False),
    'lin_c4': (_hccs(4, 'linear', use_clb=True), True, False),
    'til28_e2': (_hccs(2, 'tiled', (2, 8)), False, False),
    'til48_e1': (_hccs(1, 'tiled', (4, 8)), False, False),
    'til48_c4': (_hccs(4, 'tiled', (4, 8), use_clb=True), True, False),
    'til88_c1': (_hccs(1, 'tiled', (8, 8), use_clb=True), True, False),
    'til48_c2_i16': (_hccs(2, 'tiled', (4, 8), use_clb=True), True, True),
}


@pytest.mark.requires_vitis
def test_softmax_variants_match_oracle(tmp_path):
    """Linear and tiled (2x8/4x8/8x8), exact and CLB, uint8 and int16, split over 1, 2 and 4 tiles, in one build."""
    _compile_and_check(VARIANTS, _feed(1, ROWS, COLS), tmp_path, 'softmax_variants', simulator='x86')


@pytest.mark.requires_vitis
def test_softmax_tiled_matches_oracle_on_aiesim(tmp_path):
    """Two small tiled Softmaxes through aiesim, two iterations: two row bands in straight-line code, and four in
    the pipelined band loops."""
    variants = {
        'til48_c1': (_hccs(1, 'tiled', (4, 8), use_clb=True), True, False),
        'til28_e1': (_hccs(1, 'tiled', (2, 8)), False, False),
    }
    _compile_and_check(variants, _feed(2, 8, 16), tmp_path, 'softmax_aiesim', simulator='aie')


# --------------------------------------------------------------------------- #
# lowering
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


def test_softmax_tiled_refuses_non_int8_microtile(boundary_softmax, tmp_path):
    with pytest.raises(ValueError, match=r'reads 2x8, 4x8 or 8x8 int8 microtiles'):
        lower(boundary_softmax, tmp_path, {'til': _hccs(1, 'tiled', (4, 16))})
