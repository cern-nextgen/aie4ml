"""LayerNorm kernels: arithmetic against onnxruntime, decisions against the lowered IR."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    TensorProto,
    assert_x86_matches_onnx,
    config_of,
    direct_edges,
    dq,
    helper,
    lower,
    make_model,
    memtiles,
    numpy_helper,
    parallelism,
    qdq,
    qparams,
)

ROWS, COLS = 16, 64
EPS = 2.0**-8  # smallest epsilon with EPS_Q0 = eps * 2^(2*frac) >= 1 at frac=4


def _weight(name, seed):
    return numpy_helper.from_array(np.random.default_rng(seed).integers(-4, 4, size=(COLS, COLS), dtype=np.int8), name)


def _layernorm(nodes, src, out, prefix, gamma='gamma', beta='beta'):
    nodes.append(helper.make_node('LayerNormalization', [src, gamma, beta], [f'{prefix}_ln'], name=prefix, epsilon=EPS))
    qdq(nodes, f'{prefix}_ln', out, f'{prefix}o')


def _dense(nodes, src, out, prefix):
    nodes.append(
        helper.make_node(
            'DequantizeLinear',
            [f'{prefix}_wi8', f'{prefix}_w_scale', f'{prefix}_w_zp'],
            [f'{prefix}_w'],
            name=f'{prefix}_wdq',
        )
    )
    nodes.append(helper.make_node('MatMul', [src, f'{prefix}_w'], [f'{prefix}_mm'], name=prefix))
    qdq(nodes, f'{prefix}_mm', out, f'{prefix}o')


def _ln_params():
    return [
        numpy_helper.from_array(np.ones(COLS, np.float32), 'gamma'),
        numpy_helper.from_array(np.zeros(COLS, np.float32), 'beta'),
    ]


def _feeds():
    rng = np.random.default_rng(7)
    return {'x_i8': np.clip(np.rint(rng.normal(0, 24, (ROWS, COLS))), -96, 96).astype(np.int8)}


# --------------------------------------------------------------------------- #
# x86: kernel arithmetic
# --------------------------------------------------------------------------- #

LAYOUTS = {'lin_dense': 'linear', 'til_dense': 'tiled', 'til_chain_ln': 'tiled'}
DENSES = ('cd', 'dd', 'ch0', 'ch1')


@pytest.fixture
def layernorm_positions():
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')

    _layernorm(nodes, 'x', 'y_lin_edge', 'lin_edge')  # boundary
    _layernorm(nodes, 'x', 'y_til_edge', 'til_edge')

    _dense(nodes, 'x', 'cd_dq', 'cd')  # after a dense
    _layernorm(nodes, 'cd_dq', 'y_lin_dense', 'lin_dense')
    _dense(nodes, 'x', 'dd_dq', 'dd')
    _layernorm(nodes, 'dd_dq', 'y_til_dense', 'til_dense')

    _dense(nodes, 'x', 'ch0_dq', 'ch0')  # dense -> tiled LN -> dense: LN as a direct producer
    _layernorm(nodes, 'ch0_dq', 'ch_ln_dq', 'til_chain_ln')
    _dense(nodes, 'ch_ln_dq', 'y_til_chain', 'ch1')

    inits = [*qparams('x')]
    for p in ('lin_edgeo', 'til_edgeo', 'lin_denseo', 'til_denseo', 'til_chain_lno', 'cdo', 'ddo', 'ch0o', 'ch1o'):
        inits += qparams(p)
    for i, p in enumerate(DENSES):
        inits += qparams(f'{p}_w')
        inits.append(_weight(f'{p}_wi8', seed=i + 1))
    inits += _ln_params()
    out_names = ('y_lin_edge', 'y_til_edge', 'y_lin_dense', 'y_til_dense', 'y_til_chain')
    return make_model(
        'layernorm_positions',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[(n, TensorProto.FLOAT, [ROWS, COLS]) for n in out_names],
        initializers=inits,
    )


@pytest.mark.requires_vitis
def test_layernorm_positions_match_onnx(layernorm_positions, tmp_path):
    """Both variants at a boundary, after a dense, and (tiled) feeding a dense -- all direct."""
    edge = {'lin_edge': parallelism(1) | {'layout': 'linear'}, 'til_edge': parallelism(1) | {'layout': 'tiled'}}
    inner = {node: parallelism(1) | {'layout': layout} for node, layout in LAYOUTS.items()}
    denses = {name: parallelism(1, contract='outer') for name in DENSES}
    assert_x86_matches_onnx(layernorm_positions, _feeds(), {**edge, **inner, **denses}, tmp_path)


# --------------------------------------------------------------------------- #
# row splitting across cas_num
# --------------------------------------------------------------------------- #


@pytest.fixture
def boundary_layernorms():
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    _layernorm(nodes, 'x', 'y_lin', 'lin')
    _layernorm(nodes, 'x', 'y_til', 'til')
    inits = [*qparams('x'), *qparams('lino'), *qparams('tilo'), *_ln_params()]
    return make_model(
        'boundary_layernorms',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[('y_lin', TensorProto.FLOAT, [ROWS, COLS]), ('y_til', TensorProto.FLOAT, [ROWS, COLS])],
        initializers=inits,
    )


@pytest.mark.requires_vitis
@pytest.mark.parametrize('cas_num', [1, 2, 4], ids=['1-tile', '2-tiles', '4-tiles'])
def test_layernorm_row_split_matches_onnx(boundary_layernorms, tmp_path, cas_num):
    """Splitting the rows across tiles -- the path where the batch/statistics-vector sizing lives."""
    directives = {'lin': parallelism(cas_num) | {'layout': 'linear'}, 'til': parallelism(cas_num) | {'layout': 'tiled'}}
    assert_x86_matches_onnx(boundary_layernorms, _feeds(), directives, tmp_path)


# --------------------------------------------------------------------------- #
# graph level: compiler decisions, no Vitis
# --------------------------------------------------------------------------- #


def _edge_model(nodes, inits, outputs):
    return make_model(
        'layernorm_edge',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[(n, TensorProto.FLOAT, [ROWS, COLS]) for n in outputs],
        initializers=inits,
    )


def _boundary_model(beta=None):
    """x -> LayerNorm -> out."""
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    _layernorm(nodes, 'x', 'y', 'ln')
    inits = [*qparams('x'), *qparams('lno')]
    inits.append(numpy_helper.from_array(np.ones(COLS, np.float32), 'gamma'))
    inits.append(numpy_helper.from_array(np.zeros(COLS, np.float32) if beta is None else beta, 'beta'))
    return _edge_model(nodes, inits, ('y',))


def _after_dense_model():
    """x -> Dense -> LayerNorm -> out; the LN input carries the dense's staging."""
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    _dense(nodes, 'x', 'd_dq', 'd')
    _layernorm(nodes, 'd_dq', 'y', 'ln')
    inits = [*qparams('x'), *qparams('do'), *qparams('lno'), *qparams('d_w')]
    inits.append(_weight('d_wi8', seed=3))
    inits += _ln_params()
    return _edge_model(nodes, inits, ('y',))


@pytest.mark.parametrize('cas_num', [1, 4])
def test_boundary_layernorm_splits_rows_across_tiles(tmp_path, cas_num):
    ctx = lower(_boundary_model(), tmp_path, {'ln': parallelism(cas_num)})
    cfg = config_of(ctx, 'ln_aie')
    assert int(cfg.parallelism.cas_num) == cas_num
    assert int(cfg.cols) == COLS  # reduces over features, so every tile owns whole rows
    assert int(cfg.rows) * cas_num == ROWS


@pytest.mark.parametrize('cas_num', [1, 4])
def test_dense_to_layernorm_only_reshards_a_partitioned_edge(tmp_path, cas_num):
    """A single tiled producer connects directly; an inner split must still be re-sharded."""
    directives = {'d': parallelism(cas_num, contract='inner'), 'ln': parallelism(cas_num)}
    ctx = lower(_after_dense_model(), tmp_path, directives)
    if cas_num == 1:
        assert ('d_aie', 'ln_aie') in direct_edges(ctx)
        assert not any('d_mm' in str(t) for t in memtiles(ctx))
    else:
        assert ('d_aie', 'ln_aie') not in direct_edges(ctx)
        assert any('d_mm' in str(t) for t in memtiles(ctx))


def test_boundary_layernorm_prefers_the_tiled_variant(tmp_path):
    ctx = lower(_boundary_model(), tmp_path, {'ln': parallelism(1)})
    inst = ctx.ir.execution.get('ln_aie')
    assert inst.variant.variant_id == 'layer_norm.i8.tiled.v1'
    assert inst.config.accumulator_tag == 'acc32'


def test_boundary_layernorm_honors_explicit_linear_layout(tmp_path):
    ctx = lower(_boundary_model(), tmp_path, {'ln': {**parallelism(1), 'layout': 'linear'}})
    assert ctx.ir.execution.get('ln_aie').variant.variant_id == 'layer_norm.i8.v1'


def test_layernorm_rejects_unrepresentable_beta(tmp_path):
    beta = np.full((COLS,), 1.25, dtype=np.float32)
    with pytest.raises(ValueError, match="LayerNorm parameter 'beta' cannot be represented"):
        lower(_boundary_model(beta=beta), tmp_path, {'ln': parallelism(1)})
