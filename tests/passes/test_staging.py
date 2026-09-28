"""Layout reaches transport as a staging descriptor, and descriptor equality makes an edge direct."""

from __future__ import annotations

import numpy as np
import pytest
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls.utils.tensor_view import microtile_from_staging
from helpers import (
    PART,
    TensorProto,
    config_of,
    direct_edges,
    dq,
    helper,
    lower,
    make_model,
    memtiles,
    microtiling,
    numpy_helper,
    output_staging,
    parallelism,
    qdq,
    qparams,
)

ROWS, COLS = 32, 64
AIE1_PART = 'xcvp2802-vsva5601-2MHP-e-S'


@pytest.fixture
def dense_then_layernorm():
    """x -> dense -> LayerNorm: a tiled producer and a consumer that may or may not match it."""
    nodes: list = []
    dq(nodes, 'x_i8', 'x', 'x')
    nodes.append(helper.make_node('DequantizeLinear', ['w_i8', 'w_scale', 'w_zp'], ['w_dq'], name='w_dq'))
    nodes.append(helper.make_node('MatMul', ['x', 'w_dq'], ['fc_out'], name='fc'))
    qdq(nodes, 'fc_out', 'fc_dq', 'fc')
    # EPS_Q0 = epsilon * 2^(2*frac) must be >= 1, so at frac=4 the floor is 2^-8.
    nodes.append(
        helper.make_node('LayerNormalization', ['fc_dq', 'gamma', 'beta'], ['ln_out'], name='ln', epsilon=2.0**-8)
    )
    qdq(nodes, 'ln_out', 'y', 'y')

    rng = np.random.default_rng(3)
    return make_model(
        'dense_then_layernorm',
        nodes=nodes,
        inputs=[('x_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[('y', TensorProto.FLOAT, [ROWS, COLS])],
        initializers=[
            *qparams('x'),
            *qparams('w'),
            *qparams('fc'),
            *qparams('y'),
            numpy_helper.from_array(rng.integers(-8, 8, size=(COLS, COLS), dtype=np.int8), 'w_i8'),
            numpy_helper.from_array(np.ones(COLS, np.float32), 'gamma'),
            numpy_helper.from_array(np.zeros(COLS, np.float32), 'beta'),
        ],
    )


def _lower(model, tmp_path, ln, fc=None):
    return lower(model, tmp_path, {'fc': fc or parallelism(4, contract='outer'), 'ln': ln}, part=PART, batch=ROWS)


def test_matmul_publishes_the_microtile_it_declares(dense_then_layernorm, tmp_path):
    """The intent a descriptor is built from and the tiling read back off it are one fact.

    If they drift, a consumer inherits a shape the producer never wrote and the edge quietly
    falls back to a memtile.
    """
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(4))
    declared = config_of(ctx, 'fc_aie').io_views['fc_out'].microtile
    assert declared == microtile_from_staging(output_staging(ctx, 'fc_aie'))


def test_a_linear_op_declares_no_microtile(dense_then_layernorm, tmp_path):
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(4) | {'layout': 'linear'})
    assert config_of(ctx, 'ln_aie').microtile is None
    assert microtile_from_staging(output_staging(ctx, 'ln_aie')) is None


def test_a_tiled_consumer_adopts_the_producers_microtile(dense_then_layernorm, tmp_path):
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(4) | {'layout': 'tiled'})
    assert config_of(ctx, 'ln_aie').microtile == config_of(ctx, 'fc_aie').io_views['fc_out'].microtile


def test_matching_layout_and_partition_hands_over_directly(dense_then_layernorm, tmp_path):
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(4) | {'layout': 'tiled'})
    assert ('fc_aie', 'ln_aie') in direct_edges(ctx)
    assert 'fc_out' not in memtiles(ctx)


def test_a_linear_consumer_of_tiled_data_needs_a_memtile(dense_then_layernorm, tmp_path):
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(4) | {'layout': 'linear'})
    assert ('fc_aie', 'ln_aie') not in direct_edges(ctx)
    assert 'fc_out' in memtiles(ctx)


def test_aie1_staging_mismatch_identifies_the_ports_and_descriptor_fields(dense_then_layernorm, tmp_path):
    with pytest.raises(
        ConfigRefused,
        match=r'staging mismatch at fc_aie\.out1\[0\] -> ln_aie\.in1\[0\] \([^)]*tiling_dimension',
    ):
        lower(
            dense_then_layernorm,
            tmp_path,
            {'fc': parallelism(4, contract='outer'), 'ln': parallelism(4) | {'layout': 'linear'}},
            part=AIE1_PART,
            batch=ROWS,
        )


def test_same_layout_is_not_enough_when_the_partition_differs(dense_then_layernorm, tmp_path):
    """Both tiled, same microtile, still a memtile -- the tile heights differ.

    This is why transport compares whole descriptors rather than a layout name.
    """
    ctx = _lower(dense_then_layernorm, tmp_path, parallelism(2) | {'layout': 'tiled'})
    assert config_of(ctx, 'ln_aie').microtile == config_of(ctx, 'fc_aie').io_views['fc_out'].microtile
    assert ('fc_aie', 'ln_aie') not in direct_edges(ctx)


def test_a_microtile_the_kernel_cannot_build_is_refused_at_resolve(dense_then_layernorm, tmp_path):
    with pytest.raises(ValueError, match=r'cannot normalise a 4x4 microtile'):
        _lower(
            dense_then_layernorm,
            tmp_path,
            parallelism(4) | {'layout': 'tiled'},
            fc=parallelism(4, contract='outer') | microtiling(4, 8, 4),
        )


# --------------------------------------------------------------------------- #
# ops with more than one activation input
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_denses_into_add():
    """Two independent dense chains meeting at an Add, so each input can be staged
    differently without the producers sharing a graph input."""
    nodes: list = []
    for w in ('a', 'b'):
        dq(nodes, f'x{w}_i8', f'x{w}', f'x{w}')
        nodes.append(
            helper.make_node('DequantizeLinear', [f'{w}_i8', f'{w}_scale', f'{w}_zp'], [f'{w}_w'], name=f'{w}_wdq')
        )
        nodes.append(helper.make_node('MatMul', [f'x{w}', f'{w}_w'], [f'{w}_out'], name=f'fc_{w}'))
        qdq(nodes, f'{w}_out', f'{w}_act', f'{w}o')
    nodes.append(helper.make_node('Add', ['a_act', 'b_act'], ['sum'], name='add'))
    qdq(nodes, 'sum', 'y', 'y')

    rng = np.random.default_rng(0)
    initializers = [
        *qparams('xa'),
        *qparams('xb'),
        *qparams('a'),
        *qparams('b'),
        *qparams('ao'),
        *qparams('bo'),
        *qparams('y'),
    ]
    for w in ('a', 'b'):
        initializers.append(numpy_helper.from_array(rng.integers(-8, 8, size=(COLS, COLS), dtype=np.int8), f'{w}_i8'))
    return make_model(
        'two_denses_into_add',
        nodes=nodes,
        inputs=[('xa_i8', TensorProto.INT8, [ROWS, COLS]), ('xb_i8', TensorProto.INT8, [ROWS, COLS])],
        outputs=[('y', TensorProto.FLOAT, [ROWS, COLS])],
        initializers=initializers,
    )


def _lower_add(model, tmp_path, fc_a, fc_b, add=None):
    directives = {'fc_a': fc_a, 'fc_b': fc_b, 'add': add or parallelism(4)}
    return lower(model, tmp_path, directives, part=PART, batch=ROWS)


def test_a_two_input_op_takes_both_inputs_directly_when_they_agree(two_denses_into_add, tmp_path):
    outer = parallelism(4, contract='outer')
    ctx = _lower_add(two_denses_into_add, tmp_path, outer, outer)
    assert {('fc_a_aie', 'add_aie'), ('fc_b_aie', 'add_aie')} <= direct_edges(ctx)
    assert not {'a_out', 'b_out'} & memtiles(ctx)


@pytest.mark.parametrize(
    'other', [parallelism(2, contract='outer'), parallelism(4, contract='inner')], ids=['row-split', 'contract']
)
def test_a_two_input_op_memtiles_only_the_input_that_disagrees(two_denses_into_add, tmp_path, other):
    """The decision is per edge, not per op: the matching producer still hands over directly
    while only the odd one out pays for a memtile."""
    ctx = _lower_add(two_denses_into_add, tmp_path, parallelism(4, contract='outer'), other)
    assert ('fc_a_aie', 'add_aie') in direct_edges(ctx)
    assert ('fc_b_aie', 'add_aie') not in direct_edges(ctx)
    assert 'b_out' in memtiles(ctx) and 'a_out' not in memtiles(ctx)


def test_a_consumer_adopts_the_split_its_producers_agree_on(two_denses_into_add, tmp_path):
    """Asking the Add for a different cas_num does not force a memtile -- it inherits the
    producers' partitioning instead, which is what keeps both edges direct."""
    outer = parallelism(4, contract='outer')
    ctx = _lower_add(two_denses_into_add, tmp_path, outer, outer, add=parallelism(2))
    assert config_of(ctx, 'add_aie').parallelism.cas_num == 4
    assert {('fc_a_aie', 'add_aie'), ('fc_b_aie', 'add_aie')} <= direct_edges(ctx)
