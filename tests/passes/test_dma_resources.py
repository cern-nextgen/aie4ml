"""DMA buffer-descriptor accounting against counts the AIE compiler (Vitis 2026.1.1) allocated."""

import pytest
from aie4ml.device_catalog import resolve_device
from aie4ml.errors import ConfigRefused
from aie4ml.op_impls.utils import TraversalStep
from aie4ml.passes.transport.dma_resources import memtile_port_bds, pool_use, tile_port_bds

AIE_ML, AIE_MLV2 = 'AIE-ML', 'AIE-MLV2'


def _walk(tile, extent, boundary=None, offset=(0, 0)):
    """A 2-D memory-tile walk of `tile`-sized tiles over `extent`, the data ending at `boundary`."""
    traversal = [TraversalStep(d, t, e // t) for d, (t, e) in enumerate(zip(tile, extent)) if e > t]
    desc = {'tiling_dimension': list(tile), 'offset': list(offset), 'tile_traversal': traversal}
    if boundary is not None:
        desc['boundary_dimension'] = list(boundary)
    return desc


# (generation, walk, BDs the compiler allocated for the port's two buffers), each from a compiled project
MEASURED = [
    (AIE_ML, _walk((8, 4), (256, 8)), 4),  # a write: one run
    (AIE_ML, _walk((8, 4), (32, 8), (32, 1)), 8),  # one row of data, then a padded row of tiles
    (AIE_ML, _walk((8, 4), (32, 8), (32, 4)), 4),  # a padded row of 4 tiles folds into the data BD
    (AIE_ML, _walk((8, 4), (128, 8), (128, 4)), 8),  # ... but not 16 tiles on AIE-ML
    (AIE_MLV2, _walk((8, 8), (128, 16), (128, 8)), 4),  # ... which AIE-MLv2 folds (up to 63)
    (AIE_MLV2, _walk((8, 8), (640, 16), (640, 8)), 8),  # ... but not 80
    (AIE_ML, _walk((8, 4), (64, 8), (60, 8)), 16),  # a partial last tile on each row of tiles: 4 runs
    (AIE_ML, _walk((8, 4), (32, 8), (40, 8), offset=(32, 0)), 8),  # padded tiles after each row's data fold
    (AIE_MLV2, _walk((8, 8), (48, 16), (36, 8)), 12),  # full, partial, then zeros: 3 runs
    (AIE_ML, _walk((8, 4), (48, 8), (36, 8)), 24),  # the same on two rows of data: 6 runs, past a BD pool
    (AIE_ML, _walk((8, 4), (32, 40), (32, 36)), 8),  # a zero row of tiles after 9 data rows: no fourth pad dimension
    (AIE_MLV2, _walk((8, 4), (48, 48), (48, 44)), 8),  # ... nor on AIE-MLv2
]


def _frame_walk(cols, rows, offset=(0, 0, 0, 0), boundary=None):
    """A 4-D walk of a conv frame in a memory tile: 8-channel pixels along each row, row by row."""
    desc = {
        'tiling_dimension': [8, 1, 1, 1],
        'offset': list(offset),
        'tile_traversal': [
            TraversalStep(1, 1, cols),
            TraversalStep(2, 1, rows),
            TraversalStep(3, 1, 1),
            TraversalStep(0, 8, 1),
        ],
    }
    if boundary is not None:
        desc['boundary_dimension'] = list(boundary)
    return desc


# (generation, walk, BDs the compiler allocated for the port's two buffers), each from a compiled project
MEASURED_FRAMES = [
    (AIE_ML, _frame_walk(16, 8, offset=(0, 4, 1, 0)), 4),  # an image written into a frame's interior
    (AIE_ML, _frame_walk(24, 10), 4),  # the frame read whole
    (AIE_MLV2, _frame_walk(48, 18), 4),
]


@pytest.mark.parametrize('generation, walk, bds', MEASURED + MEASURED_FRAMES)
def test_a_memory_tile_port_costs_two_bds_per_run_of_its_stream(generation, walk, bds):
    assert memtile_port_bds(walk, 2, generation, 'port') == bds


def test_a_padded_walk_of_higher_rank_is_refused_as_unmeasured():
    with pytest.raises(ConfigRefused, match='no measured BD cost'):
        memtile_port_bds(_frame_walk(24, 10, boundary=(8, 16, 8, 1)), 2, AIE_ML, 'port')


def test_channels_draw_from_the_pool_of_their_parity():
    dma = resolve_device('xilinx_vek280_base_202610_1', {})[0].memtile_dma
    assert pool_use([4, 8, 8, 8, 8], dma) == [[20, 16]]


def test_ports_past_a_memory_tiles_channels_take_the_next_tiles():
    # The bf16 dense-chain probe (test_float_sim): an input buffer's 8 writers and 8 readers, 4 BDs each, took
    # BDs 0-47 of one memory tile and 0-3, 4-7, 24-27, 28-31 of the next.
    dma = resolve_device('vek385_base', {})[0].memtile_dma
    assert pool_use([4] * 8, dma) == [[12, 12], [4, 4]]


def test_a_compute_tile_port_repeats_its_bd_for_every_loop_past_the_dma_dimensions():
    dma = resolve_device('xilinx_vek280_base_202610_1', {})[0].tile_dma

    def walk(*wraps):
        return {'tile_traversal': [TraversalStep(d, 1, w) for d, w in enumerate(wraps)]}

    assert tile_port_bds(None, 2, dma, AIE_ML, 8) == 2  # linear
    assert tile_port_bds(walk(8, 4, 1), 2, dma, AIE_ML, 8) == 2  # the chunk and two loops: one BD per buffer (measured)
    assert tile_port_bds(walk(8, 4, 2), 2, dma, AIE_ML, 8) == 4  # a third loop of 2 repeats it (measured)


# (access, chunk of int8 elements, loops innermost first, BDs the AIE1 compiler allocated for the port's two buffers)
AIE1_MEASURED = [
    ('read', 8, (16, 2, 4), 16),  # the chunk and one loop per BD
    ('write', 8, (2, 2, 4), 16),
    ('write', 8, (1, 2, 4), 8),
    ('read', 4, (8, 4, 1), 8),  # an input's one-word chunk still takes a dimension
    ('write', 4, (16, 4, 1), 2),  # an output's does not: its BD walks both loops
]


@pytest.mark.parametrize('access, chunk, wraps, bds', AIE1_MEASURED)
def test_an_aie1_tile_port_walks_one_loop_per_bd_but_an_output_word_walks_two(access, chunk, wraps, bds):
    dma = resolve_device('xcvp2802-vsva5601-2MHP-e-S', {})[0].tile_dma
    walk = {
        'access': access,
        'tiling_dimension': [chunk, 1, 1],
        'tile_traversal': [TraversalStep(d, 1, w) for d, w in enumerate(wraps)],
    }
    assert tile_port_bds(walk, 2, dma, 'AIE', 8) == bds
