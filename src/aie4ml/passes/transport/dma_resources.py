"""Buffer descriptors a physical plan's DMAs use, counted from its finalized descriptors.

Pure functions: a descriptor, the DMA kind's `DmaSpec` and the device give an exact BD count, measured against the
AIE compiler's own allocation (Work/ps/c_rts/aie_control.cpp) on AIE1, AIE-ML and AIE-MLv2. Verification enforces
the budgets; a descriptor form outside what was measured is an error, never a guess.
"""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any, Dict, Optional, Sequence

from ...errors import ConfigRefused
from ...ir.context import DmaSpec

KERNEL_BUFFERS = 2  # a kernel's DMA-fed buffer port is double-buffered

# Generations whose AIE compiler (Vitis 2026.1.1) gives a kernel output's chunk of one 32-bit word no BD dimension of
# its own, so its BD walks one loop more; measured on AIE1 only (its inputs do not).
WORD_CHUNK_FOLDS = frozenset({'AIE'})


def tile_port_bds(
    descriptor: Optional[Dict[str, Any]], buffers: int, dma: DmaSpec, generation: str, element_bits: int
) -> int:
    """BDs a compute tile's DMA spends on one kernel buffer port, per buffer: one BD walks the contiguous chunk and
    `dma.dimensions - 1` loops more (the walked axes that repeat), or one more for a word chunk that folds (see
    WORD_CHUNK_FOLDS), and every loop beyond those repeats that BD. A linear transfer (`descriptor` None) is one BD.
    Measured on AIE1, AIE-ML and AIE-MLv2."""
    if descriptor is None:
        return buffers
    loops = [int(step['wrap']) for step in descriptor.get('tile_traversal') or () if int(step['wrap']) > 1]
    walked = dma.dimensions - 1
    if (
        generation in WORD_CHUNK_FOLDS
        and descriptor['access'] == 'write'
        and prod(int(x) for x in descriptor['tiling_dimension']) * int(element_bits) <= 32
    ):
        walked += 1
    return buffers * prod(loops[walked:])


def pool_use(port_bds: Sequence[int], dma: DmaSpec) -> list[list[int]]:
    """BDs per pool of each memory tile one direction of a buffer's ports takes. The ports sit on a tile's channels
    0, 1, 2, ... in order, and those past its channels on the next tile's, whose DMA reaches the buffer: the AIE
    compiler (Vitis 2026.1.1) places them so, measured on AIE-MLv2. Channel c draws from pool c % bd_pools."""
    tiles = []
    for first in range(0, len(port_bds), dma.channels):
        pools = [0] * dma.bd_pools
        for channel, bds in enumerate(port_bds[first : first + dma.channels]):
            pools[channel % dma.bd_pools] += bds
        tiles.append(pools)
    return tiles


# The most zero tiles a memory-tile BD folds into its padding fields, as the AIE compiler (Vitis 2026.1.1) programs
# it: the pad field's range, which the architecture manuals do not list.
MEMTILE_FOLD_TILES = {'AIE-ML': 15, 'AIE-MLV2': 63}
# The dimensions a memory-tile BD pads (XAie_PadDesc), on AIE-ML and AIE-MLv2.
MEMTILE_PAD_DIMENSIONS = 3


def memtile_port_bds(descriptor: Dict[str, Any], buffers: int, generation: str, where: str) -> int:
    """BDs a memory tile spends on one port of a shared buffer: two per run of its stream, per buffer.

    A port streams tiles in its traversal order, the first step fastest; `boundary_dimension` ends the data, and
    tiles past it carry zeros. A run is a stretch of consecutive tiles of one shape (the data each holds per axis),
    and one merge more:
      - a zero run right after a run of full tiles folds into it, as that BD's padding, when it spans at most
        MEMTILE_FOLD_TILES tiles and, on a walk of more BD dimensions than MEMTILE_PAD_DIMENSIONS (a 2-D tile
        stepped along both axes), the data run takes one step of the outermost traversal step, the dimension
        the zeros would pad.
    Measured on AIE-ML and AIE-MLv2, reads and writes alike: every 2-D port shape aie4ml generates, and unpadded
    walks of higher rank (a frame relayout); a padded walk of higher rank is refused as unmeasured.
    """
    tile = [int(v) for v in descriptor['tiling_dimension']]
    offset = [int(v) for v in descriptor['offset']]
    steps = [step for step in descriptor.get('tile_traversal') or () if int(step['wrap']) > 1]
    end = [o + t for o, t in zip(offset, tile)]
    for step in steps:
        end[int(step['dimension'])] += int(step['stride']) * (int(step['wrap']) - 1)
    boundary = [int(b) for b in descriptor.get('boundary_dimension') or end]
    if descriptor.get('boundary_offset'):
        raise ConfigRefused(f'{where}: a memory-tile walk with a boundary offset has no measured BD cost.')

    stride = {int(step['dimension']): int(step['stride']) for step in steps}

    def held(axis: int, index: int) -> int:  # the data one tile holds along an axis
        start = offset[axis] + index * stride.get(axis, 0)
        return max(0, min(boundary[axis] - start, tile[axis]))

    shapes = []  # the stream, first step fastest; None: an all-zero tile
    for indices in product(*(range(int(step['wrap'])) for step in reversed(steps))):
        at = dict(zip((int(step['dimension']) for step in reversed(steps)), indices))
        shape = tuple(held(axis, at.get(axis, 0)) for axis in range(len(tile)))
        shapes.append(shape if all(shape) else None)

    full = tuple(tile)
    if len(tile) > 2 and any(shape != full for shape in shapes):
        raise ConfigRefused(f'{where}: a padded rank-{len(tile)} memory-tile walk has no measured BD cost.')
    outer = prod(int(step['wrap']) for step in steps[:-1]) if len(tile) + len(steps) > MEMTILE_PAD_DIMENSIONS else 0
    # the tiles of one step of the outermost traversal step, where the BD cannot pad; 0 where it pads every step
    runs = []  # [shape, first tile, tiles]
    for index, shape in enumerate(shapes):
        if runs and runs[-1][0] == shape:
            runs[-1][2] += 1
        else:
            runs.append([shape, index, 1])
    folded = [runs[0]]
    for shape, first, count in runs[1:]:
        data, data_first, data_tiles = folded[-1]
        if (
            shape is None
            and data == full
            and count <= MEMTILE_FOLD_TILES[generation]
            and (not outer or data_first // outer == (data_first + data_tiles - 1) // outer)
        ):
            folded[-1] = [('folded',), data_first, data_tiles + count]
        else:
            folded.append([shape, first, count])
    return 2 * len(folded) * buffers
