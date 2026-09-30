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


def tile_port_bds(descriptor: Optional[Dict[str, Any]], buffers: int, dma: DmaSpec) -> int:
    """BDs a compute tile's DMA spends on one kernel buffer port, per buffer: one BD walks the contiguous chunk and
    `dma.dimensions - 1` loops more (the walked axes that repeat), and every loop beyond those repeats that BD.
    A linear transfer (`descriptor` None) is one BD. Measured on AIE-ML and AIE-MLv2."""
    wraps = [int(step['wrap']) for step in (descriptor or {}).get('tile_traversal') or ()]
    loops = [wrap for wrap in wraps if wrap > 1]
    return buffers * prod(loops[dma.dimensions - 1 :])


def pool_use(port_bds: Sequence[int], dma: DmaSpec) -> list[int]:
    """BDs per pool when the ports sit on channels 0, 1, 2, ... in order: channel c draws from pool c % bd_pools."""
    pools = [0] * dma.bd_pools
    for channel, bds in enumerate(port_bds):
        pools[channel % dma.bd_pools] += bds
    return pools


# The most zero tiles a memory-tile BD folds into its padding fields, as the AIE compiler (Vitis 2026.1.1) programs
# it: the pad field's range, which the architecture manuals do not list.
MEMTILE_FOLD_TILES = {'AIE-ML': 15, 'AIE-MLV2': 63}


def memtile_port_bds(descriptor: Dict[str, Any], buffers: int, generation: str, where: str) -> int:
    """BDs a memory tile spends on one port of a shared buffer: two per run of its stream, per buffer.

    A port streams tiles in its traversal order, the first step fastest; `boundary_dimension` ends the data, and
    tiles past it carry zeros. A run is a stretch of consecutive tiles of one shape (the data each holds per axis),
    and one merge more:
      - a zero run right after a run of full tiles folds into it, as that BD's padding, when it spans at most
        MEMTILE_FOLD_TILES tiles.
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
    runs = []  # [shape, tiles]
    for shape in shapes:
        if runs and runs[-1][0] == shape:
            runs[-1][1] += 1
        else:
            runs.append([shape, 1])
    folded = [runs[0]]
    for shape, count in runs[1:]:
        if shape is None and folded[-1][0] == full and count <= MEMTILE_FOLD_TILES[generation]:
            folded[-1] = [('folded',), folded[-1][1] + count]
        else:
            folded.append([shape, count])
    return 2 * len(folded) * buffers
