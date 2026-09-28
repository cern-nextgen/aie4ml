"""Buffer descriptors a physical plan's DMAs use, counted from its finalized descriptors.

Pure functions: a descriptor, the DMA kind's `DmaSpec` and the device give an exact BD count, measured against the
AIE compiler's own allocation (Work/ps/c_rts/aie_control.cpp) on AIE1, AIE-ML and AIE-MLv2. Verification enforces
the budgets; a descriptor form outside what was measured is an error, never a guess.
"""

from __future__ import annotations

from math import prod
from typing import Any, Dict, Optional, Sequence

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

    A port streams 2-D tiles, the tiles along axis 0 (contiguous) inside each row of tiles along axis 1;
    `boundary_dimension` ends the data, and tiles past it carry zeros. A run is a stretch of consecutive tiles of
    one shape (the data columns and rows each holds), across rows of tiles too, and one merge more:
      - a zero run right after a run of full tiles folds into it, as that BD's padding, when it spans at most
        MEMTILE_FOLD_TILES tiles.
    Measured on every AIE-ML and AIE-MLv2 port shape aie4ml generates, reads and writes alike.
    """
    tile = [int(v) for v in descriptor['tiling_dimension']]
    offset = [int(v) for v in descriptor['offset']]
    extent = list(tile)
    for step in descriptor.get('tile_traversal') or ():
        extent[int(step['dimension'])] = int(step['stride']) * int(step['wrap'])
    boundary = descriptor.get('boundary_dimension') or [o + e for o, e in zip(offset, extent)]
    if len(tile) != 2 or len(boundary) != 2:
        raise RuntimeError(f'{where}: a rank-{len(tile)} memory-tile walk has no measured BD cost.')

    def held(axis: int, index: int) -> int:  # the data one tile holds along an axis
        start = offset[axis] + index * tile[axis]
        return max(0, min(int(boundary[axis]) - start, tile[axis]))

    cols = [held(0, c) for c in range(extent[0] // tile[0])]
    rows = [held(1, r) for r in range(extent[1] // tile[1])]
    stream = [[(c, r) if c and r else None for c in cols] for r in rows]  # None: an all-zero tile

    full = (tile[0], tile[1])
    runs = []  # [shape, tiles]
    for shape in (shape for row in stream for shape in row):
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
