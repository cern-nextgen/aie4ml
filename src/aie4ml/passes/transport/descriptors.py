from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import ConfigRefused
from ...op_impls.utils import (
    STORAGE_LAYOUT_INNER_BLOCKED,
    STORAGE_LAYOUT_LINEAR,
    StagingDescriptor,
    TraversalStep,
)

BD_MAX_WRAP = 255
"""Steps one buffer-descriptor dimension can take (measured: Vitis rejects 720)."""

BD_WORD_BITS = 32
"""A buffer descriptor counts 32-bit words."""


def describes_natural_order(desc: Dict[str, Any]) -> bool:
    """Whether a descriptor just moves its whole buffer, in the order the buffer already has.

    Such a transfer needs no access pattern on the kernel port: constraining it only forces the
    DMA into buffer descriptors that a plain linear transfer does not need.
    """
    dims = [int(value) for value in desc['buffer_dimension']]
    chunk = [int(value) for value in desc['tiling_dimension']]
    if any(int(value) != 0 for value in desc['offset']):
        return False
    walked = {step.dimension: step for step in desc.get('tile_traversal', ())}
    for dim, extent in enumerate(dims):
        step = walked.get(dim)
        if step is None:
            if chunk[dim] != extent:
                return False
        elif step.wrap * chunk[dim] != extent:
            return False
        elif step.wrap > 1 and step.stride != 1:  # one tile never uses its stride
            return False
    return sorted(walked) == list(walked)


def boundary_access_descriptor(
    base: StagingDescriptor, *, element_bits: int, project_to_io_boundary: bool = False
) -> Tuple[Dict[str, Any], Optional[List[int]]]:
    """Lower a staging layout to an ADF graph-boundary access descriptor, and the shape the PLIO carries per
    iteration where the walk projects the buffer onto its logical rows.

    `element_bits` is the port's element width: a storage layout says how a buffer is arranged,
    not what it holds, while a buffer descriptor counts words.
    """
    if base.storage_layout == STORAGE_LAYOUT_LINEAR:
        return base.to_dict(), None
    if base.storage_layout == STORAGE_LAYOUT_INNER_BLOCKED:
        if project_to_io_boundary:
            raise ConfigRefused('A blocked frame is transferred whole; the host trims it.')
        return _inner_blocked_access_descriptor(base, int(element_bits)), None

    dimensions, tile = list(base.buffer_dimension), list(base.tiling_dimension)
    if len(dimensions) != 2:
        raise ConfigRefused('Microtiled boundary DMA requires a 2-D kernel buffer.')

    inner, outer = base.inner_dimension, base.outer_dimension
    microtile_inner = tile[inner]
    microtile_outer = tile[outer]
    if inner != 0 or outer != 1 or any(base.offset):
        raise ConfigRefused('Microtiled boundary DMA requires a local, inner-contiguous buffer.')
    if microtile_outer == 1 and not project_to_io_boundary:
        return base.to_dict(), None

    inner_extent, outer_extent = dimensions
    if inner_extent % microtile_inner or outer_extent % microtile_outer:
        raise ValueError(
            f'Direct boundary shape {(outer_extent, inner_extent)} '
            f'is not divisible by microtile {(microtile_outer, microtile_inner)}.'
        )

    # Rows walked per microtile and microtile rows walked: the whole padded buffer, or, when the
    # transfer is projected to the IO boundary, only the logical rows (a partial first microtile
    # or whole microtiles) and the logical columns rounded up to the microtile; the host trims.
    transfer_inner_extent, rows_per_tile, tile_rows = inner_extent, microtile_outer, outer_extent // microtile_outer
    if project_to_io_boundary:
        io_inner, io_outer = base.io_boundary_dimension
        if io_inner > inner_extent or io_outer > outer_extent:
            raise ValueError(
                f'Direct boundary shape {(outer_extent, inner_extent)} cannot contain IO boundary '
                f'{(io_outer, io_inner)}.'
            )
        if io_outer > microtile_outer and io_outer % microtile_outer:
            raise ConfigRefused(
                f'Direct boundary DMA cannot project {io_outer} rows from microtile rows of {microtile_outer}; '
                'rows must fill whole microtiles or fewer than one.'
            )
        transfer_inner_extent = -(-io_inner // microtile_inner) * microtile_inner
        rows_per_tile, tile_rows = min(io_outer, microtile_outer), -(-io_outer // microtile_outer)

    microtile_elements = microtile_outer * microtile_inner
    descriptor = base.to_dict()
    descriptor['buffer_dimension'] = [
        microtile_elements,
        inner_extent // microtile_inner,
        outer_extent // microtile_outer,
    ]
    descriptor['tiling_dimension'] = [microtile_inner, 1, 1]
    descriptor['offset'] = [0, 0, 0]
    descriptor['tile_traversal'] = [
        TraversalStep(1, 1, transfer_inner_extent // microtile_inner),
        TraversalStep(0, microtile_inner, rows_per_tile),
        TraversalStep(2, 1, tile_rows),
    ]
    descriptor.pop('boundary_dimension', None)
    return descriptor, [transfer_inner_extent, rows_per_tile * tile_rows]


def _inner_blocked_access_descriptor(base: StagingDescriptor, element_bits: int) -> Dict[str, Any]:
    """Boundary DMA for an inner-blocked buffer ([c/B][...][B] memory, linear wire order).

    One block of the inner axis is one contiguous region whose wire order is already its memory
    order, so it transfers as a single tile. A port spanning several blocks would need the DMA to
    interleave them per element, which this lowering does not implement (measured on AIE1: such a
    walk needs 20 S2MM descriptors against a budget of 16).
    """
    descriptor = base.to_dict()
    dims = list(base.buffer_dimension)
    inner = base.inner_dimension
    block = base.tiling_dimension[inner]
    wraps = {step.dimension: step.wrap for step in base.tile_traversal}
    if any(value for dim, value in enumerate(base.offset) if dim != inner):
        raise ConfigRefused('Inner-blocked boundary DMA transfers whole non-inner axes.')
    if block * wraps.get(inner, 1) != block:
        raise ConfigRefused(
            f'Inner-blocked boundary staging spans {wraps.get(inner, 1)} blocks of {block}; this boundary '
            'lowering transfers one block per port. Split the axis across ports, or add a boundary '
            'layout that interleaves the blocks.'
        )
    descriptor.pop('boundary_dimension', None)  # the whole padded region moves; the host trims
    # One block is contiguous and already in wire order, so it moves as a plain walk over the
    # outer axes. A BD counts at most BD_MAX_WRAP steps, so the innermost chunk takes as many
    # whole axes as fit and the rest are traversed.
    chunk = list(dims)
    traversal = []
    for dim in range(inner + 1, len(dims)):
        if int(np.prod(chunk[: dim + 1])) * element_bits // BD_WORD_BITS <= BD_MAX_WRAP and not traversal:
            continue
        chunk[dim] = 1
        if int(dims[dim]) > 1:
            traversal.append(TraversalStep(dim, 1, int(dims[dim])))
    over = [step for step in traversal if step.wrap > BD_MAX_WRAP]
    if over or int(np.prod(chunk)) * element_bits // BD_WORD_BITS > BD_MAX_WRAP:
        raise ConfigRefused(
            f"An inner-blocked boundary transfer of {dims} needs a descriptor beyond a BD's {BD_MAX_WRAP} "
            'steps; split the tensor across ports.'
        )
    descriptor['tiling_dimension'] = chunk
    descriptor['tile_traversal'] = traversal
    return descriptor


def rebase_descriptor_offset(staging: StagingDescriptor, offset_base: Sequence[int]) -> StagingDescriptor:
    """The staging in an endpoint-local frame.

    A view hands a consumer part of a tensor as a tensor of its own, so both where the window sits
    in the buffer and where it sits in the data move with it.
    """
    if not offset_base:
        return staging
    return dataclasses.replace(staging, **_rebased(staging, offset_base))


def localize_descriptor(
    staging: StagingDescriptor, offset_base: Sequence[int], buffer_dimension: Sequence[int]
) -> StagingDescriptor:
    """The staging rebased into an endpoint-local buffer of `buffer_dimension`, its boundaries clipped to it."""
    if not offset_base:
        return staging
    dims, base = [int(value) for value in buffer_dimension], [int(value) for value in offset_base]
    if len(dims) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during localization ({len(dims)} != {len(base)}).')
    boundaries = {
        key: tuple(min(extent, max(0, bound - b)) for extent, bound, b in zip(dims, getattr(staging, key), base))
        for key in ('boundary_dimension', 'io_boundary_dimension')
        if getattr(staging, key) is not None
    }
    return dataclasses.replace(staging, **_rebased(staging, base), buffer_dimension=tuple(dims), **boundaries)


def _rebased(staging: StagingDescriptor, offset_base: Sequence[int]) -> Dict[str, Tuple[int, ...]]:
    base = [int(value) for value in offset_base]
    if len(staging.offset) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during rebasing ({len(staging.offset)} != {len(base)}).')
    return {key: tuple(v - b for v, b in zip(getattr(staging, key), base)) for key in ('offset', 'logical_origin')}
