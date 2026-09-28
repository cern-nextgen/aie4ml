from __future__ import annotations

import copy
from typing import Any, Dict, MutableMapping, Sequence

import numpy as np

from ...errors import ConfigRefused
from ...op_impls.utils import (
    STORAGE_LAYOUT_INNER_BLOCKED,
    STORAGE_LAYOUT_LINEAR,
    STORAGE_LAYOUT_MICROTILED,
    STORAGE_LAYOUTS,
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
    walked = {int(step['dimension']): step for step in desc.get('tile_traversal', ())}
    for dim, extent in enumerate(dims):
        step = walked.get(dim)
        if step is None:
            if chunk[dim] != extent:
                return False
        elif int(step['wrap']) * chunk[dim] != extent:
            return False
        elif int(step['wrap']) > 1 and int(step['stride']) != 1:  # one tile never uses its stride
            return False
    return sorted(walked) == list(walked)


def boundary_access_descriptor(
    base: Dict[str, Any], *, element_bits: int, project_to_io_boundary: bool = False
) -> Dict[str, Any]:
    """Lower a staging layout to an ADF graph-boundary access descriptor.

    `element_bits` is the port's element width: a storage layout says how a buffer is arranged,
    not what it holds, while a buffer descriptor counts words.
    """

    storage_layout = base.get('storage_layout')
    if storage_layout not in STORAGE_LAYOUTS:
        raise ValueError(f'Unknown or missing staging storage_layout {storage_layout!r}.')
    if storage_layout == STORAGE_LAYOUT_LINEAR:
        return copy.deepcopy(base)
    if storage_layout == STORAGE_LAYOUT_INNER_BLOCKED:
        if project_to_io_boundary:
            raise ConfigRefused('A blocked frame is transferred whole; the host trims it.')
        return _inner_blocked_access_descriptor(base, int(element_bits))
    if storage_layout != STORAGE_LAYOUT_MICROTILED:
        raise ConfigRefused(f'No boundary DMA lowering for storage_layout {storage_layout!r}.')

    dimensions = [int(value) for value in base['buffer_dimension']]
    tile = [int(value) for value in base['tiling_dimension']]
    if len(dimensions) != 2 or len(tile) != 2:
        raise ConfigRefused('Microtiled boundary DMA requires a 2-D kernel buffer.')

    inner = int(base['inner_dimension'])
    outer = int(base['outer_dimension'])
    microtile_inner = tile[inner]
    microtile_outer = tile[outer]
    if inner != 0 or outer != 1 or any(int(value) != 0 for value in base['offset']):
        raise ConfigRefused('Microtiled boundary DMA requires a local, inner-contiguous buffer.')
    if microtile_outer == 1 and not project_to_io_boundary:
        return copy.deepcopy(base)

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
        io_boundary = [int(value) for value in base['io_boundary_dimension']]
        if len(io_boundary) != 2:
            raise ConfigRefused('Direct boundary DMA projection requires a 2-D IO boundary.')
        io_inner, io_outer = io_boundary
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
    descriptor = copy.deepcopy(base)
    descriptor['buffer_dimension'] = [
        microtile_elements,
        inner_extent // microtile_inner,
        outer_extent // microtile_outer,
    ]
    descriptor['tiling_dimension'] = [microtile_inner, 1, 1]
    descriptor['offset'] = [0, 0, 0]
    descriptor['tile_traversal'] = [
        {'dimension': 1, 'stride': 1, 'wrap': transfer_inner_extent // microtile_inner},
        {'dimension': 0, 'stride': microtile_inner, 'wrap': rows_per_tile},
        {'dimension': 2, 'stride': 1, 'wrap': tile_rows},
    ]
    descriptor['transfer_shape'] = [transfer_inner_extent, rows_per_tile * tile_rows]
    descriptor.pop('boundary_dimension', None)
    return descriptor


def _inner_blocked_access_descriptor(base: Dict[str, Any], element_bits: int) -> Dict[str, Any]:
    """Boundary DMA for an inner-blocked buffer ([c/B][...][B] memory, linear wire order).

    One block of the inner axis is one contiguous region whose wire order is already its memory
    order, so it transfers as a single tile. A port spanning several blocks would need the DMA to
    interleave them per element, which this lowering does not implement (measured on AIE1: such a
    walk needs 20 S2MM descriptors against a budget of 16).
    """
    descriptor = copy.deepcopy(base)
    dims = [int(value) for value in base['buffer_dimension']]
    inner = int(base['inner_dimension'])
    block = int(base['tiling_dimension'][inner])
    wraps = {int(step['dimension']): int(step['wrap']) for step in base.get('tile_traversal', ())}
    if any(int(value) != 0 for dim, value in enumerate(base['offset']) if dim != inner):
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
            traversal.append({'dimension': dim, 'stride': 1, 'wrap': int(dims[dim])})
    over = [step for step in traversal if step['wrap'] > BD_MAX_WRAP]
    if over or int(np.prod(chunk)) * element_bits // BD_WORD_BITS > BD_MAX_WRAP:
        raise ConfigRefused(
            f"An inner-blocked boundary transfer of {dims} needs a descriptor beyond a BD's {BD_MAX_WRAP} "
            'steps; split the tensor across ports.'
        )
    descriptor['tiling_dimension'] = chunk
    descriptor['tile_traversal'] = traversal
    return descriptor


def rebase_descriptor_offset(descriptor: MutableMapping[str, Any], offset_base: Sequence[int]) -> None:
    """Rebase a descriptor's coordinates into an endpoint-local frame.

    A view hands a consumer part of a tensor as a tensor of its own, so both where the window sits
    in the buffer and where it sits in the data move with it.
    """
    if not offset_base:
        return
    base = [int(value) for value in offset_base]
    for key in ('offset', 'logical_origin'):
        values = [int(value) for value in descriptor[key]]
        if len(values) != len(base):
            raise RuntimeError(f'descriptor rank mismatch during {key} rebasing ({len(values)} != {len(base)}).')
        descriptor[key] = [values[dim] - base[dim] for dim in range(len(values))]


def localize_descriptor(
    descriptor: MutableMapping[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> None:
    """Rebase a descriptor and its boundaries into an endpoint-local buffer."""
    if not offset_base:
        return

    dims = [int(value) for value in buffer_dimension]
    rebase_descriptor_offset(descriptor, offset_base)
    base = [int(value) for value in offset_base]
    if len(dims) != len(base):
        raise RuntimeError(f'descriptor rank mismatch during localization ({len(dims)} != {len(base)}).')

    descriptor['buffer_dimension'] = list(dims)
    for key in ('boundary_dimension', 'io_boundary_dimension'):
        if key not in descriptor:
            continue
        boundary = [int(value) for value in descriptor[key]]
        if len(boundary) != len(dims):
            raise RuntimeError(f'descriptor {key} rank mismatch during localization.')
        descriptor[key] = [min(dims[dim], max(0, boundary[dim] - base[dim])) for dim in range(len(dims))]


def localized_graph_io_descriptor(
    descriptor: Dict[str, Any],
    offset_base: Sequence[int],
    buffer_dimension: Sequence[int],
) -> Dict[str, Any]:
    """Return a graph-IO descriptor localized to one memory-tile shard."""
    localized = copy.deepcopy(descriptor)
    dims = [int(value) for value in buffer_dimension]
    boundary = list(localized['io_boundary_dimension'])
    base = [int(value) for value in offset_base]
    if len(base) != len(dims) or len(boundary) != len(dims):
        raise RuntimeError('graph-IO descriptor rank mismatch during localization.')
    rebase_descriptor_offset(localized, base)
    localized['buffer_dimension'] = list(dims)
    if localized.get('access') == 'read':
        localized['boundary_dimension'] = [
            min(dims[dim], max(0, int(boundary[dim]) - base[dim])) for dim in range(len(dims))
        ]
    else:
        localized.pop('boundary_dimension', None)
    return localized
