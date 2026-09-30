"""Where each element of a kernel port's buffer sits in the tensor, and the DMA walk that moves it.

A memory tile holds its tensor plainly: row-major over the tensor's padded axes, in buffer order. A kernel buffer holds
its part of the tensor in the order its kernel reads it -- a conv frame channel block by channel block, a matmul operand
microtile by microtile. A port's `Layout` states that order once, from the staging its op publishes, and `Layout.walk`
turns it into the access pattern that moves the port's elements to or from the plain buffer. So the two ends of a
memory tile agree on the tensor alone, never on each other's layout.

Coordinates are tensor coordinates in buffer order (the tensor's axes reversed): 0 is the tensor's first element and a
negative coordinate lies in a border before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ...errors import ConfigRefused
from ...op_impls.utils import STORAGE_LAYOUT_INNER_BLOCKED


@dataclass(frozen=True)
class Loop:
    """`count` elements of the tensor, `step` apart along buffer-order `axis`."""

    axis: int
    count: int
    step: int


@dataclass(frozen=True)
class Layout:
    """A kernel buffer: its elements in memory order, as loops from the slowest to the fastest, from the element at
    tensor coordinate `start`."""

    loops: Tuple[Loop, ...]
    start: Tuple[int, ...]

    def shifted(self, base: Sequence[int]) -> 'Layout':
        """The same buffer in the coordinates of a window of the tensor that begins at `base` (a view's part)."""
        if not base:
            return self
        return Layout(self.loops, tuple(int(start) - int(b) for start, b in zip(self.start, base)))

    def span(self) -> Tuple[Tuple[int, int], ...]:
        """Per axis, the tensor coordinates [first, last + 1) the buffer holds."""
        ends = [int(start) + 1 for start in self.start]
        for loop in self.loops:
            ends[loop.axis] += loop.step * (loop.count - 1)
        return tuple((int(start), end) for start, end in zip(self.start, ends))

    def walk(
        self, box_start: Sequence[int], box_shape: Sequence[int], *, data: Optional[Sequence[int]] = None
    ) -> Dict[str, Any]:
        """The DMA walk, as ADF tiling fields, that visits this buffer's elements in its memory order within a plain
        buffer holding the tensor coordinates [box_start, box_start + box_shape). The fastest loops over ascending
        axes, each stepping one element, make the contiguous tile; the rest traverse it.

        A read passes `data`, the tensor's extent: it bounds the real elements, and whatever the walk visits outside
        them -- a border or padding the buffer holds or not -- reads as zeros.
        """
        rank = len(box_shape)
        tile = [1] * rank
        traversal = []
        last = -1
        for loop in reversed(self.loops):
            if not traversal and loop.step == 1 and loop.axis > last:
                tile[loop.axis] = loop.count
                last = loop.axis
            else:
                traversal.append({'dimension': loop.axis, 'stride': loop.step, 'wrap': loop.count})
        if len({step['dimension'] for step in traversal}) != len(traversal):
            raise ConfigRefused(
                f'A buffer walked in the order {self.loops} steps along one axis at two strides, which one DMA '
                'descriptor does not express.'
            )
        walk = {
            'buffer_dimension': [int(extent) for extent in box_shape],
            'tiling_dimension': tile,
            'offset': [int(start) - int(base) for start, base in zip(self.start, box_start)],
            'tile_traversal': traversal,
        }
        if data is None:
            span = _span(walk)
            if any(lo < 0 or hi > int(extent) for (lo, hi), extent in zip(span, box_shape)):
                raise RuntimeError(f'A write walk spanning {span} leaves its buffer {list(box_shape)}.')
            return walk
        low = [max(0, -int(base)) for base in box_start]
        high = [min(int(extent), int(end) - int(base)) for extent, end, base in zip(box_shape, data, box_start)]
        walk['boundary_dimension'] = [hi - lo for lo, hi in zip(low, high)]
        if any(low):
            walk['boundary_offset'] = low
        return walk


def port_layout(descriptor: Dict[str, Any]) -> Layout:
    """The memory order of a kernel port's buffer, from the staging its op publishes.

    A staging walks the port's part of the tensor in the order its buffer holds it: its contiguous tile, then its
    traversal from the innermost loop out. An inner-blocked buffer ([c/B][...][B], a conv frame) holds whole blocks of
    the inner axis one after another instead, and a column-phased one holds each row's columns grouped by residue.
    """
    tile = [int(extent) for extent in descriptor['tiling_dimension']]
    steps = reversed(descriptor.get('tile_traversal') or ())
    loops = [Loop(int(step['dimension']), int(step['wrap']), int(step['stride'])) for step in steps]
    loops += [Loop(axis, tile[axis], 1) for axis in reversed(range(len(tile)))]
    if descriptor['storage_layout'] == STORAGE_LAYOUT_INNER_BLOCKED:
        inner = int(descriptor['inner_dimension'])
        blocks = [loop for loop in loops if loop.axis == inner and loop.step > 1]
        loops = blocks + [loop for loop in loops if not (loop.axis == inner and loop.step > 1)]
        phases = int(descriptor.get('column_phases', 1))
        if phases > 1:
            columns = int(descriptor['outer_dimension'])
            row = max(
                (index for index, loop in enumerate(loops) if loop.axis == columns and loop.step == 1),
                key=lambda index: loops[index].count,
            )
            width = loops[row].count
            if width % phases:
                raise ValueError(f'A row of {width} columns does not split into {phases} column phases.')
            loops[row : row + 1] = [Loop(columns, phases, 1), Loop(columns, width // phases, phases)]
    return Layout(tuple(loops), tuple(int(start) for start in descriptor['logical_origin']))


def _span(walk: Dict[str, Any]):
    """Per axis, the [first, last + 1) buffer index a walk visits."""
    ends = [int(offset) + int(tile) for offset, tile in zip(walk['offset'], walk['tiling_dimension'])]
    for step in walk['tile_traversal']:
        ends[int(step['dimension'])] += int(step['stride']) * (int(step['wrap']) - 1)
    return [(int(offset), end) for offset, end in zip(walk['offset'], ends)]
