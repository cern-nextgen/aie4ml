"""Where each element of a kernel port's buffer sits in the tensor, and the DMA walk that moves it.

A memory tile holds its tensor plainly: row-major over the tensor's padded axes, in buffer order. A kernel buffer holds
its part of the tensor in the order its kernel reads it -- a conv frame channel block by channel block, a matmul operand
microtile by microtile. A port's `Layout` states that order once, from the staging its op publishes, and `Layout.walk`
turns it into the access pattern that moves the port's elements to or from the plain buffer. So the two ends of a
memory tile agree on the tensor alone, never on each other's layout; and two ports share one buffer, with no memory
tile, exactly when their layouts are equal.

Coordinates are tensor coordinates in buffer order (the tensor's axes reversed): 0 is the tensor's first element and a
negative coordinate lies in a border before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import numpy as np

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
    tensor coordinate `start`. Per axis, `data` bounds [first, last + 1) the coordinates it holds of the tensor, whose
    own bounds are `tensor`: the rest is padding -- a border, an alignment, or a slice rounded up past its share --
    which the kernel ignores or needs as zeros, and which no other port may take for data."""

    loops: Tuple[Loop, ...]
    start: Tuple[int, ...]
    data: Tuple[Tuple[int, int], ...]
    tensor: Tuple[Tuple[int, int], ...] = field(compare=False)

    def shifted(self, base: Sequence[int]) -> 'Layout':
        """The same buffer in the coordinates of a window of the tensor that begins at `base` (a view's part)."""
        if not base:
            return self

        def shift(bounds):
            return tuple((int(lo) - int(b), int(hi) - int(b)) for (lo, hi), b in zip(bounds, base))

        return Layout(
            self.loops,
            tuple(int(start) - int(b) for start, b in zip(self.start, base)),
            *map(shift, (self.data, self.tensor)),
        )

    def canonical(self) -> 'Layout':
        """The same buffer with no single-count loop, and each loop that walks whole runs of the next one merged with
        it: two layouts hold the same elements in the same order exactly when their canonical forms are equal."""
        loops = []
        for loop in self.loops:
            if loop.count == 1:
                continue
            if loops and loops[-1].axis == loop.axis and loops[-1].step == loop.step * loop.count:
                loop = Loop(loop.axis, loops.pop().count * loop.count, loop.step)
            loops.append(loop)
        return Layout(tuple(loops), self.start, self.data, self.tensor)

    def elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per buffer position in memory order, the tensor coordinate it holds (positions x axes), and whether it is
        data rather than padding."""
        coords = np.array([self.start], dtype=np.int64)
        for loop in self.loops:  # slowest first: each further loop varies faster
            offsets = np.zeros((loop.count, len(self.start)), dtype=np.int64)
            offsets[:, loop.axis] = np.arange(loop.count) * loop.step
            coords = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, len(self.start))
        data = np.all([(coords[:, axis] >= lo) & (coords[:, axis] < hi) for axis, (lo, hi) in enumerate(self.data)], 0)
        return coords, data

    def span(self) -> Tuple[Tuple[int, int], ...]:
        """Per axis, the tensor coordinates [first, last + 1) the buffer holds."""
        ends = [int(start) + 1 for start in self.start]
        for loop in self.loops:
            ends[loop.axis] += loop.step * (loop.count - 1)
        return tuple((int(start), end) for start, end in zip(self.start, ends))

    def pads_within(self) -> bool:
        """Whether some of its padding lies inside the tensor, where another port holds data."""
        inside = []
        for axis, (first, last) in enumerate(self.tensor):
            coords = {int(self.start[axis])}
            for loop in self.loops:
                if loop.axis == axis:
                    coords = {c + i * loop.step for c in coords for i in range(loop.count)}
            inside.append({c for c in coords if first <= c < last})
        if not all(inside):
            return False
        return any(any(not lo <= c < hi for c in coords) for coords, (lo, hi) in zip(inside, self.data))

    def walk(
        self, box_start: Sequence[int], box_shape: Sequence[int], *, zero_padding: bool = False, word: int = 1
    ) -> Dict[str, Any]:
        """The DMA walk, as ADF tiling fields, that visits this buffer's elements in its memory order within a plain
        buffer holding the tensor coordinates [box_start, box_start + box_shape). The fastest loops over ascending
        axes, each stepping one element, make the contiguous tile; the rest traverse it.

        A read that wants its padding as zeros gets them wherever the walk leaves its data; the DMA fills zeros from
        a whole `word` of elements on, along the contiguous axis.
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
        span = [(first - int(base), end - int(base)) for (first, end), base in zip(self.span(), box_start)]
        if any(first < 0 or end > int(extent) for (first, end), extent in zip(span, box_shape)):
            raise RuntimeError(f'A walk spanning {span} leaves its buffer {list(box_shape)}.')
        walk = {
            'buffer_dimension': [int(extent) for extent in box_shape],
            'tiling_dimension': tile,
            'offset': [first for first, _ in span],
            'tile_traversal': traversal,
        }
        if not zero_padding:
            return walk
        # The data bounds the zeros only where the walk leaves it; elsewhere the tensor does.
        low, high = [], []
        for (first, end), (lo, hi), (tensor_lo, tensor_hi), base, extent in zip(
            self.span(), self.data, self.tensor, box_start, box_shape
        ):
            low.append(max(0, (lo if first < lo else tensor_lo) - int(base)))
            high.append(min(int(extent), (hi if end > hi else tensor_hi) - int(base)))
        high[0] = min(int(box_shape[0]), -(-high[0] // int(word)) * int(word))
        walk['boundary_dimension'] = [max(0, hi - lo) for lo, hi in zip(low, high)]
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
    start = tuple(int(origin) for origin in descriptor['logical_origin'])
    tensor = tuple((0, int(extent)) for extent in descriptor['io_boundary_dimension'])
    # The port holds the tensor's elements from its origin for its data extent (`io_tiling_dimension`).
    data = tuple(
        (max(0, origin), max(0, min(last, origin + int(held))))
        for origin, held, (_, last) in zip(start, descriptor['io_tiling_dimension'], tensor)
    )
    return Layout(tuple(loops), start, data, tensor)


def host_layout(descriptor: Dict[str, Any], extent: Sequence[int] | None = None) -> Layout:
    """A host port's share of a kernel port's tensor, as a PLIO carries it: `extent` elements per axis from the kernel
    port's origin, in the tensor's own order -- its data extent (`io_tiling_dimension`) unless the host moves more."""
    kernel = port_layout(descriptor)
    extent = [int(value) for value in (extent if extent is not None else descriptor['io_tiling_dimension'])]
    loops = tuple(Loop(axis, extent[axis], 1) for axis in reversed(range(len(extent))))
    return Layout(loops, kernel.start, kernel.data, kernel.tensor)
