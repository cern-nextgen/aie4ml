from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...utils import ParallelismConfig, SpatialAccess2D, TensorView
from ...utils.math import align_up
from ..matmul.config import MatmulMicrotileConfig

TRANSFER_ALIGN_BYTES = 16
"""A kernel buffer is whole 16-byte units -- the compiler rounds any other size up -- so an inference
transfer into one is too, the tensor followed by zeros that are not part of it."""


@dataclass(frozen=True)
class Conv2dFlags:
    use_relu: bool
    emit_flattened: bool  # write the result as a Dense LHS row (the folded flatten_2d view)


@dataclass(frozen=True)
class Pool2DConfig:
    """A pool fused into the conv's epilogue."""

    kind: str
    window: SpatialAccess2D  # kernel, pads (top, left, bottom, right), strides, dilations


@dataclass(frozen=True)
class Conv2dConfig:
    """Resolved Conv2D: the canonical attributes plus the padded frames its kernel reads/writes."""

    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    spatial_blocks: int  # mmul row tiles per accumulator set (2 on AIE, 4 on AIE-ML/MLv2)
    spatial: SpatialAccess2D  # the window: kernel, pads (top, left, bottom, right), strides, dilations
    groups: int
    alternating_horizontal: bool
    bank_mem_bytes: int  # one memory bank: what a bank-pinned buffer copy, and the weights, must fit
    flags: Conv2dFlags
    pool: Optional[Pool2DConfig]


@dataclass(frozen=True)
class RetileWindow:
    """One retiler kernel's window: `rows` tensor rows from `first_row` land at frame row `origin_row`,
    `channels` from `first_channel`; `transfer_bytes` is one inference of it from the boundary."""

    first_row: int
    rows: int
    origin_row: int
    first_channel: int
    channels: int
    transfer_bytes: int


@dataclass(frozen=True)
class FrameRetileConfig:
    """A retiler: one kernel per window builds one tile of the conv's column-grouped input frame (a row
    slice, a channel slice of its cascade, or both, row-slice-major), reading the tensor linearly from
    the boundary or the plain frame a producer wrote."""

    precision: Any  # the element type, as the conv's lhs precision
    source: str  # the value it reads
    target: str  # the frame it writes, which only the execution graph knows
    frame_view: TensorView  # one window's frame: one row slice, one channel slice
    column_phases: int
    row_step: int  # frame rows between the first rows of neighbouring row slices; 0 for one
    channel_slices: int  # the conv's cascade length: windows per row slice
    windows: Tuple[RetileWindow, ...]
    parallelism: ParallelismConfig  # a kernel per window, no cascade
    from_boundary: bool
    alternating_horizontal: bool  # AIE: odd-row cores reach their east neighbour's memory, not the west's
    bank_mem_bytes: int

    def source_steps(self, window: RetileWindow) -> Dict[str, int]:
        """Byte addressing of a window's image in its source: channels per pixel, first-pixel base, and
        pixel/row/8-channel-block steps."""
        view = self.frame_view
        width = int(view.logical[2])
        _, rows, cols, padded_channels = (int(x) for x in view.tile)
        if self.from_boundary:
            channels = window.channels
            return {'channels': channels, 'base': 0, 'pixel': channels, 'row': width * channels, 'block': 8}
        return {
            'channels': padded_channels,
            'base': (int(view.origin[1]) * cols + int(view.origin[2])) * 8,
            'pixel': 8,
            'row': cols * 8,
            'block': rows * cols * 8,
        }

    @classmethod
    def for_frame(
        cls,
        precision,
        view: TensorView,
        column_phases: int,
        *,
        source: str,
        target: str,
        from_boundary: bool,
        row_slices: int,
        row_step: int,
        channel_slices: int,
        alternating_horizontal: bool,
        bank_mem_bytes: int,
    ):
        _, height, width, channels = (int(x) for x in view.logical)
        rows, top, slice_channels = int(view.tile[1]), int(view.origin[1]), int(view.tile[3])
        if from_boundary:
            windows = []
            for row_slice in range(int(row_slices)):
                # A row slice covers frame rows from row_slice * row_step: those rows of the tensor that exist.
                start = row_slice * int(row_step) - top
                first, last = max(0, start), min(height, start + rows)
                for part in range(int(channel_slices)):
                    # The slice's channels that exist: the last one may not fill its blocks.
                    first_channel = part * slice_channels
                    count = min(channels, first_channel + slice_channels) - first_channel
                    transfer = align_up((last - first) * width * count, TRANSFER_ALIGN_BYTES)
                    windows.append(RetileWindow(first, last - first, first - start, first_channel, count, transfer))
        else:
            if int(row_slices) != 1:
                raise NotImplementedError("a retiler reads a producer's frame whole, not split by rows.")
            # Each slice is the frame the producer's chain of the same index wrote.
            windows = [
                RetileWindow(0, height, top, part * slice_channels, slice_channels, int(np.prod(view.tile)))
                for part in range(int(channel_slices))
            ]
        return cls(
            precision=precision,
            source=source,
            target=target,
            frame_view=view,
            column_phases=int(column_phases),
            row_step=int(row_step),
            channel_slices=int(channel_slices),
            windows=tuple(windows),
            parallelism=ParallelismConfig(
                cas_num=len(windows), cas_length=1, contract='outer' if int(row_slices) > 1 else 'inner'
            ),
            from_boundary=from_boundary,
            alternating_horizontal=bool(alternating_horizontal),
            bank_mem_bytes=int(bank_mem_bytes),
        )
