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

FOLD_STEP_PIXELS = 16
"""Frame columns a folder writes per step (frame_fold.cpp): its frame's columns are whole steps."""

FOLD_ROW_BYTES = 32
"""A folder's source rows start on this boundary (frame_fold.cpp): its loads are aligned and their shifts constant."""

FOLD_READ_SLACK = 128
"""Bytes a folder's loads may read past the source rows they need (frame_fold.cpp): its input carries them."""


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
    input_fold: Optional[InputFold]  # the kernel reads its input folded, `spatial` the 1x1 window over it


@dataclass(frozen=True)
class InputFold:
    """A conv whose kernel reads its input folded: per output pixel its whole window, in `unit`-byte pixels (the
    channels padded to a power of two), as the channels of a 1x1 conv."""

    unit: int
    window: SpatialAccess2D  # the conv's own window, which the folder applies


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
    against_flow: bool  # the frame on its reader's output side (`row_flow`), the converter beside it there

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
        against_flow: bool,
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
            against_flow=bool(against_flow),
        )


@dataclass(frozen=True)
class FrameFoldConfig:
    """A folder: one kernel per window builds one tile of a folded conv's frame (a row slice, a channel slice of its
    cascade, or both, row-slice-major) from the input window it reads: its rows of `unit`-byte pixels inside the
    zero border, which the boundary delivers."""

    precision: Any  # the element type, as the conv's lhs precision
    source: str
    target: str
    source_view: TensorView  # one window's input
    frame_view: TensorView  # one window's frame
    unit: int
    window: SpatialAccess2D
    channel_slices: int  # the conv's cascade length: windows per row slice
    parallelism: ParallelismConfig  # a kernel per window, no cascade
    alternating_horizontal: bool
    bank_mem_bytes: int
    against_flow: bool  # the frame on its reader's output side (`row_flow`), the folder beside it there

    @property
    def transfer_bytes(self) -> int:
        """One inference of a window's input: its rows, then what the last row's steps read past them -- their columns
        past the output's are don't-care -- and the slack of their loads."""
        _, _, columns, unit = (int(x) for x in self.source_view.tile)
        read = int(self.frame_view.tile[2]) * int(self.window.strides[1]) + int(self.window.kernel[1]) - 1
        past = max(0, read - columns) * unit
        return align_up(int(np.prod(self.source_view.tile)) + past + FOLD_READ_SLACK, TRANSFER_ALIGN_BYTES)

    @property
    def pairs_bytes(self) -> int:
        """The pixel pairs a window of one-byte pixels at stride 1 keeps (frame_fold.cpp `PAIRED`): two bytes per byte
        of its input."""
        return 2 * self.transfer_bytes if self.unit == 1 and int(self.window.strides[1]) == 1 else 0

    @classmethod
    def for_frame(
        cls,
        precision,
        shape,
        fold: InputFold,
        frame: TensorView,
        *,
        source: str,
        target: str,
        row_slices: int,
        channel_slices: int,
        alternating_horizontal: bool,
        bank_mem_bytes: int,
        against_flow: bool,
    ):
        _, height, width, _ = (int(x) for x in shape)
        top, left, bottom, right = fold.window.pads
        rows = int(frame.tile[1])
        # Every output column's window, each row on a load boundary.
        columns = align_up(left + width + right, FOLD_ROW_BYTES // fold.unit)
        tile = (1, (rows - 1) * int(fold.window.strides[0]) + int(fold.window.kernel[0]), columns, fold.unit)
        source_view = TensorView(
            logical=tuple(int(x) for x in shape),
            full=(1, top + height + bottom, columns, fold.unit),
            tile=tile,
            tile_raw=tile,
            origin=(0, top, left, 0),
        )
        return cls(
            precision=precision,
            source=source,
            target=target,
            source_view=source_view,
            frame_view=frame,
            unit=fold.unit,
            window=fold.window,
            channel_slices=int(channel_slices),
            parallelism=ParallelismConfig(
                cas_num=int(row_slices) * int(channel_slices),
                cas_length=1,
                contract='outer' if int(row_slices) > 1 else 'inner',
            ),
            alternating_horizontal=bool(alternating_horizontal),
            bank_mem_bytes=int(bank_mem_bytes),
            against_flow=bool(against_flow),
        )
