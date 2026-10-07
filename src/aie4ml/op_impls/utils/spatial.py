"""2-D spatial access: what a windowed op reads around each output pixel, and the padded frame
a tensor therefore needs. Conv uses it today; any other windowed family (pooling, resampling)
declares the same contract and reuses the frame builder unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ...errors import ConfigRefused
from .math import align_up
from .tensor_view import TensorView


@dataclass(frozen=True)
class SpatialAccess2D:
    """The window a consumer reads around each output pixel of a rank-4 NHWC activation.

    kernel/strides/dilations are (height, width); pads is (top, left, bottom, right). The window
    spans `(kernel - 1) * dilation + 1` input pixels per axis.
    """

    kernel: Tuple[int, int]
    pads: Tuple[int, int, int, int]
    strides: Tuple[int, int] = (1, 1)
    dilations: Tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if len(self.kernel) != 2 or len(self.strides) != 2 or len(self.dilations) != 2 or len(self.pads) != 4:
            raise ValueError(f'SpatialAccess2D takes 2-D kernel/strides/dilations and 4 pads, got {self}.')
        if any(int(x) <= 0 for x in (*self.kernel, *self.strides, *self.dilations)):
            raise ValueError(f'SpatialAccess2D kernel/strides/dilations must be positive, got {self}.')
        if any(int(p) < 0 for p in self.pads):
            raise ValueError(f'SpatialAccess2D pads must be non-negative, got {self.pads}.')

    @property
    def window(self) -> Tuple[int, int]:
        """Input pixels the window spans per axis, after dilation."""
        return tuple((int(k) - 1) * int(d) + 1 for k, d in zip(self.kernel, self.dilations))

    def output_extent(self, height: int, width: int) -> Tuple[int, int]:
        span_h, span_w = self.window
        return (
            (int(height) + int(self.pads[0]) + int(self.pads[2]) - span_h) // int(self.strides[0]) + 1,
            (int(width) + int(self.pads[1]) + int(self.pads[3]) - span_w) // int(self.strides[1]) + 1,
        )


def shared_consumer_spatial_access(tensor) -> Optional[SpatialAccess2D]:
    """The window a producer lays this tensor's frame out for: the one its windowed consumers all read, or None
    when none of them is windowed or they read different windows.

    A consumer that reads the frame with the window it was laid out for takes it as it is; any other consumer --
    another window, or one that reads the image itself -- takes it through the transport, which re-frames it.
    """
    from ..family_registry import get_family_resolver_registry

    registry = get_family_resolver_registry()
    accesses = set()
    for consumer in tensor.consumers:
        resolver = registry.find(consumer.op_type)
        access = resolver.spatial_access(consumer) if resolver is not None else None
        if access is not None:
            accesses.add(access)
    return next(iter(accesses)) if len(accesses) == 1 else None


def build_padded_spatial_view(
    logical: Sequence[int],
    access: Optional[SpatialAccess2D],
    *,
    column_block: int,
    column_align: int,
    inner_block: int,
    inner_slices: int = 1,
    row_slices: int = 1,
    row_bytes_align: int = 1,
) -> TensorView:
    """Padded frame of a rank-4 NHWC activation, as a windowed kernel reads and writes it.

    The logical image sits at `origin` inside `full`; around it is the zero border `access` asks
    for. Columns hold both what a producer stores (its output width rounded up to `column_block`
    whole register tiles) and what the consumer reads (its own rounded width plus the window
    span), so producer and consumer derive the same shape from the tensor alone. The inner axis
    pads to `inner_block` and `tile` takes the slice one port carries.
    """
    batch, height, width, channels = (int(x) for x in logical)
    top, left, bottom, _right = (int(p) for p in access.pads) if access else (0, 0, 0, 0)
    span_w = access.window[1] if access else 1
    stride_w = int(access.strides[1]) if access else 1
    out_width = access.output_extent(height, width)[1] if access else width
    origin_col = align_up(left, int(column_align))
    computed_w = align_up(out_width, int(column_block))
    columns = max(
        origin_col + align_up(width, int(column_block)),
        origin_col - left + (computed_w - 1) * stride_w + span_w,
    )
    if span_w > 1 and stride_w == 1 and int(row_bytes_align) > 1:
        # The window's taps start at mixed alignments, so the kernel loads them all as the least aligned one: a tap
        # that starts on a row-alignment granule then reads one granule past its window, which must stay in the row.
        start = origin_col - left
        granule = int(row_bytes_align)
        ends = [start + kx + computed_w for kx in range(span_w) if (start + kx) % granule == 0]
        columns = max(columns, max(ends, default=0) + granule)
    if stride_w > 1:
        # A strided frame holds its columns in `stride_w` polyphase classes, so every class must be
        # long enough for the computed width plus the taps that reach past it, and the row must
        # hold whole classes.
        phase_cols = computed_w + (span_w - 1 + origin_col - left) // stride_w
        columns = max(columns, stride_w * phase_cols)
    padded_channels = align_up(channels, int(inner_block))
    if padded_channels % (int(inner_block) * int(inner_slices)):
        raise ConfigRefused(
            f'{padded_channels} channels do not split into {inner_slices} ports of whole {inner_block}-blocks.'
        )
    span_h = access.window[0] if access else 1
    stride_h = int(access.strides[0]) if access else 1
    out_height = access.output_extent(height, width)[0] if access else height
    if out_height % int(row_slices):
        raise ConfigRefused(f'{out_height} output rows do not split into {row_slices} equal bands.')
    column_align_bytes = int(row_bytes_align) * stride_w // math.gcd(int(row_bytes_align) or 1, stride_w)
    full = (batch, top + height + bottom, align_up(columns, max(1, column_align_bytes)), padded_channels)
    tile = (
        batch,
        (out_height // int(row_slices) - 1) * stride_h + span_h,
        full[2],
        padded_channels // int(inner_slices),
    )
    return TensorView(
        logical=tuple(int(x) for x in logical),
        full=full,
        tile=tile,
        tile_raw=tile,
        origin=(0, top, origin_col, 0),
    )
