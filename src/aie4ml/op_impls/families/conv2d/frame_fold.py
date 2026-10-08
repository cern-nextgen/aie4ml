"""The folder: a layout conversion, like the retiler, that builds a folded conv's frame on the retiler's graph."""

from __future__ import annotations

import numpy as np

from .common import describe_frame_staging
from .config import FrameFoldConfig
from .frame_retile import FrameRetileOpImplVariant


class FrameFoldOpImplVariant(FrameRetileOpImplVariant):
    """Reads the input window the boundary carries, its channels padded to a power of two, and writes the frame a
    folded conv reads: per output pixel its whole window, as the channels of a 1x1 conv. One kernel per window, on
    consecutive rows: the whole frame, or one per tile the conv splits it into -- a row slice ('outer' chains), a
    channel slice of its cascade, or both."""

    variant_id = 'frame_fold.b.v1'
    op_type = 'frame_fold'
    param_template = 'frame_fold'

    def validate_config(self, node, config: FrameFoldConfig, device) -> None:
        # The retiler's bank schedule: one copy of each kernel's input, and of its frame, per bank (0 and 3); its pixel
        # pairs, if any, in its own data.
        sizes = (
            ('input', config.transfer_bytes),
            ('frame', int(np.prod(config.frame_view.tile))),
            ('pixel pairs', config.pairs_bytes),
        )
        for what, size in sizes:
            if size > int(config.bank_mem_bytes):
                raise ValueError(
                    f"{node.name}: a folder's {what} is {size} B but one {device.platform} memory bank holds "
                    f"{config.bank_mem_bytes} B; split the conv it feeds by rows (contract 'outer', a larger cas_num)."
                )

    def kernel_params(self, _node, config: FrameFoldConfig):
        _, rows, cols, channels = (int(x) for x in config.frame_view.tile)
        blocks = channels // 8
        return {
            'precision': config.precision,
            'parallelism': config.parallelism,
            'unit': config.unit,
            'window': config.window,
            'src_cols': int(config.source_view.tile[2]),
            'src_bytes': config.transfer_bytes,
            'rows': rows,
            'cols': cols,
            'blocks': blocks,
            'frame_bytes': int(np.prod(config.frame_view.tile)),
            'first_blocks': [
                window % config.channel_slices * blocks for window in range(int(config.parallelism.cas_num))
            ],
        }

    def _row_step(self, config: FrameFoldConfig) -> int:
        """Frame rows between neighbouring row slices."""
        return int(config.frame_view.tile[1])

    def describe_input_staging(self, _node, config: FrameFoldConfig, _tensor_name, port, _producer=None):
        return describe_frame_staging(
            config.source_view,
            'read',
            0,
            row_slice=int(port) // config.channel_slices,
            row_step=self._row_step(config) * int(config.window.strides[0]),
            transfer_bytes=config.transfer_bytes,
        )

    def describe_output_staging(self, _node, config: FrameFoldConfig, _tensor_name, port):
        row_slice, part = divmod(int(port), config.channel_slices)
        return describe_frame_staging(
            config.frame_view, 'write', part, row_slice=row_slice, row_step=self._row_step(config)
        )
