"""The retiler: a kernel graph of its own that builds a strided conv's column-grouped frame.

It implements no logical op. The layout legalization pass inserts it where a strided conv asks for
it, so placement, transport and the build see it as the kernel it is.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ...base import BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PORT_KIND_BUFFER, PortBinding, PortMap
from .common import describe_frame_staging, describe_logical_staging
from .config import FrameRetileConfig


class FrameRetileOpImplVariant(OpImplVariant):
    """Reads the tensor the boundary carries, or the frame a producer wrote, and writes the frame a
    strided conv reads: its columns grouped by residue, the image inside a zero border it clears
    every inference. One kernel per window, on consecutive rows: the whole frame, or one per tile the
    conv splits it into -- a row slice ('outer' chains), a channel slice of its cascade, or both."""

    variant_id = 'frame_retile.b.v1'
    op_type = 'frame_retile'
    graph_header = 'frame_retile_graph.h'
    graph_name = 'frame_retile_graph'
    param_template = 'frame_retile'

    def validate_config(self, node, config: FrameRetileConfig, device) -> None:
        # The op contract's bank schedule, as for the conv it feeds: one copy of each kernel's input,
        # and of its frame, per bank (0 and 3), its stack in bank 1.
        frame = int(np.prod(config.frame_view.tile))
        for what, size in [('frame', frame)] + [('input', window.transfer_bytes) for window in config.windows]:
            if int(size) > int(config.bank_mem_bytes):
                raise ValueError(
                    f"{node.name}: a retiler's {what} is {size} B but one {device.platform} memory bank holds "
                    f"{config.bank_mem_bytes} B; split the strided conv it feeds by rows (contract 'outer', a larger "
                    'cas_num).'
                )

    def buffer_locations(self, _node, config: FrameRetileConfig, anchor_row):
        """The op contract (`row_flow`), as for Dense and the conv it feeds: each kernel's input and frame
        in banks 0 and 3 of the neighbouring tile both kernels of the hand-over reach."""
        locations = []
        for window in range(int(config.parallelism.cas_num)):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + window, 1)
            locations.append(BufferLocation('in1', window, flow.input_col, window, (0, 3)))
            locations.append(BufferLocation('out1', window, flow.output_col, window, (0, 3)))
        return tuple(locations)

    def kernel_params(self, node, config: FrameRetileConfig):
        view = config.frame_view
        _, rows, cols, channels = (int(x) for x in view.tile)
        return {
            'precision': config.precision,
            'parallelism': config.parallelism,
            'windows': [{**asdict(window), **config.source_steps(window)} for window in config.windows],
            'src_w': int(view.logical[2]),
            'rows': rows,
            'cols': cols,
            'blocks': channels // 8,
            'stride': config.column_phases,
            'origin_c': int(view.origin[2]),
            'frame_bytes': int(np.prod(view.tile)),
        }

    def build_ports(self, _node, config: FrameRetileConfig) -> PortMap:
        windows = range(int(config.parallelism.cas_num))
        return PortMap(
            inputs={
                config.source: PortBinding(
                    'in1', len(windows), PORT_KIND_BUFFER, tuple((f'kk[{w}].in[0]',) for w in windows)
                )
            },
            outputs={
                config.target: PortBinding(
                    'out1', len(windows), PORT_KIND_BUFFER, tuple((f'kk[{w}].out[0]',) for w in windows)
                )
            },
        )

    def footprint(self, _node, config: FrameRetileConfig) -> OpImplFootprint:
        return OpImplFootprint(width=int(config.parallelism.cas_length), height=int(config.parallelism.cas_num))

    def describe_input_staging(self, _node, config: FrameRetileConfig, _tensor_name, port, _producer=None):
        window = config.windows[int(port)]
        if config.from_boundary:
            return describe_logical_staging(
                config.frame_view,
                'read',
                transfer_bytes=window.transfer_bytes,
                rows=(window.first_row, window.rows),
                channels=(window.first_channel, window.channels),
            )
        return describe_frame_staging(config.frame_view, 'read', int(port) % config.channel_slices, fills_border=True)

    def describe_output_staging(self, _node, config: FrameRetileConfig, _tensor_name, port):
        row_slice, part = divmod(int(port), config.channel_slices)
        return describe_frame_staging(
            config.frame_view,
            'write',
            part,
            row_slice=row_slice,
            row_step=config.row_step,
            column_phases=config.column_phases,
        )

    def input_precision(self, config: FrameRetileConfig, _role):
        return config.precision

    def output_precision(self, config: FrameRetileConfig):
        return config.precision

    def output_staging_contract(self, _node, config: FrameRetileConfig, _tensor_name):
        return config.parallelism.contract

    def output_port_count(self, _node, config: FrameRetileConfig):
        return int(config.parallelism.cas_num)

    def pack(self, _inst):
        return {}
