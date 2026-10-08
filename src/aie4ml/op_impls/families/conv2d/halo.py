"""Conv2D row bands, one tile each, handing their neighbours the rows the neighbours' windows read."""

from __future__ import annotations

from typing import Tuple

from ....errors import ConfigRefused
from ....ir.graph import VIEW_FLATTEN_2D, OpNode, input_tensor_for_role
from ...base import BufferLocation, row_flow
from ...common_types import PORT_KIND_BUFFER, PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, SpatialAccess2D, TensorView, shared_consumer_spatial_access
from .common import (
    describe_band_staging,
    frame_view,
    halo_ports,
    reads_neighbour_rows,
    spatial_access_of,
)
from .config import Conv2dConfig
from .conv2d import Conv2dOpImplVariant

def _reads_halo(node: OpNode) -> bool:
    """Whether its window reads rows of the neighbouring bands, which another kernel's bands write."""
    return input_tensor_for_role(node, 'lhs').producer is not None and reads_neighbour_rows(spatial_access_of(node))


def _window_rows(access: SpatialAccess2D) -> Tuple[int, int, int]:
    """The rows a window of row stride 1 reads above an output row's own and below it, and the rows it takes off the
    image's height."""
    top, _, bottom, _ = (int(p) for p in access.pads)
    span = int(access.window[0])
    return top, span - 1 - top, span - 1 - top - bottom


def _sent_rows(node: OpNode) -> Tuple[int, int, int]:
    """The rows its consumers' windows read above and below each band, and the rows they take off its height."""
    view = node.traits.get('output_view')
    out = node.outputs[0]
    if (view is not None and view.data['kind'] == VIEW_FLATTEN_2D) or not out.consumers:
        return 0, 0, 0
    if any(consumer.op_type != 'conv2d' for consumer in out.consumers):
        return 0, 0, 0  # only a conv band reads its halo from its neighbours
    access = shared_consumer_spatial_access(out)
    if access is None or access.strides[0] != 1 or _window_rows(access)[2] < 0:
        return 0, 0, 0  # a window that strides or grows the image does not read its halo from row bands
    return _window_rows(access)


def _band_halo(node: OpNode, bands: int, height: int, window: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """The rows above and below a band of an image `height` rows high that `window` reads (see `_window_rows`), and
    the step a band adds to the first and takes off the second: a window that shrinks the image by some rows takes
    as many off every band, so each band's window starts that much further into its own rows."""
    top, bottom, shrink = window
    step, extra = divmod(shrink, bands)
    rows = height // bands
    if extra:
        raise ConfigRefused(
            f'{node.name}: a window that takes {shrink} rows off the image does not leave {bands} equal row bands.'
        )
    deepest = max(top + (bands - 1) * step, bottom - step)
    if deepest > rows:
        raise ConfigRefused(
            f'{node.name}: its window reads {deepest} rows past a band, more than the bands of {rows} rows beside it '
            'hold; a band reads only those.'
        )
    return top, bottom, step


@register_variant
class Conv2dHaloOpImplVariant(Conv2dOpImplVariant):
    """Conv2D split into row bands (contract 'outer'), one tile each, where a window reads rows past a band: its
    own window, reading the bands of a producer split the same way, or its consumer's, reading its bands.

    Beside its own rows, a band writes the edge rows its neighbours' windows read of it, each a buffer of its own
    with a single reader (`halo_ports`); a band whose window reads its neighbours' rows takes them from those
    buffers and assembles its window on the tile (conv2d_halo.cpp). Every hand-over is then one buffer, shared
    wherever both ends reach it: no memory tile, no copy, and no row computed twice.
    """

    variant_id = 'conv2d.b.r.halo.v1'
    graph_header = 'conv2d_halo_graph.h'
    graph_name = 'conv2d_halo_graph'
    plevel = 20

    def matches(self, node: OpNode, device, directives) -> bool:
        requested = str((directives.get('parallelism') or {}).get('contract', 'inner'))
        return (
            super().matches(node, device, directives)
            and requested == 'outer'
            and (_reads_halo(node) or any(_sent_rows(node)))
        )

    def resolve(self, node: OpNode, device, directives, input_contracts) -> Conv2dConfig:
        config = super().resolve(node, device, directives, input_contracts)
        if not _reads_halo(node):
            return config
        spatial = config.spatial
        if spatial.strides != (1, 1):
            raise ConfigRefused(f'{node.name}: row bands exchange their halo at stride 1, got {spatial.strides}.')
        if _window_rows(spatial)[2] < 0:
            raise ConfigRefused(
                f'{node.name}: its window {spatial.kernel} with pads {spatial.pads} grows the image; row bands '
                'exchange their halo where a window keeps or shrinks it.'
            )
        self._halo_rows(node, config, 'lhs')  # refuses a halo the bands beside it do not hold
        return config

    def _resolve_parallelism(
        self, node, parallel_cfg, input_contracts, *, flatten: bool, kernel_input
    ) -> ParallelismConfig:
        """One tile per band: the producer's bands when its window reads them, else the requested ones."""
        if int(parallel_cfg.get('cas_length', 1)) != 1:
            raise ConfigRefused(
                f'{node.name}: a halo band is one tile; a cascade would need the halo at every one of its tiles.'
            )
        if _reads_halo(node):
            producer = input_contracts.get(input_tensor_for_role(node, 'lhs').name)
            if producer is None or producer.contract != 'outer':
                raise ConfigRefused(
                    f'{node.name}: its row bands read their halo from their neighbours, so its producer must be '
                    "split into the same row bands (contract 'outer')."
                )
            # Each pair of bands passes rows up where the window reads above a row or shrinks the image (each band
            # then reads further into the one above), and down where it reads below.
            top, bottom, shrink = _window_rows(spatial_access_of(node))
            per_pair = (top > 0 or shrink > 0) + (bottom > 0)
            ports = len(producer.port_staging)
            bands, extra = divmod(ports + per_pair, 1 + per_pair)
            if extra:
                raise ConfigRefused(
                    f'{node.name}: its producer writes {ports} ports, not row bands with the halo rows between them '
                    'that its window reads.'
                )
            asked = parallel_cfg.get('cas_num')
            if asked is not None and int(asked) != bands:
                raise ConfigRefused(
                    f'{node.name}: cas_num={asked} does not match the {bands} row bands of its producer.'
                )
            parallelism = ParallelismConfig(cas_num=bands, cas_length=1, contract='outer')
        else:
            parallelism = super()._resolve_parallelism(
                node, parallel_cfg, input_contracts, flatten=flatten, kernel_input=kernel_input
            )
        if int(parallelism.cas_num) < 2:
            raise ConfigRefused(f'{node.name}: one row band has no neighbour to exchange a halo with.')
        return parallelism

    def _output_frame(
        self, node, parallelism: ParallelismConfig, *, column_block: int, column_align: int
    ) -> TensorView:
        if not any(_sent_rows(node)):
            return super()._output_frame(node, parallelism, column_block=column_block, column_align=column_align)
        # Every band writes its own rows into its reader's window, the frame's rows the reader's window covers; the
        # edge rows its neighbours read get ports of their own (halo_ports).
        view = frame_view(
            node.outputs[0],
            shared_consumer_spatial_access(node.outputs[0]),
            column_block=column_block,
            column_align=column_align,
            row_slices=int(parallelism.cas_num),
        )
        _band_halo(node, int(parallelism.cas_num), int(view.logical[1]), _sent_rows(node))
        return view

    def _halo_rows(self, node, config: Conv2dConfig, role: str) -> Tuple[int, int, int]:
        """The rows above and below a band its input's window reads (`lhs`) or its output's reader's does, and the
        step a band adds to the first and takes off the second."""
        if role == 'lhs':
            view = config.io_views[input_tensor_for_role(node, 'lhs').name]
            window = _window_rows(config.spatial) if _reads_halo(node) else (0, 0, 0)
        else:
            view = config.io_views[node.outputs[0].name]
            window = _sent_rows(node)
        return _band_halo(node, int(config.parallelism.cas_num), int(view.logical[1]), window)

    def _halo(self, node, config: Conv2dConfig, role: str):
        """The band rows and halo ports of its input (the rows its window reads) or its output (the rows it sends)."""
        tensor = input_tensor_for_role(node, 'lhs') if role == 'lhs' else node.outputs[0]
        rows = int(config.io_views[tensor.name].logical[1]) // int(config.parallelism.cas_num)
        return rows, halo_ports(int(config.parallelism.cas_num), rows, *self._halo_rows(node, config, role))

    def uses_depthwise_core(self, _node, _config) -> bool:
        return False  # a band's window is a frame for the mmul core

    def kernel_params(self, node, config: Conv2dConfig):
        params = super().kernel_params(node, config)
        top, bottom, step = self._halo_rows(node, config, 'lhs')
        sent_top, sent_bottom, sent_step = self._halo_rows(node, config, 'output')
        params.update(
            halo_top=top,
            halo_bottom=bottom,
            halo_step=step,
            # The rows it sends: its first ones are the bottom halo of the band above, its last ones the top halo
            # of the band below.
            send_first=sent_bottom,
            send_last=sent_top,
            send_step=sent_step,
            own_rows=self._halo(node, config, 'lhs')[0],
        )
        return params

    def validate_config(self, node: OpNode, config: Conv2dConfig, device) -> None:
        super().validate_config(node, config, device)
        if not _reads_halo(node):
            return
        # A reading band's tile holds, per bank (ping and pong one bank apart): its window in banks 0 and 3, beside
        # its output unless that goes on to the band that reads it; in banks 1 and 2 the halo rows its producer band
        # writes there for the bands beside it, with the bias in bank 1 and the weights in bank 2.
        params = self.build_template_params(node, config, {'row': 0, 'col': 0})
        window = self._frame_bytes(config, 'lhs', params['in_elements'])
        out = 0 if any(_sent_rows(node)) else self._frame_bytes(config, 'output', params['out_elements'])
        halo = window // int(params['in_rows']) * (int(params['in_rows']) - int(params['own_rows']))
        bias = int(params['bias_count']) * int(config.precision['bias'].width) // 8
        for banks, what, size in (
            ('0 and 3', 'window and output', window + out),
            ('1', 'halo rows and bias', halo + bias),
            ('2', 'halo rows and weights', halo + int(params['weight_count'])),
        ):
            if size > int(config.bank_mem_bytes):
                raise ConfigRefused(
                    f"{node.name}: a band's {what} need {size} B of bank {banks}, which holds "
                    f'{config.bank_mem_bytes} B; split it into more row bands.'
                )

    def reads_against_flow(self, node, config: Conv2dConfig) -> bool:
        """Bands hand their rows on against the flow: a band reading its halo reads it there. So does a band whose
        converter builds its input: the reader of the rows it sends sits on its input side, so the converter sits
        on the other."""
        return _reads_halo(node) or self.converts_input(config)

    def buffer_locations(self, node, config: Conv2dConfig, anchor_row):
        """Band b on row b. The rows a band writes for its reader -- its own and its edge rows -- sit where the
        band and its reader, one column west, both reach them (`row_flow`); the reader's neighbours reach them
        north and south, and read them in their own column: on an AIE1 odd row that is not where the band writes
        them, and a planned DMA carries them over."""
        bands = int(config.parallelism.cas_num)
        against, sends = self.reads_against_flow(node, config), any(_sent_rows(node))
        flows = [row_flow(config.alternating_horizontal, int(anchor_row) + band, 1) for band in range(bands)]
        locations = []
        for band, flow in enumerate(flows):
            locations.append(BufferLocation('in1', band, flow.output_col if against else flow.input_col, band, (0, 3)))
            locations.append(BufferLocation('out1', band, flow.input_col if sends else flow.output_col, band, (0, 3)))
        # Halo rows are a row or two: they go beside the stack, bias and weights, and leave banks 0 and 3 to the
        # windows.
        for index, port in enumerate(self._halo(node, config, 'lhs')[1], start=bands):
            locations.append(BufferLocation('in1', index, 0, port.band, (1, 2)))
        for index, port in enumerate(self._halo(node, config, 'output')[1], start=bands):
            locations.append(BufferLocation('out1', index, flows[port.band].input_col, port.band, (1, 2)))
        return tuple(locations)

    def output_staging_contract(self, _node, config, _tensor_name):
        # A flattened band is one contiguous slice of the row the consuming Dense reads.
        return 'inner' if config.flags.emit_flattened else 'outer'

    def output_port_count(self, node, config):
        return int(config.parallelism.cas_num) + len(self._halo(node, config, 'output')[1])

    def output_inner_shards(self, _node, _config, _tensor_name):
        return None

    def describe_input_staging(self, node, config, tensor_name, port, _producer=None):
        if not _reads_halo(node):
            return super().describe_input_staging(node, config, tensor_name, port, _producer)
        rows, halo = self._halo(node, config, 'lhs')
        step = self._halo_rows(node, config, 'lhs')[2]
        return describe_band_staging(config.io_views[tensor_name], 'read', rows, int(port), halo, step)

    def describe_output_staging(self, node, config, tensor_name, port):
        if not any(_sent_rows(node)):
            return super().describe_output_staging(node, config, tensor_name, port)
        rows, halo = self._halo(node, config, 'output')
        step = self._halo_rows(node, config, 'output')[2]
        return describe_band_staging(config.io_views[tensor_name], 'write', rows, int(port), halo, step)

    def build_ports(self, node: OpNode, config: Conv2dConfig):
        """Kernel ports, as conv2d_halo.h orders them: its own rows, then the top halo and the bottom halo it reads;
        its own rows, then the first rows and the last rows it sends."""
        bands = int(config.parallelism.cas_num)
        top, _, step = self._halo_rows(node, config, 'lhs')
        _, sent_bottom, sent_step = self._halo_rows(node, config, 'output')
        inputs = [(f'kk[{band}].in[0]',) for band in range(bands)]
        for port in self._halo(node, config, 'lhs')[1]:
            above = port.band < port.reader  # the band above's last rows are the reader's top halo
            reads_top = port.reader > 0 and top + port.reader * step > 0
            inputs.append((f'kk[{port.reader}].in[{1 if above else 1 + reads_top}]',))
        outputs = [(f'kk[{band}].out[0]',) for band in range(bands)]
        for port in self._halo(node, config, 'output')[1]:
            up = port.reader < port.band  # its first rows are the bottom halo of the band above
            sends_first = port.band > 0 and sent_bottom - port.band * sent_step > 0
            outputs.append((f'kk[{port.band}].out[{1 if up else 1 + sends_first}]',))
        # A strided or folded band reads the frame its converter writes, not the tensor itself.
        in_tensor = self.retiled_frame(node) if self.converts_input(config) else input_tensor_for_role(node, 'lhs').name
        return PortMap(
            inputs={in_tensor: PortBinding('in1', len(inputs), PORT_KIND_BUFFER, tuple(inputs))},
            outputs={node.outputs[0].name: PortBinding('out1', len(outputs), PORT_KIND_BUFFER, tuple(outputs))},
        )
