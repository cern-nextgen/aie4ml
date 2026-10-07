"""Conv2D on the Dense mmul core: one kernel variant and everything it needs to configure it."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from ....aie_types import FloatIntent
from ....errors import ConfigRefused
from ....ir.graph import (
    STAGING_CONTRACTS,
    VIEW_FLATTEN_2D,
    ExecutionInstance,
    OpNode,
    input_role,
    input_tensor_for_role,
)
from ....passes.utils import sanitize_identifier
from ...base import BufferLocation, LayoutConversion, OpImplFootprint, OpImplVariant, StreamLocation, cascade_ports
from ...common_types import PORT_KIND_BUFFER, PORT_KIND_STREAM, PortBinding, PortMap
from ...registry import register_variant
from ...utils import MicrotileShape, ParallelismConfig, TensorView, parse_directives, shared_consumer_spatial_access
from ...utils.math import align_up
from ...utils.precision import (
    aie_rounding_token,
    resolve_accumulator_output_shift,
    resolve_bias_dtype,
    resolve_exact_storage_dtype,
    resolve_operand_precision,
    resolve_output_scale_shift,
)
from ..matmul.common import (
    MICROTILE_OPTIONS,
    describe_inner_output_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    quantize_to_int,
    select_generation_key,
)
from ..matmul.config import MatmulMicrotileConfig
from .common import (
    CHANNEL_BLOCK,
    describe_frame_staging,
    describe_logical_staging,
    frame_view,
    fused_pool_of,
    reads_neighbour_rows,
    spatial_access_of,
)
from .config import Conv2dConfig, Conv2dFlags, FrameRetileConfig
from .frame_retile import FrameRetileOpImplVariant

STREAM_BAND_ROWS = 4
"""Output rows a streamed kernel computes per core call: the band it keeps in local memory.

Four rows keep the halo copy between bands (window - 1 rows) small relative to the work, while
the frame it holds stays a fraction of the whole image.
"""

_SPATIAL_BLOCKS = {'AIE': (2,), 'AIE-ML': (4, 2), 'AIE-MLV2': (4, 2)}
"""Register blockings per generation, mmul row tiles per accumulator set: the first that pads the output width
least, which is the larger one where both fit it (measured faster), else the one computing fewer padded pixels."""


def _core_microtile(generation: str, lhs_width: int) -> Optional[Tuple[int, int, int]]:
    """The mmul the core runs on activations of `lhs_width` bits against int8 weights: the generation's first shape
    of 8-channel K and N blocks, or None where it has none."""
    options = MICROTILE_OPTIONS[generation].get((f'int{lhs_width}', 'int8'), [])
    return next((shape for shape in options if shape[1:] == (CHANNEL_BLOCK, CHANNEL_BLOCK)), None)


def _frame_columns(generation: str) -> Tuple[int, int]:
    """(column block, column alignment) of every frame on the generation, whatever its element width: the int8
    core's widest register block and row tile, which every core's row tile divides. Every op on a tensor derives
    the same frame from it."""
    m = _core_microtile(generation, 8)[0]
    return max(_SPATIAL_BLOCKS[generation]) * m, m


def _depthwise_rows(compact: np.ndarray) -> np.ndarray:
    """[block][ky][tap][channel] from the compact [kh, kw, 1, cout] depthwise weights: each kernel row's taps in
    whole groups of four, the ones past kw zero."""
    kh, kw, _, cout = compact.shape
    blocks = align_up(cout, CHANNEL_BLOCK) // CHANNEL_BLOCK
    taps = np.zeros((kh, align_up(kw, 4), blocks * CHANNEL_BLOCK), dtype=compact.dtype)
    taps[:, :kw, :cout] = compact[:, :, 0, :]
    return taps.reshape(kh, -1, blocks, CHANNEL_BLOCK).transpose(2, 0, 1, 3)


_POINTWISE_STEP_MMULS = 8
"""Mmul issues the pointwise core unrolls in a step at most: its taps (input blocks) times its row tiles. Larger steps
compile much slower on AIE-ML, and are left unpipelined."""


def _uses_pointwise_core(config: Conv2dConfig, in_blocks: int, out_blocks: int) -> bool:
    """Whether a tile runs the pointwise core (conv2d_core_pointwise.h): a 1x1 conv whose step fits
    `_POINTWISE_STEP_MMULS` and where the other cores leave a short loop innermost -- several taps, or one tap over more
    than one pair of output blocks. With one tap and one pair, their innermost loop is already the pixel loop."""
    return (
        tuple(config.spatial.kernel) == (1, 1)
        and in_blocks * config.spatial_blocks <= _POINTWISE_STEP_MMULS
        and (in_blocks > 1 or out_blocks > 2)
    )


def _padded_blocks(blocks: int, pointwise_core: bool) -> int:
    """Output blocks a tile's weights and bias hold: the paired core steps blocks two at a time, so it pads an odd
    count; a tile of one block, and the pointwise core, step one block at a time and need no padding."""
    return blocks if blocks == 1 or pointwise_core else blocks + blocks % 2


@register_variant
class Conv2dOpImplVariant(OpImplVariant):
    """Conv2D with int8 weights as an implicit GEMM over the Dense mmul core, on channel-blocked NHWC frames.

    Partitioning uses the Dense vocabulary on the frame: `cas_length` splits the reduction (input
    channel blocks) across a cascade chain, and `cas_num` splits either the output channel blocks
    ('inner') or the output rows ('outer', row slices that overlap by the window span).
    """

    variant_id = 'conv2d.b.r.v1'
    op_type = 'conv2d'
    graph_header = 'conv2d_graph.h'
    graph_name = 'conv2d_graph'
    param_template = 'conv2d'
    plevel = 10
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        out = node.outputs[0]
        if any(isinstance(t.precision, FloatIntent) for t in (lhs, rhs, out)):
            return False
        widths = (
            resolve_exact_storage_dtype(lhs.precision, namespace='lhs', layer_name=node.name).width,
            resolve_exact_storage_dtype(rhs.precision, namespace='rhs', layer_name=node.name).width,
            resolve_exact_storage_dtype(out.precision, namespace='output', layer_name=node.name).width,
        )
        lhs_width, rhs_width, out_width = widths
        if rhs_width != 8 or lhs_width not in (8, 16) or out_width not in (8, 16):
            return False
        # int16 activations, in or out, need a core for them on the generation
        return _core_microtile(select_generation_key(device.generation), max(lhs_width, out_width)) is not None

    def resolve(self, node: OpNode, device, directives, input_contracts) -> Conv2dConfig:
        io_route, parallel_cfg = parse_directives(directives)
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        out = node.outputs[0]
        spatial = spatial_access_of(node)
        # Kernel limits, as opposed to what the operation means (which the family verified).
        if int(lhs.shape[0]) != 1:
            raise NotImplementedError(f'{node.name}: {self.variant_id} runs one sample per call, got N={lhs.shape[0]}.')
        if spatial.dilations != (1, 1):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} does not implement dilations {spatial.dilations}.'
            )
        if (
            max(spatial.pads[0], spatial.pads[2]) >= spatial.kernel[0]
            or max(spatial.pads[1], spatial.pads[3]) >= spatial.kernel[1]
        ):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} pads {spatial.pads} must stay inside the kernel {spatial.kernel}.'
            )

        precision, accumulator_tag = resolve_operand_precision(node, device)
        precision['bias'] = resolve_bias_dtype(node, precision)
        if spatial.strides[1] > 1 and int(precision['lhs'].width) != 8:
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} implements a horizontal stride for int8 inputs only, '
                f'got {precision["lhs"].width}-bit.'
            )
        generation = select_generation_key(device.generation)
        m, k, n = _core_microtile(generation, int(precision['lhs'].width))
        microtiling = MatmulMicrotileConfig(microtile_m=m, microtile_k=k, microtile_n=n)
        out_w = spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[1]
        spatial_blocks = min(_SPATIAL_BLOCKS[generation], key=lambda blocks: align_up(out_w, blocks * m))

        view = node.traits.get('output_view')
        flatten = view is not None and view.data['kind'] == VIEW_FLATTEN_2D
        parallelism = self._resolve_parallelism(node, parallel_cfg, input_contracts, flatten=flatten)
        block, column_align = _frame_columns(generation)
        outer = parallelism.contract == 'outer'
        row_slices = parallelism.cas_num if outer else 1
        pool = fused_pool_of(node)
        if pool is not None:
            window = pool.window
            if window.kernel != (2, 2) or window.strides != (2, 2) or any(window.pads) or window.dilations != (1, 1):
                raise NotImplementedError(
                    f'{node.name}: {self.variant_id} fuses a 2x2 max pool of stride 2 without pads, not {window}.'
                )
            conv_rows = spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
            if outer and (conv_rows // parallelism.cas_num) % 2:
                raise ConfigRefused(
                    f"{node.name}: each of its {parallelism.cas_num} row slices (contract 'outer') must hold whole "
                    f'pool windows, but {conv_rows} output rows split into odd slices.'
                )

        io_views = {
            lhs.name: frame_view(
                lhs,
                spatial,
                column_block=block,
                column_align=column_align,
                channel_slices=parallelism.cas_length,
                row_slices=row_slices,
            ),
        }
        if flatten:
            if int(rhs.shape[-1]) % CHANNEL_BLOCK:
                raise NotImplementedError(
                    f'{node.name}: a flattened conv needs output channels in whole {CHANNEL_BLOCK}-blocks, '
                    f'got {int(rhs.shape[-1])}.'
                )
            # The Dense LHS the consumer reads: its one row in a block of M, K in 2*microtile_k blocks, one slice per
            # chain.
            chains = parallelism.cas_num
            rows, shard = int(out.shape[0]), int(out.shape[-1]) // chains
            slice_k = align_up(shard, 2 * k)
            io_views[out.name] = TensorView(
                logical=tuple(int(x) for x in out.shape),
                full=(align_up(rows, m), chains * slice_k),
                tile=(align_up(rows, m), slice_k),
                tile_raw=(rows, shard),
                microtile=MicrotileShape(outer=m, inner=k),
            )
        else:
            io_views[out.name] = self._output_frame(node, parallelism, column_block=block, column_align=column_align)

        shift = resolve_accumulator_output_shift(lhs.precision, out.precision, rhs.precision)
        shift += resolve_output_scale_shift(node, is_float=False)
        fused_act = node.traits.get('fused_activation')
        use_relu = fused_act is not None and fused_act.data['activation'] == 'relu'

        return Conv2dConfig(
            precision=precision,
            parallelism=parallelism,
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode=aie_rounding_token(precision['output']),
            spatial_blocks=spatial_blocks,
            spatial=spatial,
            groups=int(node.metadata['groups']),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            bank_mem_bytes=int(device.bank_mem_bytes),
            flags=Conv2dFlags(use_relu=use_relu, emit_flattened=flatten),
            pool=pool,
        )

    def _output_frame(
        self, node, parallelism: ParallelismConfig, *, column_block: int, column_align: int
    ) -> TensorView:
        """The output frame, cut into the chains' shares."""
        outer = parallelism.contract == 'outer'
        view = frame_view(
            node.outputs[0],
            shared_consumer_spatial_access(node.outputs[0]),
            column_block=column_block,
            column_align=column_align,
            channel_slices=1 if outer else parallelism.cas_num,
            row_slices=parallelism.cas_num if outer else 1,
        )
        # Each row slice writes its own rows once: no zero border, and no rows another slice also holds.
        if outer and (any(view.origin) or int(view.tile[1]) * int(parallelism.cas_num) > int(view.full[1])):
            raise ConfigRefused(
                f"{node.name}: an output split by rows (contract 'outer') cannot feed a consumer whose window "
                'reads a zero border or rows past its slice; partition the channels instead, or let the row '
                'slices end at a 1x1 conv or the graph output.'
            )
        return view

    def _resolve_parallelism(self, node, parallel_cfg, input_contracts, *, flatten: bool) -> ParallelismConfig:
        """Tiles over the channel-block axis, in the Dense contract vocabulary."""
        contract = str(parallel_cfg.get('contract', 'inner'))
        if contract not in STAGING_CONTRACTS:
            raise ValueError(f'{node.name}: unknown parallelism contract {contract!r}.')
        lhs = input_tensor_for_role(node, 'lhs')
        spatial = spatial_access_of(node)
        in_blocks = align_up(int(lhs.shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        out_blocks = align_up(int(input_tensor_for_role(node, 'rhs').shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        neighbour_rows = reads_neighbour_rows(spatial)

        cas_length = int(parallel_cfg.get('cas_length', 1))
        producer = input_contracts.get(lhs.name)
        if producer is not None and producer.contract == 'outer':
            # The rows arrive already split ('outer'); a window that reaches past its own row slice
            # would need rows another chain owns.
            if neighbour_rows:
                raise ConfigRefused(
                    f"{node.name}: its input arrives split by rows (contract 'outer'), but its {spatial.kernel} "
                    f'window with pads {spatial.pads} reads rows a neighbouring chain owns; split it into the same '
                    "row bands (parallelism contract 'outer') to read them from its neighbours."
                )
            if contract == 'inner' and 'contract' in parallel_cfg:
                raise ConfigRefused(
                    f"{node.name}: its input is split by rows (contract 'outer'), so it cannot be partitioned "
                    'by channel.'
                )
            if flatten:
                raise ConfigRefused(
                    f"{node.name}: its input arrives split by rows (contract 'outer'), but a flattened output is one "
                    'row that the consuming Dense reads whole.'
                )
            bands = len(producer.port_staging)
            for name, value in (('cas_num', bands), ('cas_length', 1)):
                if name in parallel_cfg and int(parallel_cfg[name]) != value:
                    raise ConfigRefused(
                        f'{node.name}: {name}={parallel_cfg[name]} conflicts with the {bands} row slices its producer '
                        f'writes, which it reads one per tile ({name}={value}).'
                    )
            return ParallelismConfig(cas_num=bands, cas_length=1, contract='outer')
        if producer is not None:
            # A frame is never re-staged on the way, so a chain reads exactly the slices its
            # producer wrote -- the same rule the Dense 'inner' contract follows.
            required = len(producer.port_staging)
            if 'cas_length' in parallel_cfg and cas_length != required:
                raise ConfigRefused(
                    f'{node.name}: cas_length={cas_length} conflicts with the {required} ports its producer '
                    'writes; a spatial frame is handed over slice for slice.'
                )
            cas_length = required
        cas_num = int(parallel_cfg.get('cas_num', 1))
        if contract == 'outer':
            # Row slices overlap by the window span, so a chain reads rows its neighbours also read.
            # Only the graph boundary can serve that: the host clips each port's window against
            # the tensor and zero-fills the rest, while a producing kernel writes each row once.
            if lhs.producer is not None and neighbour_rows:
                raise ConfigRefused(
                    f"{node.name}: an input split by rows (contract 'outer') whose window reads neighbouring rows "
                    'must come from the graph boundary, because the row slices overlap and a kernel writes every '
                    'row exactly once.'
                )
            if flatten:
                raise ConfigRefused(f"{node.name}: a flattened output cannot be split by rows (contract 'outer').")
            out_rows = spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
            if cas_num < 1 or out_rows % cas_num:
                raise ConfigRefused(
                    f'{node.name}: cas_num={cas_num} does not split {out_rows} output rows into equal row slices.'
                )
            if cas_length < 1 or in_blocks % cas_length:
                raise ConfigRefused(
                    f'{node.name}: cas_length={cas_length} does not split {in_blocks} '
                    f'{CHANNEL_BLOCK}-channel blocks evenly.'
                )
            return ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract=contract)
        for name, value, blocks in (('cas_length', cas_length, in_blocks), ('cas_num', cas_num, out_blocks)):
            if value < 1 or blocks % value:
                raise ConfigRefused(
                    f'{node.name}: {name}={value} does not split {blocks} {CHANNEL_BLOCK}-channel blocks evenly.'
                )
        return ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract=contract)

    def validate_config(self, node: OpNode, config: Conv2dConfig, device) -> None:
        if config.shift < 0:
            raise ValueError(f'{node.name}: conv2d accumulator output shift must be non-negative, got {config.shift}.')
        params = self.build_template_params(node, config, {'row': 0, 'col': 0})
        # The kernel computes whole register tiles, so it reads past the last output pixel. A strided
        # frame holds its columns in `stride_w` polyphase classes and every class must reach that
        # far; at stride 1 there is one class, the whole row. Mirrors the kernel's static_assert.
        stride_w = int(config.spatial.strides[1])
        phase_cols = int(params['in_cols']) // stride_w
        read_span = (
            params['out_w_computed']
            + (config.spatial.kernel[1] - 1 + params['in_origin_c'] - config.spatial.pads[1]) // stride_w
        )
        if int(params['in_cols']) % stride_w or read_span > phase_cols:
            raise RuntimeError(
                f'{node.name}: the kernel reads {read_span} columns of each of {stride_w} column '
                f'class(es) but the padded frame holds {params["in_cols"]} columns.'
            )
        if self.input_port_kind == PORT_KIND_BUFFER:
            # Dense's bank schedule: one frame copy per bank (0 and 3) wherever the op contract puts it,
            # the weights in bank 2 of the kernel's tile, stack and bias in bank 1.
            bank = int(config.bank_mem_bytes)
            for what, size, splits in (
                (
                    'input frame',
                    self._frame_bytes(config, 'lhs', params['in_elements']),
                    "`cas_num` over rows ('outer') or `cas_length` over channels",
                ),
                (
                    'output frame',
                    self._frame_bytes(config, 'output', params['out_elements']),
                    "`cas_num` over rows (contract 'outer') or over output channels",
                ),
                (
                    'weights',
                    params['weight_count'],
                    '`cas_length` over input channels or `cas_num` over output '
                    'channels; a split by rows copies the weights to every chain',
                ),
            ):
                if int(size) > bank:
                    raise ConfigRefused(
                        f"{node.name}: each tile's {what} is {size} B but one {device.platform} memory bank holds "
                        f'{bank} B. What shrinks it, where the shape splits evenly: {splits}.'
                    )
            return
        # The stream wrapper owns one frame each way, and its staging, in its own tile.
        tile_bytes = (
            self._frame_bytes(config, 'lhs', params['in_elements'])
            + self._frame_bytes(config, 'output', params['out_elements'])
            + params['weight_count']
            + 4 * params['bias_count']
            + self.staging_bytes(params)
        )
        if tile_bytes > int(device.tile_mem_bytes):
            raise ConfigRefused(
                f'{node.name}: one tile needs {tile_bytes} B for its frames, weights and bias but a '
                f'{device.platform} tile has {device.tile_mem_bytes} B; split the layer with '
                '`parallelism: {cas_num: .., cas_length: ..}`.'
            )

    @staticmethod
    def _frame_bytes(config: Conv2dConfig, role: str, elements: int) -> int:
        return int(elements) * int(config.precision[role].width) // 8

    def staging_bytes(self, _params) -> int:
        """Tile memory the kernel holds beyond its frames. A buffer kernel holds none."""
        return 0

    def uses_depthwise_core(self, node, config: Conv2dConfig) -> bool:
        """Whether one tile runs this depthwise conv (one channel per group) on the channelwise core, which AIE-ML
        and AIE-MLv2 have (sliding_mul_ch; their int8 cores are 4 and 8 rows, AIE1's is 2). Anything else runs the
        mmul core, a depthwise conv as block-diagonal tiles. Row bands keep it, each holding every channel; channel
        chains don't, each owning some channels of a frame that holds them all."""
        lhs = input_tensor_for_role(node, 'lhs')
        widths = tuple(int(config.precision[role].width) for role in ('lhs', 'rhs', 'output'))
        return (
            self.input_port_kind == PORT_KIND_BUFFER
            and config.microtiling.microtile_m > 2
            and int(config.groups) == int(lhs.shape[-1]) == int(input_tensor_for_role(node, 'rhs').shape[-1])
            and widths == (8, 8, 8)
            and (int(config.parallelism.cas_num) == 1 or config.parallelism.contract == 'outer')
            and int(config.parallelism.cas_length) == 1
            and int(config.spatial.strides[1]) == 1
            and config.pool is None
            and not config.flags.emit_flattened
        )

    def retiles_input(self, config: Conv2dConfig) -> bool:
        """Whether this conv reads a frame a retiler built.

        A strided window reads its columns grouped by residue, which neither source provides: the
        boundary carries the tensor in plain order, and a producing kernel writes whole register
        tiles, which span several groups. The stream variant refuses stride altogether.
        """
        return self.input_port_kind == PORT_KIND_BUFFER and int(config.spatial.strides[1]) > 1

    @staticmethod
    def retiled_frame(node) -> str:
        """The execution-only tensor a retiler writes and this conv reads in place of its input."""
        return f'{input_tensor_for_role(node, "lhs").name}__{node.name}_frame'

    def input_conversions(self, node, config: Conv2dConfig, sources):
        if not self.retiles_input(config):
            return ()
        lhs = input_tensor_for_role(node, 'lhs')
        source = sources[lhs.name]
        if source.view is not None:
            raise NotImplementedError(
                f'{node.name}: its strided input is the {source.view.kind} view {source.view.node!r}; a '
                'retiler reads only the boundary tensor or a whole frame a kernel wrote.'
            )
        view = config.io_views[lhs.name]
        from_boundary = source.producer is None
        row_slices = int(config.parallelism.cas_num) if config.parallelism.contract == 'outer' else 1
        if row_slices > 1 and not from_boundary:
            raise ConfigRefused(
                f'{node.name}: its strided input arrives from {source.producer} split by rows; a retiler reads a '
                "producer's frame whole."
            )
        frame = self.retiled_frame(node)
        retile = FrameRetileConfig.for_frame(
            config.precision['lhs'],
            view,
            int(config.spatial.strides[1]),
            source=lhs.name,
            target=frame,
            from_boundary=from_boundary,
            row_slices=row_slices,
            row_step=self._input_row_step(node, config),
            channel_slices=int(config.parallelism.cas_length),
            alternating_horizontal=config.alternating_horizontal,
            bank_mem_bytes=config.bank_mem_bytes,
        )
        # A performance constraint, not a functional one: a retiler beside its conv tile hands the frame
        # over in shared memory, which every single-tile figure was measured with. A cascade's tiles read
        # their inputs in their own row, which the retilers stacked beside it cannot reach, and chains
        # of output channels each read every slice, so there the frame moves by DMA.
        one_reader = int(config.parallelism.cas_length) == 1 and (
            config.parallelism.contract == 'outer' or int(config.parallelism.cas_num) == 1
        )
        return (
            LayoutConversion(
                name=f'{node.name}_retile',
                source=lhs.name,
                target=frame,
                variant=FrameRetileOpImplVariant(),
                config=retile,
                shared_memory=one_reader,
            ),
        )

    def buffer_locations(self, _node, config: Conv2dConfig, anchor_row):
        """Dense's contract (`cascade_ports`), mirroring `place_graph`: each hand-over in banks 0 and 3."""
        return tuple(BufferLocation(g, p, col, row, (0, 3)) for g, p, row, _, col in cascade_ports(config, anchor_row))

    def band_rows(self, node, config: Conv2dConfig) -> int:
        """Output rows one core call covers. A buffer kernel does the whole tile in one call; a
        stream kernel walks the image in bands, keeping only a window of it."""
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = config.spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        if self.input_port_kind == PORT_KIND_BUFFER:
            return out_rows // int(config.parallelism.cas_num) if config.parallelism.contract == 'outer' else out_rows
        band = min(STREAM_BAND_ROWS, out_rows)
        while out_rows % band:
            band -= 1
        return band

    def _tile_extent(self, node, config: Conv2dConfig) -> Tuple[int, int, int, int, int]:
        """What one tile computes per call: its input and output channel blocks, its output rows and
        columns, and the columns it computes (whole register tiles)."""
        lhs = input_tensor_for_role(node, 'lhs')
        _, in_h, in_w, _ = (int(x) for x in lhs.shape)
        in_blocks = int(config.io_views[lhs.name].tile[3]) // CHANNEL_BLOCK
        out_blocks = align_up(int(input_tensor_for_role(node, 'rhs').shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        out_h, out_w = config.spatial.output_extent(in_h, in_w)
        # 'inner' chains own a share of the output channels; 'outer' chains own rows and each
        # computes every channel.
        if config.parallelism.contract == 'inner':
            out_blocks //= int(config.parallelism.cas_num)
        else:
            out_h //= int(config.parallelism.cas_num)
        out_w_computed = align_up(out_w, config.spatial_blocks * config.microtiling.microtile_m)
        return in_blocks, out_blocks, out_h, out_w, out_w_computed

    def fills_border(self, node, config: Conv2dConfig) -> bool:
        """Who puts the zeros around the image: the kernel re-fills the border of a buffer another kernel
        wrote; the host delivers it with the padded window at the boundary, the stream wrapper keeps it in
        a frame it owns, and a retiler builds the whole frame."""
        producer = input_tensor_for_role(node, 'lhs').producer
        return producer is not None and self.input_port_kind == PORT_KIND_BUFFER and not self.retiles_input(config)

    def kernel_params(self, node, config: Conv2dConfig):
        lhs = input_tensor_for_role(node, 'lhs')
        out = node.outputs[0]
        in_view, out_view = config.io_views[lhs.name], config.io_views[out.name]
        _, in_rows, in_cols, _ = (int(x) for x in in_view.tile)
        kh, kw = config.spatial.kernel
        _, in_h, in_w, cin = (int(x) for x in lhs.shape)
        outer = config.parallelism.contract == 'outer'
        cout = int(input_tensor_for_role(node, 'rhs').shape[-1])
        in_blocks, out_blocks, out_h, out_w, out_w_computed = self._tile_extent(node, config)
        band = self.band_rows(node, config)
        streamed = self.input_port_kind == PORT_KIND_STREAM
        depthwise = self.uses_depthwise_core(node, config)
        pointwise = not depthwise and _uses_pointwise_core(config, in_blocks, out_blocks)
        out_blocks_padded = _padded_blocks(out_blocks, pointwise)
        # A band's window starts mid-image, so the image no longer sits at the frame's origin.
        whole_image = not outer
        params = {field: getattr(config, field) for field in config.__dataclass_fields__}
        params.update(
            cin=cin,
            cout=cout,
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            out_blocks_padded=out_blocks_padded,
            in_h=in_h if whole_image else in_rows,
            band_rows=band,
            bands=out_h // band,
            in_w=in_w,
            in_rows=in_rows,
            fills_border=self.fills_border(node, config),
            in_cols=in_cols,
            in_origin_r=int(in_view.origin[1]) if whole_image else 0,
            in_origin_c=int(in_view.origin[2]),
            out_h=out_h,
            out_w=out_w,
            out_w_computed=out_w_computed,
            in_elements=int(np.prod(in_view.tile)),
            out_elements=int(np.prod(out_view.tile)),
            weight_count=(
                out_blocks * kh * align_up(kw, 4) * CHANNEL_BLOCK
                if depthwise
                else kh * kw * in_blocks * out_blocks_padded * CHANNEL_BLOCK**2
            ),
            bias_count=out_blocks_padded * CHANNEL_BLOCK,
            depthwise_core=depthwise,
            pointwise_core=pointwise,
            stream_io=self.input_port_kind == PORT_KIND_STREAM,
            # aie_api's int16 x int8 mmul of 8-channel blocks accumulates in 64 bits whatever it is asked for
            cascade_accumulator_tag='acc64' if int(config.precision['lhs'].width) == 16 else config.accumulator_tag,
        )
        if streamed:
            # A streamed kernel holds one band, not the image: the rows its window reads and the
            # rows it writes. Everything else about the frame is unchanged.
            span_h = config.spatial.window[0]
            params.update(
                in_rows=band + span_h - 1,
                out_h=band,
                out_rows=band,
                out_origin_r=0,
                out_cols=int(out_view.tile[2]),
                out_origin_c=int(out_view.origin[2]),
                flat_k_padded=0,
                in_elements=params['in_blocks'] * (band + span_h - 1) * in_cols * CHANNEL_BLOCK,
                out_elements=params['out_blocks'] * band * int(out_view.tile[2]) * CHANNEL_BLOCK,
            )
            return params
        if config.flags.emit_flattened:
            params.update(out_rows=1, out_cols=1, out_origin_r=0, out_origin_c=0, flat_k_padded=int(out_view.tile[-1]))
        else:
            params.update(
                out_rows=int(out_view.tile[1]),
                out_cols=int(out_view.tile[2]),
                out_origin_r=int(out_view.origin[1]),
                out_origin_c=int(out_view.origin[2]),
                flat_k_padded=0,
            )
        return params

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(width=int(config.parallelism.cas_length), height=int(config.parallelism.cas_num))

    def work(self, node, config) -> int:
        in_blocks, out_blocks, out_h, _, out_w_computed = self._tile_extent(node, config)
        kh, kw = config.spatial.kernel
        if self.uses_depthwise_core(node, config):
            return out_h * out_w_computed * kh * kw * out_blocks * CHANNEL_BLOCK
        blocks = _padded_blocks(out_blocks, _uses_pointwise_core(config, in_blocks, out_blocks))
        return out_h * out_w_computed * kh * kw * in_blocks * blocks * CHANNEL_BLOCK**2

    def output_staging_contract(self, _node, config, _tensor_name):
        return str(config.parallelism.contract)

    def output_port_count(self, _node, config):
        return int(config.parallelism.cas_num)

    def output_inner_shards(self, node, config, _tensor_name):
        """A chain flattens its own output channels pixel by pixel, so the flattened row holds each pixel's channels
        dealt to the chains in turn, stored chain by chain."""
        chains = int(config.parallelism.cas_num)
        if not config.flags.emit_flattened or chains == 1:
            return None
        return chains, int(input_tensor_for_role(node, 'rhs').shape[-1]) // chains

    def _rows_per_chain(self, node, config) -> int:
        """Output rows one chain owns, or 0 when the chains split channels instead."""
        if config.parallelism.contract != 'outer':
            return 0
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = config.spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        return out_rows // int(config.parallelism.cas_num)

    def _input_row_step(self, node, config) -> int:
        """Frame rows between the windows of neighbouring row slices: each one's output rows, strided."""
        return self._rows_per_chain(node, config) * int(config.spatial.strides[0])

    def describe_input_staging(self, node, config, tensor_name, port, _producer=None):
        if self.input_port_kind == PORT_KIND_STREAM:
            return describe_logical_staging(config.io_views[tensor_name], 'read')
        # 'inner': the port is a channel slice every chain reads. 'outer': the port belongs to one
        # (row slice, channel slice) tile, so it selects both. A retiled frame is the same frame, its
        # columns grouped by residue, and the port reads it from the retiler.
        cas_length = int(config.parallelism.cas_length)
        outer = config.parallelism.contract == 'outer'
        row_slice, channel_port = (int(port) // cas_length, int(port) % cas_length) if outer else (0, int(port))
        return describe_frame_staging(
            config.io_views[input_tensor_for_role(node, 'lhs').name],
            'read',
            channel_port,
            row_slice=row_slice,
            row_step=self._input_row_step(node, config),
            column_phases=int(config.spatial.strides[1]) if self.retiles_input(config) else 1,
            fills_border=self.fills_border(node, config),
        )

    def describe_output_staging(self, node, config, tensor_name, port):
        view = config.io_views[tensor_name]
        if self.output_port_kind == PORT_KIND_STREAM:
            return describe_logical_staging(view, 'write')
        if config.flags.emit_flattened:
            return describe_inner_output_staging(view, port)
        outer = config.parallelism.contract == 'outer'
        # 'outer': each chain writes its share of the output rows -- the pooled ones, under a fused pool.
        return describe_frame_staging(
            view,
            'write',
            0 if outer else int(port),
            row_slice=int(port) if outer else 0,
            row_step=int(view.logical[1]) // int(config.parallelism.cas_num) if outer else 0,
        )

    def build_ports(self, node: OpNode, config: Conv2dConfig):
        cas_length = int(config.parallelism.cas_length)
        cas_num = int(config.parallelism.cas_num)
        lhs = input_tensor_for_role(node, 'lhs')
        # A retiled conv reads the frame its retiler writes, not the tensor itself.
        in_tensor = self.retiled_frame(node) if self.retiles_input(config) else lhs.name
        if config.parallelism.contract == 'outer':
            # Every tile reads its own row slice, so no port is shared.
            lhs_endpoints = tuple((f'kk[{tile}].in[0]',) for tile in range(cas_num * cas_length))
        else:
            # One input port per reduction column, multicast to the chain owning each output slice.
            lhs_endpoints = tuple(
                tuple(f'kk[{chain * cas_length + port}].in[0]' for chain in range(cas_num))
                for port in range(cas_length)
            )
        out_endpoints = tuple((f'kk[{chain * cas_length + cas_length - 1}].out[0]',) for chain in range(cas_num))
        return PortMap(
            inputs={in_tensor: PortBinding('in1', len(lhs_endpoints), self.input_port_kind, lhs_endpoints)},
            outputs={node.outputs[0].name: PortBinding('out1', cas_num, self.output_port_kind, out_endpoints)},
        )

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        """Weights per tile as Dense B tiles: [tap (ky, kx, cin block)][cout block][8 x 8], or for the depthwise
        core each channel's own taps (see _depthwise_rows).

        The compact `[kh, kw, Cin/groups, Cout]` tensor expands here into the dense form the mmul
        core consumes -- a grouped conv simply leaves the off-diagonal tiles zero -- and is then
        cut into the (chain, column) slice each tile owns.
        """
        p = inst.config
        weight = input_tensor_for_role(inst.node, 'rhs')
        lhs = input_tensor_for_role(inst.node, 'lhs')
        bias = next((t for t in inst.node.inputs if input_role(inst.node, t.name) == 'bias'), None)
        cas_num, cas_length = int(p.parallelism.cas_num), int(p.parallelism.cas_length)
        wi = weight.precision
        compact = np.asarray(
            quantize_to_int(
                weight.data,
                wi.frac,
                wi.width,
                signed=wi.signed,
                rounding_mode=wi.rounding,
                saturation_mode=wi.saturation,
            )
        )
        kh, kw, cin_g, cout = compact.shape
        groups = int(p.groups)
        cout_g = cout // groups
        in_channels = int(p.io_views[lhs.name].full[-1])
        blocks = align_up(cout, CHANNEL_BLOCK) // CHANNEL_BLOCK
        outer = p.parallelism.contract == 'outer'
        chain_blocks = blocks if outer else blocks // cas_num
        column_blocks = in_channels // CHANNEL_BLOCK // cas_length
        chain_blocks_padded = _padded_blocks(chain_blocks, _uses_pointwise_core(p, column_blocks, chain_blocks))

        if self.uses_depthwise_core(inst.node, p):
            packed_weights = np.tile(_depthwise_rows(compact).reshape(1, 1, -1), (cas_num, 1, 1))  # every band
        else:
            dense = np.zeros((kh, kw, in_channels, blocks * CHANNEL_BLOCK), dtype=np_dtype_for_spec(p.precision['rhs']))
            for group in range(groups):
                dense[:, :, group * cin_g : (group + 1) * cin_g, group * cout_g : (group + 1) * cout_g] = compact[
                    :, :, :, group * cout_g : (group + 1) * cout_g
                ]
            # (kh, kw, cb, 8, nb, 8) -> tap-major (ky, kx, cb) x (nb) x (8 x 8), then cut per tile.
            tiles = dense.reshape(kh, kw, in_channels // CHANNEL_BLOCK, CHANNEL_BLOCK, blocks, CHANNEL_BLOCK).transpose(
                0, 1, 2, 4, 3, 5
            )
            packed_weights = np.zeros(
                (cas_num, cas_length, kh * kw * column_blocks * chain_blocks_padded * CHANNEL_BLOCK**2),
                dtype=tiles.dtype,
            )
            for chain in range(cas_num):
                block_base = 0 if outer else chain * chain_blocks
                for column in range(cas_length):
                    tile = np.zeros(
                        (kh, kw, column_blocks, chain_blocks_padded, CHANNEL_BLOCK, CHANNEL_BLOCK), dtype=tiles.dtype
                    )
                    tile[:, :, :, :chain_blocks] = tiles[
                        :,
                        :,
                        column * column_blocks : (column + 1) * column_blocks,
                        block_base : block_base + chain_blocks,
                    ]
                    packed_weights[chain, column] = tile.reshape(-1)

        packed_bias = np.zeros(
            (cas_num, chain_blocks_padded * CHANNEL_BLOCK), dtype=np_bias_dtype_for_spec(p.precision['bias'])
        )
        if bias is not None:
            bi = bias.precision
            values = np.asarray(
                quantize_to_int(
                    bias.data,
                    lhs.precision.frac + wi.frac,
                    int(p.precision['bias'].width),
                    signed=bi.signed,
                    rounding_mode=bi.rounding,
                    saturation_mode=bi.saturation,
                )
            ).reshape(-1)
            for chain in range(cas_num):
                base = 0 if outer else chain * chain_blocks * CHANNEL_BLOCK
                chunk = values[base : base + chain_blocks * CHANNEL_BLOCK]
                packed_bias[chain, : chunk.size] = chunk
        return {'packed_weights': packed_weights, 'packed_bias': packed_bias}

    def get_artifacts(self, inst: ExecutionInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        return [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': p.precision['rhs'].c_type,
                'storage_dtype': p.precision['rhs'].storage_dtype,
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            },
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_bias'],
                'dtype': p.precision['bias'].c_type,
                'storage_dtype': p.precision['bias'].storage_dtype,
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            },
        ]


@register_variant
class Conv2dStreamOpImplVariant(Conv2dOpImplVariant):
    """The same Conv2D on core streams: the frame crosses on the wire in linear row order and the
    kernel lands it in the blocked layout the compute core reads (`ports: stream`).

    It buys legality rather than speed: a frame of several channel blocks crosses the graph
    boundary in one port, which a DMA-fed frame cannot do.
    """

    variant_id = 'conv2d.s.r.v1'
    input_port_kind = output_port_kind = PORT_KIND_STREAM

    def resolve(self, node: OpNode, device, directives, input_contracts) -> Conv2dConfig:
        config = super().resolve(node, device, directives, input_contracts)
        if (int(config.precision['lhs'].width), int(config.precision['output'].width)) != (8, 8):
            raise NotImplementedError(f'{node.name}: {self.variant_id} streams int8 frames only.')
        if config.spatial.strides != (1, 1):
            raise NotImplementedError(
                f'{node.name}: {self.variant_id} does not implement strides {config.spatial.strides}; the '
                'band it keeps and the columns it places are both written for a dense window.'
            )
        if config.parallelism.cas_num != 1 or config.parallelism.cas_length != 1:
            raise ConfigRefused(f'{node.name}: {self.variant_id} does not implement partitioning yet.')
        if config.flags.emit_flattened:
            raise NotImplementedError(f'{node.name}: {self.variant_id} writes a frame, not a flattened row.')
        if config.pool is not None:
            raise NotImplementedError(f'{node.name}: {self.variant_id} does not fuse a pool yet.')
        return config

    def validate_config(self, node: OpNode, config: Conv2dConfig, device) -> None:
        super().validate_config(node, config, device)
        params = self.build_template_params(node, config, {'row': 0, 'col': 0})
        lhs = input_tensor_for_role(node, 'lhs')
        out_rows = config.spatial.output_extent(int(lhs.shape[1]), int(lhs.shape[2]))[0]
        if int(params['bands']) * int(params['band_rows']) != out_rows:
            raise NotImplementedError(
                f'{node.name}: {out_rows} output rows do not divide into whole bands of {params["band_rows"]}.'
            )
        beat = 16  # bytes in one 128-bit stream access
        for name, elements in (
            ('input', int(np.prod(lhs.shape))),
            ('output', int(np.prod(node.outputs[0].shape))),
        ):
            if elements % beat:
                raise NotImplementedError(
                    f'{node.name}: its {name} is {elements} bytes, which is not a whole number of '
                    f'{beat}-byte stream beats.'
                )

    def buffer_locations(self, _node, _config, _anchor_row):
        return ()  # core streams: no buffer to place

    def stream_locations(self, _node, config, anchor_row):
        return tuple(StreamLocation(g, p, col, row) for g, p, row, col, _ in cascade_ports(config, anchor_row))

    def staging_bytes(self, params) -> int:
        """A channel count that does not fill a block stages the band's bytes in a buffer: the
        wire reads into one before placing, and gathers into one before writing, because a stream
        access inside either loop stops it pipelining."""
        total = 0
        if int(params['cin']) % CHANNEL_BLOCK:
            total += int(params['in_rows']) * int(params['in_w']) * int(params['cin']) + 16
        if int(params['cout']) % CHANNEL_BLOCK:
            total += int(params['band_rows']) * int(params['out_w']) * int(params['cout']) + 16
        return total

    def footprint(self, _node, _config) -> OpImplFootprint:
        return OpImplFootprint(width=1, height=1)
