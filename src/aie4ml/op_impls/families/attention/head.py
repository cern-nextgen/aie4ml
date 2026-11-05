from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from ....errors import ConfigRefused
from ....ir.graph import OpNode, has_input_role, input_tensor_for_role
from ....quant_utils import quantize_to_int
from ...base import PING_PONG_BANKS, BufferLocation, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import MicrotileShape, ParallelismConfig, align_up, build_tensor_view, ceildiv
from ...utils.tiling import ROW_HEIGHTS
from ..matmul.common import describe_outer_lhs_staging, describe_outer_output_staging
from .common import (
    BIAS_BLOCK_CYCLES,
    ROW_BANDS,
    TABLE_BYTES,
    attends,
    attention_params,
    bias_locations,
    bias_shift,
    bias_view,
    chain_tiles,
    chains,
    check_bank,
    kernel_microtile,
    shift_of,
    softmax_arithmetic,
    spill_bytes,
    stack_bytes,
    storage_precision,
    tokens_of,
)
from .config import AttentionHeadConfig

PROJECTIONS = ('q', 'k', 'v')
#: x's columns in two halves, one per input DMA channel of every tile, as a Dense split in two along K reads them.
X_HALVES = ParallelismConfig(cas_num=1, cas_length=2)


@register_variant
class AttentionHeadOpImplVariant(OpImplVariant):
    """AttentionOpImplVariant with Q, K and V projected from x on the tiles reading them: each chain recomputes every
    key's K and V, which saves the Dense tiles and their hand-overs."""

    variant_id = 'attention.head.i8.v1'
    op_type = 'attention'
    graph_header = 'attention_head_graph.h'
    graph_name = 'attention_head_graph'
    param_template = 'attention_head'
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism', 'microtiling'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        return _projects(node, device)

    def resolve(self, node: OpNode, device, directives, _input_contracts) -> AttentionHeadConfig:
        cas_num, cas_length = chains(node, device, directives)
        x, out = input_tensor_for_role(node, 'x'), node.outputs[0]
        tokens, head_dim = (int(d) for d in out.shape)
        mo = _out_microtile(node, device, directives)
        model_dim, out_columns = _geometry(node, mo)
        projections = node.metadata['projections']
        fracs = [int(projections[role]['frac']) for role in PROJECTIONS]
        shifts = tuple(
            int(x.precision.frac) + int(input_tensor_for_role(node, f'w{role}').precision.frac) - frac
            + shift_of(float(projections[role]['output_scale']), f'{role} projection scale', node)
            for role, frac in zip(PROJECTIONS, fracs)
        )  # fmt: skip
        if min(shifts) < 0:
            raise ValueError(f'{node.name}: a projection finer than its products ({shifts}).')
        stack = _check_fits(node, device, cas_num, cas_length, model_dim, out_columns)
        io_views = {  # x in column panels (a microtile of every token by 8), which a DMA walks from a PLIO at once
            x.name: build_tensor_view(
                node, x, 'inputs', full_inner=model_dim, tile_inner=model_dim // 2,
                tile_inner_raw=ceildiv(int(x.shape[1]), 2), full_outer=tokens, tile_outer=tokens, tile_outer_raw=tokens,
                microtile=MicrotileShape(outer=tokens, inner=8),
            ),
            out.name: build_tensor_view(
                node, out, 'outputs', full_inner=out_columns, tile_inner=out_columns, tile_inner_raw=head_dim,
                full_outer=tokens, tile_outer=tokens // cas_num, tile_outer_raw=tokens // cas_num, microtile=mo,
            ),
        }  # fmt: skip
        if has_input_role(node, 'bias'):
            io_views[input_tensor_for_role(node, 'bias').name] = bias_view(node, cas_num)
        return AttentionHeadConfig(
            precision=storage_precision(node, [('x', x), ('output', out)]),
            parallelism=ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract='outer'),
            io_views=io_views,
            alternating_horizontal=device.alternating_horizontal,
            columns=head_dim,
            **softmax_arithmetic(node, *fracs, int(out.precision.frac)),
            table_bank_bits=int(device.lookup_bank_bits),
            stack_bytes=stack,
            int8_macs_per_cycle=int(device.int8_macs_per_cycle),
            model_dim=model_dim,
            out_columns=out_columns,
            looked_blocks=tokens // 8 if cas_length == 1 else tokens // 16,
            projection_shifts=shifts,
        )

    def kernel_params(self, node: OpNode, config: AttentionHeadConfig):
        params = {
            **attention_params(node, config),
            'model_dim': config.model_dim,
            'out_columns': config.out_columns,
            'looked_blocks': config.looked_blocks,
            'o_rows': int(config.io_views[node.outputs[0].name].microtile.outer),
            'projection_shifts': dict(zip(PROJECTIONS, config.projection_shifts)),
        }
        for role in PROJECTIONS:
            params[f'w{role}'], params[f'b{role}'] = _packed(node, role, config)
        return params

    def build_ports(self, node: OpNode, config: AttentionHeadConfig) -> PortMap:
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        x_port = [0] if c == 1 else [0, 1]  # the values' tile reads its cascade first
        halves = X_HALVES.cas_length
        # both halves for the chains' first tiles, then for their last: a first tile's x never waits on a last tile
        return PortMap(
            inputs={
                input_tensor_for_role(node, 'x').name: PortBinding(
                    'in1',
                    c * halves,
                    endpoints=tuple(
                        tuple(f'kk[{i * c + p}].in[{x_port[p] + half}]' for i in range(n))
                        for p in range(c)
                        for half in range(halves)
                    ),
                )
            },
            outputs={
                node.outputs[0].name: PortBinding(
                    'out1', n, endpoints=tuple((f'kk[{i * c + c - 1}].out[0]',) for i in range(n))
                )
            },
        )

    def describe_input_staging(self, _node, config, tensor_name, port):
        """Every token, half of x's columns a port, laid out as a Dense split in two along K reads its input."""
        return describe_outer_lhs_staging(config.io_views[tensor_name], X_HALVES, port % X_HALVES.cas_length)

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_outer_output_staging(config.io_views[tensor_name], port)

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return 'outer'

    def buffer_locations(self, _node: OpNode, config: AttentionHeadConfig, anchor_row: int):
        """Both halves of x in each tile's own memory, a chain's output after its last tile (as a row-wise op's)."""
        halves = X_HALVES.cas_length
        locations = []
        for chain, flow, cols in chain_tiles(config, anchor_row):
            for p, col in enumerate(cols):  # the chain's tiles in order, as `build_ports` lists them
                for half in range(halves):  # the halves' pings in different banks: a projection loads both at once
                    banks = PING_PONG_BANKS[::-1] if half else PING_PONG_BANKS
                    locations.append(BufferLocation('in1', p * halves + half, col, chain, banks, reader=(col, chain)))
            locations.append(BufferLocation('out1', chain, cols[-1] + flow.output_col, chain, PING_PONG_BANKS))
        return tuple(locations)

    def work(self, node: OpNode, config: AttentionHeadConfig) -> int:
        """The slower tile's cycles per call, its projections and its share of the attention, as its core's
        multiply-accumulates: fitted to AIE-ML profiles at 32 and 64 tokens, model dimensions 32 and 64."""
        par = config.parallelism
        qb, kb = tokens_of(node) // par.cas_num // 8, tokens_of(node) // 8
        panels = config.model_dim // 8 * config.columns // 8  # 8x8 tiles of one token block's projection
        bias = qb * kb * BIAS_BLOCK_CYCLES[config.bias_bits] if config.bias_bits else 0  # on the first tile
        if par.cas_length == 1:
            cycles = 3.62 * (2 * kb + qb) * panels + 50.8 * qb * kb + bias
        else:
            cycles = 3.62 * (kb + qb) * panels + 37.4 * qb * kb + bias
        return round(4.5 + cycles) * config.int8_macs_per_cycle


@register_variant
class AttentionHeadBiasOpImplVariant(AttentionHeadOpImplVariant):
    """AttentionHeadOpImplVariant with an additive bias on the logits (attention_head_bias.h): a chain's first tile
    reads its queries' rows of it, as AttentionBiasOpImplVariant's."""

    variant_id = 'attention.head.bias.i8.v1'
    graph_header = 'attention_head_bias_graph.h'
    graph_name = 'attention_head_bias_graph'
    supported_fusions: ClassVar[frozenset] = frozenset({'bias'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not (has_input_role(node, 'bias') and _projects(node, device)):
            return False
        projections = node.metadata['projections']
        return bias_shift(node, sum(int(projections[role]['frac']) for role in ('q', 'k'))) is not None

    def build_ports(self, node: OpNode, config: AttentionHeadConfig) -> PortMap:
        ports = super().build_ports(node, config)
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        u = PortBinding('in2', n, endpoints=tuple((f'kk[{i * c}].in[2]',) for i in range(n)))
        return PortMap(inputs={**ports.inputs, input_tensor_for_role(node, 'bias').name: u}, outputs=ports.outputs)

    def describe_input_staging(self, node, config, tensor_name, port):
        if tensor_name == input_tensor_for_role(node, 'bias').name:  # a chain's rows, as the attention's Q
            return describe_outer_lhs_staging(config.io_views[tensor_name], ROW_BANDS, port)
        return super().describe_input_staging(node, config, tensor_name, port)

    def buffer_locations(self, node: OpNode, config: AttentionHeadConfig, anchor_row: int):
        """As AttentionHeadOpImplVariant's, and a chain's rows of the bias as AttentionBiasOpImplVariant's."""
        return (*super().buffer_locations(node, config, anchor_row), *bias_locations(config, anchor_row, 'in2'))


def _projects(node: OpNode, device) -> bool:
    """Whether the head variants take the node, a bias aside: x's second half holds some of its columns, and the head
    fits as chains of two splitting the queries most."""
    if not (has_input_role(node, 'x') and attends(node, device)) or int(input_tensor_for_role(node, 'x').shape[1]) <= 8:
        return False
    rows = int(device.rows) - int(device.row_start)
    cas_num = max(n for n in range(1, rows + 1) if (tokens_of(node) // 8) % n == 0)
    try:
        _check_fits(node, device, cas_num, 2, *_geometry(node, _out_microtile(node, device, {})))
    except ConfigRefused:
        return False
    return True


def _out_microtile(node: OpNode, device, directives) -> MicrotileShape:
    """The microtile O is written in: the row band a `microtiling` directive gives (the design search's, so the Dense
    after it reads O directly), else the generation's first (`ROW_HEIGHTS`)."""
    mt = directives.get('microtiling')
    if mt is None:
        return MicrotileShape(outer=ROW_HEIGHTS[device.generation][('int8', 'int8')][0], inner=8)
    shape = MicrotileShape(outer=int(mt.get('microtile_m', 0)), inner=int(mt.get('microtile_n', 0)))
    if set(mt) != {'microtile_m', 'microtile_n'} or kernel_microtile(shape) != shape:
        raise ValueError(f'{node.name}: the head writes O in 8-wide bands of 4 or 8 rows, got microtiling {mt}.')
    return shape


def _geometry(node: OpNode, mo: MicrotileShape) -> Tuple[int, int]:
    """x's stored columns (two halves of whole 8s), and O's (padded as the Dense reading it in `mo` pads its input)."""
    x, out = input_tensor_for_role(node, 'x'), node.outputs[0]
    return 2 * align_up(ceildiv(int(x.shape[1]), 2), 8), align_up(int(out.shape[1]), 2 * int(mo.inner))


def _check_fits(node: OpNode, device, cas_num: int, cas_length: int, model_dim: int, out_columns: int) -> int:
    """The stack a chain's busiest tile keeps, where its buffers, tables, weights and scratch fit its banks: all three
    projections and K's and V's token scratch on a lone tile, Q's and K's and K's on a pair's first."""
    tokens, head_dim = (int(d) for d in node.outputs[0].shape)
    projections, scratches = (3, 2) if cas_length == 1 else (2, 1)
    # in each ping-pong bank: every token, and on a chain's last tile its output
    check_bank(node, device, tokens * model_dim + tokens // cas_num * out_columns, 'buffers')
    # in the weight bank: the other copy of each table, and the weights and biases
    weights = projections * (model_dim * head_dim + 4 * head_dim)
    check_bank(node, device, 2 * TABLE_BYTES + weights, 'tables, weights and biases')
    # in the stack bank: one copy of each table (both on every tile but a pair's first) and the kernel
    spill = spill_bytes(device, 'head', cas_length, tokens, head_dim, model_dim)
    return stack_bytes(node, device, 2, spill, scratches * tokens * head_dim)


def _packed(node: OpNode, role: str, config: AttentionHeadConfig):
    """A projection's weights as 8x8 tiles (model block, head block), zero past the model dimension, and its bias at
    the products' fixed point, as its Dense quantizes them."""
    x, w = input_tensor_for_role(node, 'x'), input_tensor_for_role(node, f'w{role}')
    wi = w.precision
    weights = quantize_to_int(w.data, wi.frac, wi.width, wi.signed, wi.rounding, wi.saturation)
    rows, head_dim = (int(d) for d in np.shape(weights))
    padded = np.zeros((config.model_dim, config.columns), dtype=np.int64)
    padded[:rows, :head_dim] = weights
    tiles = padded.reshape(config.model_dim // 8, 8, config.columns // 8, 8).transpose(0, 2, 1, 3).reshape(-1)
    bias = np.zeros(config.columns, dtype=np.int64)
    if has_input_role(node, f'b{role}'):
        b = input_tensor_for_role(node, f'b{role}')
        bi = b.precision
        bias[:head_dim] = quantize_to_int(
            np.reshape(b.data, -1), int(x.precision.frac) + int(wi.frac), 32, bi.signed, bi.rounding, bi.saturation
        )
    return [int(v) for v in tiles], [int(v) for v in np.reshape(bias, -1)]
