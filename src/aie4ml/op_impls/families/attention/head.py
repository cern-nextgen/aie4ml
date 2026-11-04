from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from ....errors import ConfigRefused
from ....ir.graph import OpNode, has_input_role, input_tensor_for_role
from ....quant_utils import quantize_to_int
from ...base import PING_PONG_BANKS, BufferLocation, OpImplVariant, row_flow
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import MicrotileShape, ParallelismConfig, align_up, build_tensor_view, ceildiv
from ...utils.precision import resolve_exact_storage_dtype
from ..matmul.common import describe_outer_lhs_staging, describe_outer_output_staging
from ..matmul.resolver import default_lhs_microtile
from .config import AttentionHeadConfig
from .fused import (
    MAX_TOKENS,
    MICROTILE,
    TABLE_BANK_BITS,
    TABLE_BYTES,
    attention_params,
    chains,
    check_bank,
    ranked_cycles,
    shift_of,
    softmax_arithmetic,
    tokens_of,
)

PROJECTIONS = ('q', 'k', 'v')
#: x's columns in two halves, one per input DMA channel of every tile, as a Dense split in two along K reads them.
X_HALVES = ParallelismConfig(cas_num=1, cas_length=2)
#: The least stack a tile keeps: the shared attention code spills up to 4.6 KB (64 keys, head dimension 32).
MIN_STACK_BYTES = 4096


@register_variant
class AttentionHeadOpImplVariant(OpImplVariant):
    """softmax(Q Kᵀ) V as AttentionOpImplVariant computes it, with Q, K and V projected from x on the tiles that read
    them: a chain's first tile projects every key and its own queries, its last every value. Each chain recomputes the
    keys and values one Dense each would compute once, and saves those Dense tiles and their hand-overs."""

    variant_id = 'attention.head.i8.v1'
    op_type = 'attention'
    graph_header = 'attention_head_graph.h'
    graph_name = 'attention_head_graph'
    param_template = 'attention_head'
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not (has_input_role(node, 'x') and device.generation in TABLE_BANK_BITS and tokens_of(node) <= MAX_TOKENS):
            return False
        # x's second half holds some of its columns, and the head fits as chains of two splitting the queries most
        if int(input_tensor_for_role(node, 'x').shape[1]) <= 8:
            return False
        rows = int(device.rows) - int(device.row_start)
        cas_num = max(n for n in range(1, rows + 1) if (tokens_of(node) // 8) % n == 0)
        try:
            _check_fits(node, device, cas_num, 2, *_geometry(node, device)[:2])
        except ConfigRefused:
            return False
        return True

    def resolve(self, node: OpNode, device, directives, _input_contracts) -> AttentionHeadConfig:
        cas_num, cas_length = chains(node, device, directives)
        x, out = input_tensor_for_role(node, 'x'), node.outputs[0]
        tokens, head_dim = (int(d) for d in out.shape)
        model_dim, out_columns, precision, mo = _geometry(node, device)
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
        return AttentionHeadConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract='outer'),
            io_views=io_views,
            alternating_horizontal=device.alternating_horizontal,
            columns=head_dim,
            **softmax_arithmetic(node, *fracs, int(out.precision.frac)),
            table_bank_bits=TABLE_BANK_BITS[device.generation],
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
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        halves = X_HALVES.cas_length
        locations = []
        for chain in range(n):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + chain, c)
            cols = [c - 1 - p if flow.reversed else p for p in range(c)]
            for p, col in enumerate(cols):  # the chain's tiles in order, as `build_ports` lists them
                for half in range(halves):  # the halves' pings in different banks: a projection loads both at once
                    banks = PING_PONG_BANKS[::-1] if half else PING_PONG_BANKS
                    locations.append(BufferLocation('in1', p * halves + half, col, chain, banks, reader=(col, chain)))
            locations.append(BufferLocation('out1', chain, cols[-1] + flow.output_col, chain, PING_PONG_BANKS))
        return tuple(locations)

    def work(self, node: OpNode, config: AttentionHeadConfig) -> int:
        """The slower tile's cycles per call, as multiply-accumulates: its projections' 8x8 tiles (every key's K, or V,
        and the chain's queries' Q on the first tile; all three on a lone tile) and its share of the attention per query
        and key block, fitted to the kernels' AIE-ML profiles at 32 and 64 tokens, model dimensions 32 and 64."""
        par = config.parallelism
        qb, kb = int(node.outputs[0].shape[0]) // par.cas_num // 8, int(node.outputs[0].shape[0]) // 8
        panels = config.model_dim // 8 * config.columns // 8  # 8x8 tiles of one token block's projection
        if par.cas_length == 1:
            cycles = 3.62 * (2 * kb + qb) * panels + 50.8 * qb * kb
        else:
            cycles = 3.62 * (kb + qb) * panels + 37.4 * qb * kb
        return ranked_cycles(4.5 + cycles, config) * config.int8_macs_per_cycle


def _geometry(node: OpNode, device):
    """x's stored columns (two halves of whole 8s), O's (padded as the Dense after it pads its input), the storage
    dtypes, and the microtile that Dense reads O in, where the kernel stores it."""
    x, out = input_tensor_for_role(node, 'x'), node.outputs[0]
    precision = {
        'x': resolve_exact_storage_dtype(x.precision, namespace='x', layer_name=node.name),
        'output': resolve_exact_storage_dtype(out.precision, namespace='output', layer_name=node.name),
    }
    mo = default_lhs_microtile(device.generation, precision['output'], precision['output'])
    mo = mo if mo.inner == 8 and mo.outer in (4, 8) else MICROTILE
    model_dim = 2 * align_up(ceildiv(int(x.shape[1]), 2), 8)
    return model_dim, align_up(int(out.shape[1]), 2 * int(mo.inner)), precision, mo


def _check_fits(node: OpNode, device, cas_num: int, cas_length: int, model_dim: int, out_columns: int) -> int:
    """The stack a tile keeps, where its buffers, scratch, tables and weights fit its banks."""
    tokens, head_dim = (int(d) for d in node.outputs[0].shape)
    # in each ping-pong bank: every token, and on a chain's last tile its output
    check_bank(node, device, tokens * model_dim + tokens // cas_num * out_columns)
    _check_weight_bank(node, device, cas_length, model_dim * head_dim + 4 * head_dim)
    return _stack_bytes(node, device, cas_length, tokens * head_dim)


def _tiles_projecting(cas_length: int) -> Tuple[int, int]:
    """(projections, token scratches) on a chain's busiest tile: all three and K's and V's on one tile, else Q and K
    and K's on the first."""
    return (3, 2) if cas_length == 1 else (2, 1)


def _stack_bytes(node: OpNode, device, cas_length: int, scratch: int) -> int:
    """The stack bank holds the stack, one copy of each table (both on every tile but a pair's first) and the kernel,
    whose K and V scratch take the rest."""
    _, scratches = _tiles_projecting(cas_length)
    stack = int(device.bank_mem_bytes) - 2 * TABLE_BYTES - 1024 - scratches * scratch
    if stack < MIN_STACK_BYTES:
        raise ConfigRefused(
            f'{node.name}: its {scratches} token scratches leave its stack {stack} B, under {MIN_STACK_BYTES}.'
        )
    return stack


def _check_weight_bank(node: OpNode, device, cas_length: int, projection: int) -> None:
    """The weight bank holds the other copy of each table and the tile's weights and biases."""
    projections, _ = _tiles_projecting(cas_length)
    held = 2 * TABLE_BYTES + projections * projection
    if held > int(device.bank_mem_bytes):
        raise ConfigRefused(
            f'{node.name}: {held} B of tables, weights and biases in a bank of {device.bank_mem_bytes}.'
        )


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
