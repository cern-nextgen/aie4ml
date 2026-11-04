from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, Tuple

from ....errors import ConfigRefused
from ....ir.graph import OpNode, has_input_role, input_tensor_for_role
from ...base import PING_PONG_BANKS, BufferLocation, OpImplVariant, row_flow
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import MicrotileShape, ParallelismConfig, build_tensor_view
from ...utils.precision import resolve_exact_storage_dtype
from ...utils.tensor_view import microtile_from_staging
from ..matmul.common import describe_outer_lhs_staging, describe_outer_output_staging
from .config import AttentionConfig

#: The row sums of the uint8 exp (255 at a row's maximum) stay in int16 up to 128 keys.
MAX_TOKENS = 128
#: A lookup table repeats each chunk every data-memory bank width: 128 bits on AIE-ML, 256 on AIE-MLv2.
TABLE_BANK_BITS = {'AIE-ML': 128, 'AIE-MLV2': 256}
#: The division: round(2^RECIP_SHIFT / n) seeds n in [2^14, 2^15) at round((n - 2^14) / 64); theta is kept
#: / 2^THETA_SHIFT.
RECIP_SHIFT, THETA_SHIFT = 28, 7
RECIP_TABLE = tuple(round(2**RECIP_SHIFT / (2**14 + 64 * i)) for i in range(256))
MICROTILE = MicrotileShape(outer=8, inner=8)
#: Each input port is one row band, never a slice of the head dimension.
ROW_BANDS = ParallelismConfig(cas_num=1)
#: One copy of a lookup table: its 256 entries twice, in int16 slots (lut_layout).
TABLE_BYTES = 1024
#: The exp table's last entry rounds to zero where 255 steps span this: 255 exp(-EXP_SPAN) < 1/2.
EXP_SPAN = math.log(510)


def tokens_of(node: OpNode) -> int:
    return int(node.outputs[0].shape[0])


def stored_microtile(contract) -> MicrotileShape:
    """The microtile a producer stores the tensor in where the kernel reads it (8 wide, 4 or 8 rows), else 8x8."""
    mt = microtile_from_staging(contract.port_staging[0]) if contract is not None and contract.port_staging else None
    return mt if mt is not None and mt.inner == 8 and mt.outer in (4, 8) else MICROTILE


def shift_of(scale: float, what: str, node: OpNode) -> int:
    """-log2(scale) of a power-of-two scale <= 1."""
    shift = -math.log2(scale) if scale > 0 else math.nan
    if not (shift >= 0 and shift == int(shift)):
        raise ValueError(f'{node.name}: the fused attention takes a power-of-two {what} of at most 1, got {scale}.')
    return int(shift)


def lut_layout(values, bank_bits: int, uint8: bool = False):
    """An aie::lut<4, T> table in int16 slots: each bank width's chunk stored twice in a row. A uint8 table goes in as
    the int8 of the same bits (an 8-bit table saturates each slot to int8)."""
    chunk = bank_bits // 16
    out = [0] * (2 * len(values))
    for d, v in enumerate(values):
        v = int(v) - 256 if uint8 and v > 127 else int(v)
        at = (d // chunk) * 2 * chunk + d % chunk
        out[at] = out[at + chunk] = v
    return out


@register_variant
class AttentionOpImplVariant(OpImplVariant):
    """One head's softmax(Q Kᵀ) V with the probabilities unnormalised (exp(0) = 255) and each output divided once:
    within a code of the QDQ graph, which rounds every normalised probability to uint8 first, and as accurate. Query
    chains split the queries; a chain is one tile, or a tile that scores the keys and hands the probabilities down the
    cascade to one that weighs the values and divides. The exp and reciprocal are lookups, which AIE cores have none
    of."""

    variant_id = 'attention.i8.v1'
    op_type = 'attention'
    graph_header = 'attention_graph.h'
    graph_name = 'attention_graph'
    param_template = 'attention'
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        return has_input_role(node, 'q') and device.generation in TABLE_BANK_BITS and tokens_of(node) <= MAX_TOKENS

    def resolve(self, node: OpNode, device, directives, input_contracts) -> AttentionConfig:
        cas_num, cas_length = chains(node, device, directives)
        q, k, v = (input_tensor_for_role(node, role) for role in ('q', 'k', 'v'))
        out = node.outputs[0]
        tokens, head_dim = (int(x) for x in q.shape)
        # the head dimension as the producers store it: a Dense pads its columns to whole steps with zeros, which the
        # kernel reads as more of the head dimension (zero scores, zero outputs), so the hand-overs stay direct
        contracts = [input_contracts.get(t.name) for t in (q, k, v)]
        columns = max([head_dim, *(int(c.view.full_inner) for c in contracts if c is not None and c.view is not None)])
        # in each ping-pong bank: every key on a chain's first tile, every value and the chain's output on its last (on
        # one tile, all three)
        check_bank(node, device, (3 - cas_length) * tokens * columns + tokens // cas_num * columns)
        precision = {
            role: resolve_exact_storage_dtype(t.precision, namespace=role, layer_name=node.name)
            for role, t in (('q', q), ('k', k), ('v', v), ('output', out))
        }
        fracs = [int(t.precision.frac) for t in (q, k, v, out)]

        def view(tensor, direction, rows, microtile):
            return build_tensor_view(
                node, tensor, direction, full_inner=columns, tile_inner=columns, tile_inner_raw=head_dim,
                full_outer=tokens, tile_outer=rows, tile_outer_raw=rows, microtile=microtile,
            )  # fmt: skip

        # each buffer in its producer's microtile; O in Q's, its rows the same
        mq, mk, mv = (stored_microtile(c) for c in contracts)
        io_views = {
            q.name: view(q, 'inputs', tokens // cas_num, mq),
            k.name: view(k, 'inputs', tokens, mk),
            v.name: view(v, 'inputs', tokens, mv),
            out.name: view(out, 'outputs', tokens // cas_num, mq),
        }
        return AttentionConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract='outer'),
            io_views=io_views,
            alternating_horizontal=device.alternating_horizontal,
            columns=columns,
            **softmax_arithmetic(node, *fracs),
            table_bank_bits=TABLE_BANK_BITS[device.generation],
            stack_bytes=stack_bytes(device, cas_length),
            int8_macs_per_cycle=int(device.int8_macs_per_cycle),
        )

    def kernel_params(self, node: OpNode, config: AttentionConfig):
        roles = {role: input_tensor_for_role(node, role).name for role in ('q', 'k', 'v')}
        roles['output'] = node.outputs[0].name
        return {
            **attention_params(node, config),
            'microtile_rows': {role: int(config.io_views[name].microtile.outer) for role, name in roles.items()},
        }

    def build_ports(self, node: OpNode, config: AttentionConfig) -> PortMap:
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        q, k, v = (input_tensor_for_role(node, role) for role in ('q', 'k', 'v'))
        return PortMap(
            inputs={
                q.name: PortBinding('in1', n, endpoints=tuple((f'kk[{i * c}].in[0]',) for i in range(n))),
                k.name: PortBinding('in2', 1, endpoints=(tuple(f'kk[{i * c}].in[1]' for i in range(n)),)),
                v.name: PortBinding(
                    'in3', 1, endpoints=(tuple(f'kk[{i * c + c - 1}].in[{2 if c == 1 else 1}]' for i in range(n)),)
                ),
            },
            outputs={
                node.outputs[0].name: PortBinding(
                    'out1', n, endpoints=tuple((f'kk[{i * c + c - 1}].out[0]',) for i in range(n))
                )
            },
        )

    def describe_input_staging(self, _node, config, tensor_name, port):
        """A row band at the stored width, laid out as a Dense writes one (Q a chain's band, K and V one of every
        row), so a Dense hands it over directly."""
        return describe_outer_lhs_staging(config.io_views[tensor_name], ROW_BANDS, port)

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_outer_output_staging(config.io_views[tensor_name], port)

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return 'outer'

    def buffer_locations(self, _node: OpNode, config: AttentionConfig, anchor_row: int):
        """A chain's queries where its first tile reads its producer's (as a row-wise op's input), every key in that
        tile's own memory and every value in its last tile's, its output after its last tile (as a row-wise op's)."""
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        locations = []
        for chain in range(n):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + chain, c)
            cols = [c - 1 - p if flow.reversed else p for p in range(c)]
            locations.append(
                BufferLocation('in1', chain, cols[0] + flow.input_col, chain, PING_PONG_BANKS, reader=(cols[0], chain))
            )
            locations.append(BufferLocation('in2', 0, cols[0], chain, PING_PONG_BANKS, reader=(cols[0], chain)))
            locations.append(BufferLocation('in3', 0, cols[-1], chain, PING_PONG_BANKS, reader=(cols[-1], chain)))
            locations.append(BufferLocation('out1', chain, cols[-1] + flow.output_col, chain, PING_PONG_BANKS))
        return tuple(locations)

    def work(self, node: OpNode, config: AttentionConfig) -> int:
        return ranked_cycles(attention_cycles(node, config), config) * config.int8_macs_per_cycle


def chains(node: OpNode, device, directives) -> Tuple[int, int]:
    """(cas_num, cas_length) as directed: query chains of one tile or two, a whole number of 8 queries each."""
    par = dict(directives.get('parallelism') or {})
    cas_num, cas_length = int(par.get('cas_num', 1)), int(par.get('cas_length', 1))
    tokens = int(node.outputs[0].shape[0])
    if cas_length not in (1, 2):
        raise ConfigRefused(f'{node.name}: a chain is one tile or two, not {cas_length}.')
    if (tokens // 8) % cas_num:
        raise ConfigRefused(f'{node.name}: {tokens} tokens split into whole 8s, not {cas_num}.')
    if cas_num > int(device.rows) - int(device.row_start):
        raise ConfigRefused(f'{node.name}: {cas_num} query chains need a row each, the array has {device.rows}.')
    return cas_num, cas_length


def check_bank(node: OpNode, device, ping: int) -> None:
    if ping > int(device.bank_mem_bytes):
        raise ConfigRefused(f'{node.name}: {ping} B of buffers in a bank of {int(device.bank_mem_bytes)}.')


def stack_bytes(device, cas_length: int) -> int:
    """The stack bank holds the stack and one copy of each table: the spilled scores and probabilities grow with the
    keys and head dimension (4.6 KB measured at 64 keys, head dimension 32)."""
    return int(device.bank_mem_bytes) - (2 if cas_length == 1 else 1) * TABLE_BYTES - 1024


def softmax_arithmetic(node: OpNode, q_frac: int, k_frac: int, v_frac: int, out_frac: int) -> Dict[str, Any]:
    """The logits' format and shifts, the exp table and the output's shift, from Q's, K's, V's and O's fixed points."""
    meta = node.metadata
    products = q_frac + k_frac
    logits16 = int(meta['logit_bits']) == 32
    if logits16:
        # exact products, no logit format given: int16 logits as fine as the exp table can span (its last entry
        # zero), the score scale and the temperature both in its step
        scale = float(meta['score_scale']) * float(meta['temperature'])
        logit_frac = min(products, math.floor(math.log2(scale * 255 / EXP_SPAN)))
        score_shift = products - logit_frac
        step = 2.0**-logit_frac * scale
    else:
        logit_frac = int(meta['logit_frac'])
        score_shift = products - logit_frac + shift_of(float(meta['score_scale']), 'logit prescale', node)
        step = 2.0**-logit_frac * float(meta['temperature'])
    # O = (theta / 2^7) 2^e y / 2^out_shift, y ~ 2^28 / (sum 2^e): theta / sum is in v's format
    out_shift = RECIP_SHIFT - THETA_SHIFT - (out_frac - v_frac)
    out_shift += shift_of(float(meta['output_scale']), 'output scale', node)
    if score_shift < 0 or out_shift < 0:
        raise ValueError(f'{node.name}: logits or output finer than their products ({score_shift}, {out_shift}).')
    return {
        'logits16': logits16,
        'score_shift': score_shift,
        'out_shift': out_shift,
        'exp_table': tuple(round(255 * math.exp(-d * step)) for d in range(256)),
    }


def attention_params(node: OpNode, config: AttentionConfig) -> Dict[str, Any]:
    tokens = int(node.outputs[0].shape[0])
    par = config.parallelism
    return {
        'parallelism': par,
        'alternating_horizontal': config.alternating_horizontal,
        'columns': config.columns,
        'query_blocks': tokens // par.cas_num // 8,
        'key_blocks': tokens // 8,
        'logits16': config.logits16,
        'score_shift': config.score_shift,
        'out_shift': config.out_shift,
        'stack_bytes': config.stack_bytes,
        'exp_table': lut_layout(config.exp_table, config.table_bank_bits, uint8=True),
        'recip_table': lut_layout(RECIP_TABLE, config.table_bank_bits),
    }


# The estimate charges a chain a cascade hand-over per further tile (`estimate.CASCADE_HOP_CYCLES`); a chain of two
# here works on successive frames, so the hand-over is latency, not interval: `work` leaves it out.
CHAIN_HANDOVER_CYCLES = 30


def ranked_cycles(cycles: float, config) -> int:
    """A chain's slower tile's cycles as `work` reports them, the hand-over of a chain of two left out."""
    return round(cycles - (CHAIN_HANDOVER_CYCLES if config.parallelism.cas_length == 2 else 0))


def attention_cycles(node: OpNode, config: AttentionConfig) -> int:
    """Cycles of a chain's slower tile per call: per query block, fitted to the kernels' AIE-ML profiles at 32 and 64
    tokens, head dimensions 16 to 64 (AIE-MLv2 runs them faster)."""
    tokens = int(node.outputs[0].shape[0])
    par = config.parallelism
    qb, kb, nj = tokens // par.cas_num // 8, tokens // 8, config.columns // 8
    if par.cas_length == 1:
        block = 38 * kb + 12.3 * kb * nj + 21
    else:  # the probabilities' tile or the values'
        block = max(48 * kb + 6 * kb * nj - 50, 14 * kb * nj + 4 * nj + 10)
    return round(25 + qb * block)
