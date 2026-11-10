from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from ....errors import ConfigRefused
from ....ir.graph import OpNode, has_input_role, input_tensor_for_role
from ...base import PING_PONG_BANKS, BufferLocation, row_flow
from ...utils import MicrotileShape, ParallelismConfig, build_tensor_view
from ...utils.precision import is_int, resolve_exact_storage_dtype, scale_shift
from ...utils.tensor_view import microtile_from_staging

#: The row sums of the uint8 exp (255 at a row's maximum) stay in int16 up to 128 keys.
MAX_TOKENS = 128
#: The roles the kernels read as int8: Q, K and V, or x and the projections' weights.
INT8_ROLES = ('q', 'k', 'v', 'x', 'wq', 'wk', 'wv')
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
#: Cycles a key block's bias adds to its scores (its tile's transpose and shift into the accumulator), by bias width:
#: AIE-ML profiles at 64 tokens.
BIAS_BLOCK_CYCLES = {8: 0.5, 16: 3}
#: What a stack bank keeps besides the stack and the tables: the kernel object and ADF's system data.
KERNEL_BYTES = 1024
#: The stack a tile spills, per head-dimension column: bytes per key and per model-dimension column (the projections'
#: weights), a lone tile's and a chain's busiest, by generation: upper bounds measured up to 128 keys and head and
#: model dimensions 16-128.
SPILL_BYTES = {
    'AIE-ML': {'head': ((3 / 8, 13 / 8), (0, 3 / 2)), 'plain': ((1 / 2, 0), (3 / 8, 0))},
    'AIE-MLV2': {'head': ((17 / 8, 21 / 8), (5 / 4, 5 / 4)), 'plain': ((11 / 8, 0), (9 / 8, 0))},
}


def tokens_of(node: OpNode) -> int:
    return int(node.outputs[0].shape[0])


def attends(node: OpNode, device) -> bool:
    """Whether the attention variants take the node on the device, a bias's shift aside: int8 operands, weights and
    output, an int8 or int16 bias streamed in, and parallel lookups for their tables."""
    if device.lookup_bank_bits is None or tokens_of(node) > MAX_TOKENS:
        return False
    eights = [node.outputs[0], *(input_tensor_for_role(node, r) for r in INT8_ROLES if has_input_role(node, r))]
    if not all(is_int(t.precision, 8, True) for t in eights):
        return False
    if not has_input_role(node, 'bias'):
        return True
    u = input_tensor_for_role(node, 'bias')
    return not u.is_parameter and (is_int(u.precision, 8, True) or is_int(u.precision, 16, True))


def shift_of(scale: float, what: str, node: OpNode) -> int:
    """The right-shift applying a power-of-two scale of at most 1."""
    shift = scale_shift(scale)
    if shift is None:
        raise ValueError(f'{node.name}: the fused attention takes a power-of-two {what} of at most 1, got {scale}.')
    return shift


def bias_shift(node: OpNode, products: int) -> Optional[int]:
    """The shift taking the logits' bias to the products' fixed point (`products` fraction bits), where it starts
    their accumulator; None where a score scale that is no power of two up to 1, or a finer bias, makes it inexact."""
    prescale = scale_shift(float(node.metadata['score_scale']))
    if prescale is None:
        return None
    shift = products + prescale - int(input_tensor_for_role(node, 'bias').precision.frac)
    return shift if shift >= 0 else None


def bias_view(node: OpNode, cas_num: int):
    """The logits' bias as a chain's first tile reads it: its queries' rows, in 8x8 tiles."""
    u, tokens = input_tensor_for_role(node, 'bias'), tokens_of(node)
    return build_tensor_view(
        node, u, 'inputs', full_inner=tokens, tile_inner=tokens, tile_inner_raw=tokens, full_outer=tokens,
        tile_outer=tokens // cas_num, tile_outer_raw=tokens // cas_num, microtile=MICROTILE,
    )  # fmt: skip


def kernel_microtile(microtile: Optional[MicrotileShape]) -> MicrotileShape:
    """`microtile` where the kernels read and write it (8 wide, 4 or 8 rows), else 8x8."""
    ok = microtile is not None and microtile.inner == 8 and microtile.outer in (4, 8)
    return microtile if ok else MICROTILE


def stored_microtile(contract) -> MicrotileShape:
    """The microtile a producer stores the tensor in, where the kernel reads it (`kernel_microtile`)."""
    stored = contract is not None and contract.port_staging
    return kernel_microtile(microtile_from_staging(contract.port_staging[0]) if stored else None)


def storage_precision(node: OpNode, roles) -> Dict[str, Any]:
    """The storage dtype of each (role, tensor), and of the bias where the node adds one."""
    roles = [*roles, *([('bias', input_tensor_for_role(node, 'bias'))] if has_input_role(node, 'bias') else [])]
    return {role: resolve_exact_storage_dtype(t.precision, namespace=role, layer_name=node.name) for role, t in roles}


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


def chain_tiles(config, anchor_row: int):
    """Each chain's (index, `row_flow`, its tiles' columns in chain order)."""
    n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
    for chain in range(n):
        flow = row_flow(config.alternating_horizontal, int(anchor_row) + chain, c)
        yield chain, flow, [c - 1 - p if flow.reversed else p for p in range(c)]


def bias_locations(config, anchor_row: int, group: str):
    """Each chain's rows of the bias, port group `group`, where its first tile reads its producer's (as a row-wise op's
    input): its own input DMA channels may all be taken (keys and values, or x's halves)."""
    return [
        BufferLocation(group, chain, cols[0] + flow.input_col, chain, PING_PONG_BANKS, reader=(cols[0], chain))
        for chain, flow, cols in chain_tiles(config, anchor_row)
    ]


def chains(node: OpNode, device, directives) -> Tuple[int, int]:
    """(cas_num, cas_length) as directed: query chains of one tile or two, a whole number of 8 queries each."""
    par = dict(directives.get('parallelism') or {})
    cas_num, cas_length = int(par.get('cas_num', 1)), int(par.get('cas_length', 1))
    tokens = tokens_of(node)
    if cas_length not in (1, 2):
        raise ConfigRefused(f'{node.name}: a chain is one tile or two, not {cas_length}.')
    if (tokens // 8) % cas_num:
        raise ConfigRefused(f'{node.name}: {tokens} tokens split into whole 8s, not {cas_num}.')
    if cas_num > int(device.rows) - int(device.row_start):
        raise ConfigRefused(f'{node.name}: {cas_num} query chains need a row each, the array has {device.rows}.')
    return cas_num, cas_length


def check_bank(node: OpNode, device, size: int, what: str) -> None:
    if size > int(device.bank_mem_bytes):
        raise ConfigRefused(f'{node.name}: {size} B of {what} in a bank of {int(device.bank_mem_bytes)}.')


def stack_bytes(node: OpNode, device, tables: int, spill: int, scratch: int = 0) -> int:
    """The stack of a tile whose stack bank holds one copy of `tables` tables and the kernel, whose token scratch
    takes `scratch` bytes, where it holds the `spill` bytes the kernel spills."""
    stack = int(device.bank_mem_bytes) - tables * TABLE_BYTES - KERNEL_BYTES - scratch
    if stack < spill:
        raise ConfigRefused(
            f'{node.name}: {stack} B of stack beside {scratch} B of token scratch, under the {spill} B it spills.'
        )
    return stack


def spill_bytes(device, kernel: str, cas_length: int, keys: int, head_dim: int, model_dim: int = 0) -> int:
    per_key, per_model = SPILL_BYTES[device.generation][kernel][cas_length - 1]
    return math.ceil((per_key * keys + per_model * model_dim) * head_dim)


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
    bias = input_tensor_for_role(node, 'bias') if has_input_role(node, 'bias') else None
    return {
        'logits16': logits16,
        'score_shift': score_shift,
        'out_shift': out_shift,
        'exp_table': tuple(round(255 * math.exp(-d * step)) for d in range(256)),
        'bias_bits': 0 if bias is None else int(bias.precision.width),
        'bias_shift': 0 if bias is None else bias_shift(node, products),
    }


def attention_params(node: OpNode, config) -> Dict[str, Any]:
    tokens = tokens_of(node)
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
        'bias_bits': config.bias_bits,
        'bias_type': config.precision['bias'].c_type if config.bias_bits else None,
        'bias_shift': config.bias_shift,
    }
