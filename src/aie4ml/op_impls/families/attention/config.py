from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ...utils import ParallelismConfig, TensorView


@dataclass(frozen=True)
class AttentionConfig:
    precision: Dict[str, Any]
    #: cas_num chains split the queries; a chain is one tile, or (cas_length 2) a tile for the probabilities and one
    #: for the values and the division
    #: ('outer': a chain's output is whole)
    parallelism: ParallelismConfig
    io_views: Dict[str, TensorView]
    alternating_horizontal: bool
    #: the head dimension as its buffers hold it: a Dense producer's columns, zeros past the head dimension
    columns: int
    #: int16 logits (exact products: the op chose their format), else int8 (the graph's)
    logits16: bool
    #: the logits: round(q.k / 2^score_shift)
    score_shift: int
    #: the output: theta / sum through the division's int16 path, then round(. / 2^out_shift)
    out_shift: int
    #: round(255 exp(-gap * step)) for logit gaps 0..255, a step of the logits in the Softmax's units: the
    #: unnormalised probabilities
    exp_table: Tuple[int, ...]
    #: the width its lookup tables repeat each chunk at (`DeviceSpec.lookup_bank_bits`)
    table_bank_bits: int
    #: each tile's stack: what its stack bank leaves
    stack_bytes: int
    #: the core's peak int8 multiply-accumulates a cycle: the unit of `work`
    int8_macs_per_cycle: int
    #: the logits' additive bias, 8 or 16 bits (0: none), which starts their accumulator shifted to the products'
    #: fixed point by bias_shift
    bias_bits: int
    bias_shift: int


@dataclass(frozen=True)
class AttentionHeadConfig(AttentionConfig):
    #: x's columns as its two halves hold them (zeros past the model dimension, whose weight rows are zero too)
    model_dim: int
    #: O's columns as its buffer holds them: the head dimension, then zeros to what the Dense after it reads
    out_columns: int
    #: the key blocks a chain's first tile looks the probabilities up for: half of them in a pair, the rest its last
    looked_blocks: int
    #: each projection's accumulator to int8, as its Dense's: (q, k, v)
    projection_shifts: Tuple[int, int, int]
