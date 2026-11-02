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
    #: the int8 logits: round(q.k / 2^score_shift)
    score_shift: int
    #: the output: theta / sum through the division's int16 path, then round(. / 2^out_shift)
    out_shift: int
    #: round(255 exp(-gap * 2^-logit_frac * temperature)) for logit gaps 0..255: the unnormalised probabilities
    exp_table: Tuple[int, ...]
    #: the data-memory bank width its lookup tables repeat each chunk at
    table_bank_bits: int
    #: each tile's stack: what its stack bank leaves
    stack_bytes: int
    #: the core's peak int8 multiply-accumulates a cycle: the unit of `work`
    int8_macs_per_cycle: int
