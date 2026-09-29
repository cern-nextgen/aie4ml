from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...utils import ParallelismConfig, TensorView


@dataclass(frozen=True)
class MatmulMicrotileConfig:
    """aie::mmul intrinsic microtile dimensions.

    This is the smallest granularity: data must be rearranged to match these
    shapes before the aie::mmul intrinsic is called.
    """

    microtile_m: int
    microtile_k: int
    microtile_n: int


@dataclass(frozen=True)
class DenseFlags:
    use_relu: bool
    transpose_lhs: bool
    use_bias: bool


@dataclass(frozen=True)
class MatmulFlags:
    transpose_lhs: bool
    transpose_rhs: bool


@dataclass(frozen=True)
class DenseConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    bank_mem_bytes: int
    alternating_horizontal: bool
    flags: DenseFlags
    lhs_inner_shards: Optional[Tuple[int, int]] = None  # the producer's shard-by-shard K order, adopted


@dataclass(frozen=True)
class MatmulConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    microtiling: MatmulMicrotileConfig
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    alternating_horizontal: bool
    flags: MatmulFlags
