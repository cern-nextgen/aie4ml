from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ...utils import MicrotileShape, ParallelismConfig, TensorView


@dataclass(frozen=True)
class AddFlags:
    transpose_lhs: bool
    transpose_rhs: bool


@dataclass(frozen=True)
class AddConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    vec_size: int
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    shift: int
    accumulator_tag: Optional[str]
    rounding_mode: Optional[str]
    alternating_horizontal: bool
    #: The producer stagings the add lays every tensor out in, port for port; None for its own layout.
    adopted_staging: Optional[Tuple[Dict[str, Any], ...]] = None
    flags: AddFlags = AddFlags(transpose_lhs=False, transpose_rhs=False)
    microtile: Optional[MicrotileShape] = None
