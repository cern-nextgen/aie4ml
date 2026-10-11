from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...utils import MicrotileShape, ParallelismConfig, TensorView


@dataclass(frozen=True)
class SoftmaxConfig:
    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    param_sets: int
    vec_size: int
    accumulator_tag: Optional[str]
    inv_shift: int
    use_clb: bool
    alternating_horizontal: bool
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    hccs: Dict[str, Any]
    layout: str = 'linear'
    microtile: Optional[MicrotileShape] = None
    transpose_input: bool = False
    #: 'hccs' (calibrated clipped-linear surrogate) or 'exp' (accurate exp, no calibration).
    approximation: str = 'hccs'
    #: For the exp variant: log2(e) * input_scale in Q(EXP_ZF). 0 for HCCS.
    exp_kq: int = 0
    #: For the exp variant: log2(e) * input_scale. 0 for HCCS.
    exp_k: float = 0.0
