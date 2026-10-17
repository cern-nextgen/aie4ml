from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...utils import MicrotileShape, ParallelismConfig, TensorView


@dataclass(frozen=True)
class LayerNormConfig:
    """Resolved configuration for a fully-integer LayerNorm.

    cols must be a power of two and a multiple of vec_size;
    outer extent must be exactly partitionable across cas_num kernels.
    """

    precision: Dict[str, Any]
    parallelism: ParallelismConfig
    rows: int
    cols: int
    vec_size: int
    accumulator_tag: Optional[str]
    gamma_shift: int
    #: Fraction bits of gamma / sigma and of x - mu.
    scale_frac: int
    mean_frac: int
    out_shift: int
    eps_q0: int
    rounding_mode: Optional[str]
    alternating_horizontal: bool
    io_views: Dict[str, TensorView]
    io_route: Dict[str, Any]
    #: One of ir.graph.TENSOR_LAYOUTS; picks the kernel via the template's LAYOUT_TILED.
    layout: str = 'linear'
    #: Block a 'tiled' layout is stored in, inherited from the producer so the edge stays direct.
    microtile: Optional[MicrotileShape] = None
    transpose_input: bool = False
