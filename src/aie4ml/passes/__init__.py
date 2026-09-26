"""Optimizer passes specific to the AIE backend."""

from .base import AIEPass, run_aie_passes
from .compact_buffer_rank import CompactBufferRank
from .fold_apply_alpha import FoldApplyAlpha
from .fold_bias import FoldBias
from .fold_scale import FoldScale
from .fold_views import FoldViewOps
from .force_float_mode import ForceFloatMode
from .fuse_activation import FuseActivationCasts
from .fuse_pool import FusePool
from .legalize_layouts import LegalizeLayouts
from .pack import PackKernelArtifacts
from .placement import PlaceKernels
from .resolve import Resolve
from .transport import (
    BuildMemoryPlan,
    ClassifyTransportEntries,
    CollectMemoryEntries,
    LegalizeFanoutEntries,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
)
from .verify_physical import VerifyPhysicalPlan

__all__ = [
    'AIEPass',
    'run_aie_passes',
    'FuseActivationCasts',
    'FusePool',
    'FoldApplyAlpha',
    'FoldBias',
    'FoldScale',
    'ForceFloatMode',
    'FoldViewOps',
    'CompactBufferRank',
    'LegalizeFanoutEntries',
    'LegalizeLayouts',
    'LegalizeMemtilePortLimits',
    'Resolve',
    'VerifyPhysicalPlan',
    'PackKernelArtifacts',
    'PlaceKernels',
    'CollectMemoryEntries',
    'ClassifyTransportEntries',
    'MaterializeMemoryPlan',
    'BuildMemoryPlan',
]
