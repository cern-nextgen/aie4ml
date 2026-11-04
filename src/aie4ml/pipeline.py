"""Core AIE compiler pipeline definition."""

from .passes import (
    ChooseParallelism,
    ClassifyTransportEntries,
    CollectMemoryEntries,
    CompactBufferRank,
    FoldApplyAlpha,
    FoldBias,
    FoldScale,
    FoldViewOps,
    ForceFloatMode,
    FuseAttention,
    FuseOps,
    LegalizeFanoutEntries,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
    PackKernelArtifacts,
    PlaceKernels,
    Resolve,
    VerifyPhysicalPlan,
    absorb_projections,
)

DEFAULT_PIPELINE = (
    ForceFloatMode,
    FoldApplyAlpha,
    FoldBias,
    FoldScale,
    FuseOps,
    FoldViewOps,
    FuseAttention,
    ChooseParallelism,  # AIEConfig Optimize
    Resolve,
    PackKernelArtifacts,
    CollectMemoryEntries,
    LegalizeFanoutEntries,
    ClassifyTransportEntries,
    PlaceKernels,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
    CompactBufferRank,
    VerifyPhysicalPlan,
)

# Rewrites the design search weighs (ChooseParallelism): each graph is searched with and without it, and the better
# ranking design kept.
SEARCHED_REWRITES = (absorb_projections,)
