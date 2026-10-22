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
    FuseOps,
    LegalizeFanoutEntries,
    LegalizeMemtilePortLimits,
    MaterializeMemoryPlan,
    PackKernelArtifacts,
    PlaceKernels,
    Resolve,
    VerifyPhysicalPlan,
)

DEFAULT_PIPELINE = (
    ForceFloatMode,
    FoldApplyAlpha,
    FoldBias,
    FoldScale,
    FuseOps,
    FoldViewOps,
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
