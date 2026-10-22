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

HLS4ML_FLOW_SPEC = (
    ('force_float', ForceFloatMode),
    ('fold_apply_alpha', FoldApplyAlpha),
    ('fold_bias', FoldBias),
    ('fold_scale', FoldScale),
    ('fuse', FuseOps),
    ('fold_views', FoldViewOps),
    ('choose_parallelism', ChooseParallelism),  # AIEConfig Optimize
    ('resolve', Resolve),
    ('pack', PackKernelArtifacts),
    ('memory_collect', CollectMemoryEntries),
    ('fanout_legalize', LegalizeFanoutEntries),
    ('transport_classify', ClassifyTransportEntries),
    ('placement', PlaceKernels),
    ('memtile_legalize', LegalizeMemtilePortLimits),
    ('memory_plan', MaterializeMemoryPlan),
    ('compact_batch', CompactBufferRank),
    ('verify_physical', VerifyPhysicalPlan),
)

DEFAULT_PIPELINE = tuple(pass_cls for _, pass_cls in HLS4ML_FLOW_SPEC)
