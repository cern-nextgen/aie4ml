"""Core AIE compiler pipeline definition."""

from .passes import (
    ClassifyTransportEntries,
    CollectMemoryEntries,
    CompactBufferRank,
    FoldApplyAlpha,
    FoldBias,
    FoldScale,
    FoldViewOps,
    ForceFloatMode,
    FuseActivationCasts,
    FusePool,
    LegalizeFanoutEntries,
    LegalizeLayouts,
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
    ('fuse', FuseActivationCasts),
    ('fuse_pool', FusePool),
    ('fuse_pooled', FuseActivationCasts),  # a ReLU after a fused pool: max and ReLU commute
    ('fold_views', FoldViewOps),
    ('resolve', Resolve),
    ('legalize_layouts', LegalizeLayouts),
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
