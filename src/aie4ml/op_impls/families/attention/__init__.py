"""Attention implementation family: one head's softmax(Q Kᵀ) V fused in one kernel graph, its projections too where
they fuse."""

from . import resolver  # noqa: F401
from .attention import AttentionOpImplVariant
from .config import AttentionConfig, AttentionHeadConfig
from .head import AttentionHeadOpImplVariant

__all__ = ['AttentionConfig', 'AttentionHeadConfig', 'AttentionHeadOpImplVariant', 'AttentionOpImplVariant']
