from __future__ import annotations

from pathlib import Path

from ..ir.attributes import OUTPUT_VIEWS
from .base import OpImplVariant
from .family_registry import FUSIONS

_TEMPLATE_ROOT = Path(__file__).resolve().parent.parent / 'templates' / 'firmware' / 'variants'


class OpImplRegistry:
    def __init__(self):
        self._variants: dict[str, list[OpImplVariant]] = {}

    def register(self, variant: OpImplVariant) -> None:
        if any(v.variant_id == variant.variant_id for vs in self._variants.values() for v in vs):
            raise ValueError(f'variant {variant.variant_id!r} is already registered.')
        self._variants.setdefault(variant.op_type, []).append(variant)

    def candidates(self, op_type: str) -> list[OpImplVariant]:
        """Return registered variants for op_type sorted by descending plevel."""
        return sorted(self._variants.get(op_type, []), key=lambda v: v.plevel, reverse=True)


_GLOBAL_OP_IMPL_REGISTRY = OpImplRegistry()


def get_op_impl_registry() -> OpImplRegistry:
    return _GLOBAL_OP_IMPL_REGISTRY


def register_variant(cls):
    missing = [hook for hook in ('matches', 'resolve') if getattr(cls, hook) is getattr(OpImplVariant, hook)]
    if missing:
        raise TypeError(f'{cls.__name__}: registered for resolution to select, it must implement {missing}.')
    unknown = sorted((cls.supported_fusions - FUSIONS) | (cls.supported_output_views - OUTPUT_VIEWS))
    if unknown:
        raise TypeError(f'{cls.__name__}: declares {unknown}, which no pass fuses; see FUSIONS and OUTPUT_VIEWS.')
    if cls.param_template:
        expected = _TEMPLATE_ROOT / cls.param_template / 'parameters.h.jinja'
        if not expected.exists():
            raise FileNotFoundError(
                f'{cls.__name__}: param_template={cls.param_template!r} has no template at {expected}. '
                'Create the template before registering the variant.'
            )
    _GLOBAL_OP_IMPL_REGISTRY.register(cls())
    return cls
