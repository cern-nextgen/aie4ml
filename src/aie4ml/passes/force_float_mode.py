# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""ForceFloatMode: replace all IR tensor precisions with FloatIntent.

Testing pass that runs a quantized model through the float kernels.
Activated by setting AIEConfig.ComputeDtype to 'bfloat16' or 'float32'.
"""

from ..aie_types import FloatFormat, FloatIntent, QuantIntent
from ..ir import get_backend_context
from .base import AIEPass

_FORMAT_MAP = {
    'bfloat16': (FloatFormat.BF16, 16),
    'float32': (FloatFormat.FP32, 32),
    'float': (FloatFormat.FP32, 32),
}


class ForceFloatMode(AIEPass):
    """Replace every QuantIntent precision in the logical IR with FloatIntent."""

    def __init__(self):
        self.name = 'force_float_mode'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        fmt_str = (ctx.aie_config.get('ComputeDtype') or '').lower()
        if not fmt_str:
            return False

        if fmt_str not in _FORMAT_MAP:
            raise ValueError(f'ComputeDtype={fmt_str!r} is not recognised. ' f'Use one of: {list(_FORMAT_MAP)}')

        fmt, width = _FORMAT_MAP[fmt_str]
        float_intent = FloatIntent(width=width, format=fmt)

        changed = False
        for tv in ctx.ir.logical.tensors.values():
            if isinstance(tv.precision, QuantIntent):
                tv.precision = float_intent
                changed = True

        return changed
