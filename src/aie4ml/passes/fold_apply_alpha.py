# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Fold an hls4ml ApplyAlpha, a per-channel scale and bias, into the Dense before it."""

from dataclasses import replace

import numpy as np

from ..aie_types import FloatIntent
from ..ir import get_backend_context
from ..ir.graph import has_input_role, input_tensor_for_role
from ..op_impls.family_registry import fuses
from .base import AIEPass


class FoldApplyAlpha(AIEPass):
    """Fold a per-channel `y = x * scale + bias` (an hls4ml ApplyAlpha) into the Dense before it: its weights take
    the scale, its bias the scale and the bias.

    Integer weights keep their codes on a grid the scale moves, which is exact only for one power of two every
    channel shares; another scale is refused rather than rounded. Float weights take any scale.
    """

    def __init__(self):
        self.name = 'fold_apply_alpha'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for node in list(graph.nodes):
            if node.op_type != 'applyalpha':
                continue
            dense = node.inputs[0].producer
            if not fuses(dense, 'apply_alpha'):
                raise NotImplementedError(
                    f'{node.name}: a per-channel scale folds into the Dense before it, not into '
                    f'{dense.op_type if dense is not None else "the graph input"}.'
                )
            scale = np.asarray(node.metadata['scale'], dtype=np.float64)
            alpha_bias = np.asarray(node.metadata['bias'], dtype=np.float64)

            weight = input_tensor_for_role(dense, 'rhs')
            if not isinstance(weight.precision, FloatIntent):
                shift = -np.log2(scale[0])
                if np.any(scale != scale[0]) or shift != np.round(shift):
                    raise NotImplementedError(
                        f'{node.name}: folds into {dense.name} only one power-of-two scale every channel shares, '
                        f'got {sorted(set(scale.tolist()))}; quantize its weights with alpha=1 or one po2 alpha.'
                    )
                weight.precision = replace(weight.precision, frac=weight.precision.frac + int(shift))
            weight.data = np.asarray(weight.data, dtype=np.float64) * scale

            if has_input_role(dense, 'bias'):
                bias = input_tensor_for_role(dense, 'bias')
                bias.data = np.asarray(bias.data, dtype=np.float64) * scale + alpha_bias
            elif np.any(alpha_bias):
                raise NotImplementedError(f'{node.name}: {dense.name} has no bias to take its bias.')

            graph.remove_node(node, mode='contract')
            changed = True

        return changed
