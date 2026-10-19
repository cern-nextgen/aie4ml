# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Fold per-channel scale nodes into their predecessor Dense weight precision."""

from dataclasses import replace

import numpy as np

from ..aie_types import FloatIntent
from ..ir import get_backend_context
from ..ir.graph import has_input_role, input_tensor_for_role
from ..op_impls.family_registry import fuses
from .base import AIEPass


class FoldApplyAlpha(AIEPass):
    """Fold output scale into a preceding Dense when legal.

    For applyalpha nodes with constant per-channel scale, this pass updates the
    producer Dense quant intent and bias, then removes the scale node.

    This is mainly a graph simplification/optimization pass. It is only bit-exact
    when the incoming weight data already matches the frontend quantization grid,
    or that grid has been reconstructed exactly before folding.
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

            scale = np.asarray(node.metadata['scale'], dtype=np.float64)
            p_min = int(np.round(-np.log2(scale.max())))

            dense_node = node.inputs[0].producer
            if not fuses(dense_node, 'apply_alpha'):
                raise NotImplementedError(
                    f'{node.name}: a per-channel scale folds into the Dense before it, not into '
                    f'{dense_node.op_type if dense_node is not None else "the graph input"}.'
                )

            weight_tv = input_tensor_for_role(dense_node, 'rhs')
            old_w = weight_tv.precision
            if isinstance(old_w, FloatIntent):
                continue
            weight_tv.precision = replace(old_w, frac=old_w.frac + p_min)

            if has_input_role(dense_node, 'bias'):
                bias_tv = input_tensor_for_role(dense_node, 'bias')
                bias_tv.data = np.asarray(bias_tv.data, dtype=np.float64) * scale

            graph.remove_node(node, mode='contract')
            changed = True

        return changed
