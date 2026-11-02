# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Constant folding at import: what an exporter computes from constants and static shapes becomes a constant."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .utils import attr, copied, require_onnx

KEPT = frozenset({'QuantizeLinear', 'DequantizeLinear'})  # their handlers read the formats they carry


def fold_constants(
    model_proto, initializers: Dict[str, np.ndarray], shapes: Callable[[object], Dict[str, Tuple[int, ...]]]
):
    """The model with each node of constant inputs, and each Shape of a static tensor, replaced by its outputs as
    constants (also added to `initializers`), until none is left. `shapes(model)` infers a model's shapes."""
    from onnx.reference import ReferenceEvaluator

    _, _, numpy_helper = require_onnx()
    model = copied(model_proto)
    while True:
        shape_of = shapes(model)
        kept, folded = [], []
        for node in model.graph.node:
            if node.op_type == 'Shape' and node.input[0] in shape_of:
                start, end = attr(node, 'start', 0), attr(node, 'end', None)
                outputs = [np.asarray(shape_of[node.input[0]], dtype=np.int64)[start:end]]
            elif node.op_type not in KEPT and all(not name or name in initializers for name in node.input):
                outputs = ReferenceEvaluator(node).run(None, {name: initializers[name] for name in node.input if name})
            else:
                kept.append(node)
                continue
            for name, value in zip(node.output, outputs):
                initializers[name] = np.asarray(value)
                folded.append(numpy_helper.from_array(initializers[name], name))
        if not folded:
            return model
        del model.graph.node[:]
        model.graph.node.extend(kept)
        model.graph.initializer.extend(folded)
