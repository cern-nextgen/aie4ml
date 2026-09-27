# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Normalization ops: LayerNormalization and (HCCS) Softmax; BatchNormalization is refused."""

from __future__ import annotations

import numpy as np

from ....aie_types import FloatIntent
from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..shapes import normalize_axis
from ..utils import attr


@onnx_handler('BatchNormalization')
def _batch_norm(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    # Its per-channel scale folds exactly only into float weights, before quantization.
    raise NotImplementedError(
        f'{node_name}: fold BatchNormalization into the layer before it ahead of quantizing (a PyTorch eval-mode '
        'export and onnxruntime quant_pre_process fold Conv + BatchNormalization).'
    )


@onnx_handler('LayerNormalization')
def _layer_norm(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) not in (2, 3):
        raise ValueError(f'{node_name}: LayerNormalization must have 2 or 3 inputs.')
    x_name, scale_name = node.input[0], node.input[1]
    bias_name = node.input[2] if len(node.input) == 3 else None

    x_shape = ctx.output_shape(x_name, node_name)
    axis = normalize_axis(int(attr(node, 'axis', -1)), len(x_shape), node_name, 'LayerNormalization')
    norm_shape = tuple(x_shape[axis:])
    epsilon = float(attr(node, 'epsilon', 1e-5))

    x_tensor = ctx.canonical_source(x_name, node_name)
    scale_tensor = ctx.parameter_source_for(scale_name, node_name)
    if not scale_tensor.is_parameter:
        raise ValueError(f'{node_name}: LayerNormalization Scale must be a constant initializer.')

    if bias_name is not None:
        bias_tensor = ctx.parameter_source_for(bias_name, node_name)
        if not bias_tensor.is_parameter:
            raise ValueError(f'{node_name}: LayerNormalization Bias must be a constant initializer.')
    else:
        zeros = np.zeros(norm_shape, dtype=np.float64)
        bias_tensor = ctx.param_tensor(f'{node_name}_beta_zero', zeros, scale_tensor.precision)

    out_precision = x_tensor.precision if isinstance(x_tensor.precision, FloatIntent) else None
    ctx.emit(
        'layer_norm',
        node_name,
        inputs=[x_tensor, scale_tensor, bias_tensor],
        outputs=[(node.output[0], ctx.output_shape(node.output[0], node_name), out_precision)],
        roles=['lhs', 'gamma', 'beta'],
        metadata={
            'layer_class': 'LayerNormalization',
            'source_class': 'LayerNormalization',
            'source_layer': node_name,
            'epsilon': epsilon,
            'axis': axis,
        },
        directives=directives,
    )


@onnx_handler('Softmax')
def _softmax(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) != 1:
        raise ValueError(f'{node_name}: Softmax must have exactly 1 input.')
    src = ctx.canonical_source(node.input[0], node_name)
    in_shape = ctx.output_shape(node.input[0], node_name)
    axis = normalize_axis(int(attr(node, 'axis', -1)), len(in_shape), node_name, 'Softmax')
    out_name = node.output[0]
    ctx.emit(
        'softmax',
        node_name,
        inputs=[src],
        outputs=[(out_name, ctx.output_shape(out_name, node_name), None)],
        roles=['lhs'],
        metadata={'axis': axis, 'layer_class': 'Softmax', 'source_class': 'Softmax', 'source_layer': node_name},
        directives=directives,
    )
