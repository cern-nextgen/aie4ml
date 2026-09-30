# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Elementwise ops: Mul/Div (scalar scale) and Add."""

from __future__ import annotations

import numpy as np

from ....aie_types import FloatIntent
from ..context import OnnxImportContext
from ..registry import onnx_handler


@onnx_handler('Mul', 'Div')
def _mul_div(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    op_type = node.op_type
    if len(node.input) != 2:
        raise ValueError(f'{node_name}: {op_type} must have exactly 2 inputs.')

    if op_type == 'Mul':
        constant_names = [name for name in node.input if name in ctx.initializers]
        if len(constant_names) != 1:
            raise NotImplementedError(f'{node_name}: Mul currently requires exactly one constant input.')
        constant_name = constant_names[0]
        source_name = node.input[1] if node.input[0] == constant_name else node.input[0]
        scale = float(np.asarray(ctx.initializers[constant_name]).reshape(-1)[0])
    else:
        source_name, constant_name = node.input
        if constant_name not in ctx.initializers:
            raise NotImplementedError(f'{node_name}: Div currently requires a constant divisor.')
        divisor = float(np.asarray(ctx.initializers[constant_name]).reshape(-1)[0])
        if divisor == 0.0:
            raise ValueError(f'{node_name}: Div constant divisor must be nonzero.')
        scale = 1.0 / divisor

    if np.asarray(ctx.initializers[constant_name]).size != 1:
        raise NotImplementedError(f'{node_name}: {op_type} currently requires a scalar constant.')

    source = ctx.source_for(source_name, node_name)
    out_name = node.output[0]
    ctx.propagate_order(source_name, out_name)  # a scalar scale is order-agnostic
    ctx.emit(
        'scale',
        node_name,
        inputs=[source],
        outputs=[(out_name, ctx.canonical_shape(out_name, node_name), source.precision)],
        roles=['lhs'],
        metadata={'scale': scale, 'layer_class': op_type, 'source_class': op_type, 'source_layer': node_name},
        directives=directives,
    )


@onnx_handler('Add')
def _add(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) != 2:
        raise ValueError(f'{node_name}: Add must have exactly 2 inputs.')
    lhs_name, rhs_name = node.input
    lhs = ctx.any_source_for(lhs_name, node_name)
    rhs = ctx.any_source_for(rhs_name, node_name)
    order = ctx.common_order([lhs_name, rhs_name], node_name)  # elementwise: any shared order
    out_name = node.output[0]
    out_precision = lhs.precision if isinstance(lhs.precision, FloatIntent) else None

    # A constant addend is a bias candidate (FoldBias re-fuses a dense bias). Two
    # activations must be exact-shape: the elementwise add kernel does not broadcast.
    if lhs.is_parameter == rhs.is_parameter:
        lhs_shape = ctx.output_shape(lhs_name, node_name)
        rhs_shape = ctx.output_shape(rhs_name, node_name)
        if lhs_shape != rhs_shape:
            raise ValueError(
                f'{node_name}: generic Add only supports exact-shape elementwise inputs; '
                f'got {lhs_shape} and {rhs_shape}.'
            )
    ctx.set_order(out_name, order, node_name)  # the sum is the canonical tensor its inputs view alike
    ctx.emit(
        'add',
        node_name,
        inputs=[lhs, rhs],
        outputs=[(out_name, ctx.canonical_shape(out_name, node_name), out_precision)],
        roles=['lhs', 'rhs'],
        metadata={'layer_class': 'Add', 'source_class': 'Add', 'source_layer': node_name},
        directives=directives,
    )
