# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""MatMul / Gemm lowering: static rank-2 RHS -> dense, else matmul."""

from __future__ import annotations

import numpy as np

from ....aie_types import FloatIntent
from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..utils import attr


@onnx_handler('MatMul', 'Gemm')
def _matmul_gemm(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    op_type = node.op_type
    if op_type == 'MatMul':
        if len(node.input) != 2:
            raise ValueError(f'{node_name}: MatMul must have exactly 2 inputs.')
        lhs_name, rhs_name = node.input
        bias_name = None
        trans_b = False
        if attr(node, 'transA', 0) not in (0, False):
            raise ValueError(f'{node_name}: transA is not supported.')
        rhs_is_constant = rhs_name in ctx.initializers or (
            rhs_name in ctx.value_tensors and ctx.value_tensors[rhs_name].is_parameter
        )
    else:
        if len(node.input) not in (2, 3):
            raise ValueError(f'{node_name}: Gemm must have 2 or 3 inputs.')
        if float(attr(node, 'alpha', 1.0)) != 1.0:
            raise ValueError(f'{node_name}: Gemm alpha must be 1.')
        if float(attr(node, 'beta', 1.0)) != 1.0:
            raise ValueError(f'{node_name}: Gemm beta must be 1.')
        if int(attr(node, 'transA', 0)) != 0:
            raise ValueError(f'{node_name}: Gemm transA is not supported.')
        lhs_name, rhs_name = node.input[:2]
        bias_name = node.input[2] if len(node.input) == 3 else None
        trans_b = int(attr(node, 'transB', 0)) == 1
        # Dynamic-RHS Gemm is intentionally unsupported; add a Gemm->MatMul(+Add)
        # canonicalization first if that subset is ever needed.
        rhs_is_constant = True

    lhs_tensor = ctx.canonical_source(lhs_name, node_name)
    if lhs_tensor.is_parameter:
        raise ValueError(f'{node_name}: activation input cannot be constant.')
    out_name = node.output[0]
    out_shape = ctx.output_shape(out_name, node_name)
    out_precision = lhs_tensor.precision if isinstance(lhs_tensor.precision, FloatIntent) else None

    if not rhs_is_constant:
        if bias_name is not None:
            raise ValueError(f'{node_name}: dynamic MatMul does not support fused bias.')
        rhs_tensor = ctx.canonical_source(rhs_name, node_name)
        rhs_shape = ctx.output_shape(rhs_name, node_name)
        n_in = int(rhs_shape[0] if len(rhs_shape) == 1 else rhs_shape[-2])
        n_out = int(1 if len(rhs_shape) == 1 else rhs_shape[-1])
        ctx.emit(
            'matmul',
            node_name,
            inputs=[lhs_tensor, rhs_tensor],
            outputs=[(out_name, out_shape, out_precision)],
            roles=['lhs', 'rhs'],
            metadata=_matmul_meta('MatMul', n_in, n_out, node_name),
            directives=directives,
        )
        return

    rhs_tensor = ctx.parameter_source_for(rhs_name, node_name)
    if not rhs_tensor.is_parameter:
        raise ValueError(f'{node_name}: weight input must be a constant initializer.')
    rhs_data = np.asarray(rhs_tensor.data, dtype=np.float64)
    if trans_b:
        rhs_data = np.transpose(rhs_data)

    if rhs_data.ndim != 2:
        if op_type == 'Gemm':
            raise ValueError(f'{node_name}: ONNX Gemm weight matrix must be rank-2.')
        n_in = int(rhs_data.shape[0] if rhs_data.ndim == 1 else rhs_data.shape[-2])
        n_out = int(1 if rhs_data.ndim == 1 else rhs_data.shape[-1])
        rhs_param = ctx.param_tensor(f'{node_name}_rhs', rhs_data, rhs_tensor.precision)
        ctx.emit(
            'matmul',
            node_name,
            inputs=[lhs_tensor, rhs_param],
            outputs=[(out_name, out_shape, out_precision)],
            roles=['lhs', 'rhs'],
            metadata=_matmul_meta('MatMul', n_in, n_out, node_name),
            directives=directives,
        )
        return

    n_in, n_out = int(rhs_data.shape[0]), int(rhs_data.shape[1])
    rhs_param = ctx.param_tensor(f'{node_name}_weight', rhs_data, rhs_tensor.precision)

    ctx.emit_with_bias(
        'dense',
        node_name,
        inputs=[lhs_tensor, rhs_param],
        roles=['lhs', 'rhs'],
        output=(out_name, out_shape, out_precision),
        metadata=_dense_meta(op_type, n_in, n_out, node_name),
        directives=directives,
        bias_name=bias_name,
        channels=n_out,
        source_class='Gemm',
    )


def _matmul_meta(source_class: str, n_in: int, n_out: int, node_name: str) -> dict:
    return {
        'n_in': n_in,
        'n_out': n_out,
        'layer_class': 'MatMul',
        'source_class': source_class,
        'source_layer': node_name,
    }


def _dense_meta(source_class: str, n_in: int, n_out: int, node_name: str) -> dict:
    return {
        'n_in': n_in,
        'n_out': n_out,
        'layer_class': 'Dense',
        'source_class': source_class,
        'source_layer': node_name,
    }
