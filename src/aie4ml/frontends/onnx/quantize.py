# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""QuantizeLinear / DequantizeLinear -> symmetric per-tensor power-of-two precision on tensors."""

from __future__ import annotations

import numpy as np

from ...aie_types import QuantIntent
from .context import OnnxImportContext
from .registry import onnx_handler
from .utils import dequantize_data, intent_from_qparams

_ONNX_INT_ELEM_TYPE_TO_NP = {2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64}


def _pin_if_unset(tensor, intent: QuantIntent, *, force_signed: bool) -> None:
    """Pin an un-quantized activation to a downstream Quantize's intent."""
    if tensor is None or tensor.is_parameter or tensor.precision is not None:
        return
    tensor.precision = (
        QuantIntent(
            width=int(intent.width),
            frac=int(intent.frac),
            signed=True,
            rounding=intent.rounding,
            saturation=intent.saturation,
        )
        if force_signed
        else intent
    )


def _pin_back(tensor, intent: QuantIntent, *, force_signed: bool = False) -> None:
    """Pin the un-quantized tensors a Quantize reaches back through a Relu, a scale or a bias add -- a Dense's
    accumulator behind MatMul -> Add -> Relu, say -- to its intent; signed behind a Relu, which clips the sign."""
    producer = tensor.producer
    if producer is None:
        return
    if producer.op_type == 'activation' and producer.metadata.get('activation') == 'relu':
        behind, force_signed = producer.inputs[:1], True
    elif producer.op_type in ('scale', 'add'):
        behind = producer.inputs if producer.op_type == 'add' else producer.inputs[:1]
        if producer.op_type == 'scale':
            producer.metadata['before_quantizer'] = True  # it belongs to its producer's requantization
    else:
        return
    for source in behind:
        if source is not None and not source.is_parameter and source.precision is None:
            _pin_if_unset(source, intent, force_signed=force_signed)
            _pin_back(source, intent, force_signed=force_signed)


@onnx_handler('QuantizeLinear')
def _quantize_linear(ctx: OnnxImportContext, node, node_name: str, _directives: dict) -> None:
    if len(node.input) != 3:
        raise ValueError(f'{node_name}: QuantizeLinear must have exactly 3 inputs.')
    src_name, _scale_name, zero_name = node.input
    if zero_name not in ctx.initializers:
        raise ValueError(f'{node_name}: zero_point must be a constant initializer.')
    intent = intent_from_qparams(
        ctx.initializers, node.input[1], zero_name, ctx.initializers[zero_name].dtype, node_name
    )
    out_name = node.output[0]

    if src_name in ctx.value_tensors:
        tensor = ctx.value_tensors[src_name]
        producer = tensor.producer
        relu_producer = (
            producer is not None and producer.op_type == 'activation' and producer.metadata.get('activation') == 'relu'
        )
        if tensor.precision is None:
            _pin_back(tensor, intent)
            tensor.precision = intent
        elif tensor.precision != intent:
            if relu_producer:
                _pin_back(tensor, intent)
                tensor.precision = intent
            else:
                raise ValueError(f'{node_name}: QuantizeLinear intent does not match source tensor precision.')
        ctx.q_aliases[out_name] = ('tensor', tensor, intent, tuple(int(x) for x in tensor.shape))
        ctx.propagate_order(src_name, out_name)
    elif src_name in ctx.input_shapes:
        ctx.q_aliases[out_name] = ('input', src_name, intent, ctx.input_shapes[src_name])
    elif src_name in ctx.initializers:
        ctx.q_aliases[out_name] = (
            'initializer',
            src_name,
            intent,
            tuple(int(x) for x in ctx.initializers[src_name].shape),
        )
    else:
        raise ValueError(f'{node_name}: QuantizeLinear input {src_name} is unsupported.')


@onnx_handler('DequantizeLinear')
def _dequantize_linear(ctx: OnnxImportContext, node, node_name: str, _directives: dict) -> None:
    if len(node.input) != 3:
        raise ValueError(f'{node_name}: DequantizeLinear must have exactly 3 inputs.')
    src_name, scale_name, zero_name = node.input
    out_name = node.output[0]

    if src_name in ctx.q_aliases:
        source_kind, source_ref, q_intent, src_shape = ctx.q_aliases[src_name]
        intent = intent_from_qparams(
            ctx.initializers, scale_name, zero_name, ctx.initializers[zero_name].dtype, node_name
        )
        if intent != q_intent:
            raise ValueError(f'{node_name}: QuantizeLinear/DequantizeLinear parameters do not match.')
        if source_kind == 'tensor':
            tensor = source_ref
        elif source_kind == 'input':
            tensor = ctx.graph_input_tensor(source_ref, src_shape, intent)
        else:  # initializer
            tensor = ctx.param_tensor(
                source_ref,
                dequantize_data(ctx.initializers[source_ref], ctx.initializers, scale_name, zero_name, node_name),
                intent,
            )
        ctx.bind(out_name, tensor)
        ctx.propagate_order(src_name, out_name)
        return

    if src_name in ctx.initializers:
        intent = intent_from_qparams(
            ctx.initializers, scale_name, zero_name, ctx.initializers[src_name].dtype, node_name
        )
        tensor = ctx.param_tensor(
            src_name,
            dequantize_data(ctx.initializers[src_name], ctx.initializers, scale_name, zero_name, node_name),
            intent,
        )
        ctx.bind(out_name, tensor)
        return

    if src_name in ctx.input_shapes:
        raw_np_dtype = _ONNX_INT_ELEM_TYPE_TO_NP.get(ctx.input_dtypes[src_name])
        if raw_np_dtype is None:
            raise ValueError(f'{node_name}: direct DequantizeLinear on graph input requires integer input type.')
        intent = intent_from_qparams(ctx.initializers, scale_name, zero_name, raw_np_dtype, node_name)
        tensor = ctx.graph_input_tensor(src_name, ctx.input_shapes[src_name], intent)
        ctx.bind(out_name, tensor)
        return

    raise ValueError(f'{node_name}: DequantizeLinear input {src_name} is unsupported.')
