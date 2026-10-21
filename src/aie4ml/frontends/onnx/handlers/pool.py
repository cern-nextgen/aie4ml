# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Pooling: ONNX NCHW MaxPool onto a canonical NHWC pool, which a producing conv fuses; ReduceMean and ReduceSum
over the axes between the first and the last onto a pool spanning them, which a producing Dense fuses (FuseOps).
"""

from __future__ import annotations

import numpy as np

from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..shapes import normalize_axis
from ..utils import attr, window_attributes
from .conv import NCHW_OVER_NHWC

REDUCTION_POOLS = {'ReduceMean': 'average', 'ReduceSum': 'sum'}


@onnx_handler('MaxPool')
def _max_pool(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    """Keeps its input's quantization: a max of quantized values is one of them."""
    if len(node.input) != 1 or len(node.output) != 1:
        raise NotImplementedError(f'{node_name}: MaxPool with an Indices output is not supported.')
    x_name = node.input[0]
    ctx.require_order(x_name, NCHW_OVER_NHWC, node_name)
    x = ctx.source_for(x_name, node_name)
    if len(x.shape) != 4:
        raise ValueError(f'{node_name}: MaxPool takes a rank-4 activation, got {x.shape}.')
    window = window_attributes(node, node_name)
    kernel, strides, dilations, pads = (window[k] for k in ('kernel_shape', 'strides', 'dilations', 'pads'))
    out_name = node.output[0]
    batch, channels, out_h, out_w = (int(d) for d in ctx.output_shape(out_name, node_name))
    # A pool rounds its output extent down; ceil_mode is accepted where it does not change it.
    floor = tuple(
        (int(x.shape[1 + a]) + pads[a] + pads[2 + a] - dilations[a] * (kernel[a] - 1) - 1) // strides[a] + 1
        for a in (0, 1)
    )
    if floor != (out_h, out_w):
        raise NotImplementedError(
            f'{node_name}: ceil_mode adds a partial window at the edge ({out_h}x{out_w}, not {floor[0]}x{floor[1]}); '
            're-export with ceil_mode=0 and explicit pads.'
        )
    metadata = {
        'kind': 'max',
        **window,
        'layer_class': 'MaxPooling2D',
        'source_class': 'MaxPool',
        'source_layer': node_name,
    }
    ctx.mirror_precision(out_name, x_name)
    ctx.emit(
        'pool',
        node_name,
        inputs=[x],
        outputs=[(out_name, (batch, out_h, out_w, channels), x.precision)],
        roles=['lhs'],
        metadata=metadata,
        directives=directives,
    )
    ctx.set_order(out_name, NCHW_OVER_NHWC, node_name)


def _reduced_axes(ctx: OnnxImportContext, node, node_name: str, rank: int) -> list:
    """The axes a reduction names, from its attribute or its constant second input (ReduceSum from opset 13,
    ReduceMean from opset 18); every axis when it names none."""
    if len(node.input) > 1 and node.input[1]:
        if node.input[1] not in ctx.initializers:
            raise NotImplementedError(f'{node_name}: {node.op_type} axes must be a constant initializer.')
        axes = np.asarray(ctx.initializers[node.input[1]]).reshape(-1).tolist()
    else:
        axes = attr(node, 'axes', None)
    if not axes and int(attr(node, 'noop_with_empty_axes', 0)):
        raise NotImplementedError(f'{node_name}: {node.op_type} names no axes, so it is an identity; remove it.')
    return sorted({normalize_axis(int(a), rank, node_name, node.op_type) for a in (axes or range(rank))})


@onnx_handler('ReduceMean', 'ReduceSum')
def _reduce_pool(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    """A mean or a sum over the axes between the first and the last, as a pool whose window spans them: (N, T, C)
    over T, say, as Keras' GlobalAveragePooling1D. It computes new values, quantized by the QuantizeLinear after it."""
    x_name, out_name = node.input[0], node.output[0]
    shape = tuple(int(d) for d in ctx.output_shape(x_name, node_name))
    ctx.require_order(x_name, range(len(shape)), node_name)
    axes = _reduced_axes(ctx, node, node_name, len(shape))
    spatial = list(range(1, len(shape) - 1))
    if not spatial or axes != spatial:
        raise NotImplementedError(
            f'{node_name}: {node.op_type} over axes {axes} of {shape}: a pool reduces the axes between the first and '
            f'the last, {spatial}.'
        )
    window = tuple(shape[axis] for axis in spatial)
    ctx.emit(
        'pool',
        node_name,
        inputs=[ctx.source_for(x_name, node_name)],
        outputs=[(out_name, ctx.output_shape(out_name, node_name), None)],
        roles=['lhs'],
        metadata={
            'kind': REDUCTION_POOLS[node.op_type],
            'kernel_shape': window,
            'strides': window,
            'dilations': (1,) * len(window),
            'pads': (0,) * (2 * len(window)),
            'layer_class': node.op_type,
            'source_layer': node_name,
        },
        directives=directives,
    )
