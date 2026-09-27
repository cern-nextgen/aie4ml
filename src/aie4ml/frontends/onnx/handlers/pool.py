# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Pooling: ONNX NCHW MaxPool onto a canonical NHWC pool2d, which a producing conv fuses (FusePool)."""

from __future__ import annotations

from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..utils import attr
from .conv import NCHW_OVER_NHWC


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
    auto_pad = attr(node, 'auto_pad', b'NOTSET')
    auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else str(auto_pad)
    if auto_pad != 'NOTSET':
        raise NotImplementedError(
            f'{node_name}: auto_pad={auto_pad} leaves the padding to shape inference; re-export with explicit pads.'
        )
    kernel = tuple(int(k) for k in attr(node, 'kernel_shape', []))
    strides = tuple(int(s) for s in attr(node, 'strides', [1, 1]))
    dilations = tuple(int(d) for d in attr(node, 'dilations', [1, 1]))
    pads = tuple(int(p) for p in attr(node, 'pads', [0, 0, 0, 0]))
    out_name = node.output[0]
    batch, channels, out_h, out_w = (int(d) for d in ctx.output_shape(out_name, node_name))
    # pool2d rounds its output extent down; ceil_mode is accepted where it does not change it.
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
        'kernel_shape': kernel,
        'strides': strides,
        'dilations': dilations,
        'pads': pads,
        'layer_class': 'MaxPooling2D',
        'source_class': 'MaxPool',
        'source_layer': node_name,
    }
    ctx.mirror_precision(out_name, x_name)
    ctx.emit(
        'pool2d',
        node_name,
        inputs=[x],
        outputs=[(out_name, (batch, out_h, out_w, channels), x.precision)],
        roles=['lhs'],
        metadata=metadata,
        directives=directives,
    )
    ctx.set_order(out_name, NCHW_OVER_NHWC, node_name)
