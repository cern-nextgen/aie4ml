# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Conv lowering: ONNX NCHW semantics onto the canonical NHWC conv2d contract."""

from __future__ import annotations

import numpy as np

from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..utils import attr, window_attributes

NCHW_OVER_NHWC = (0, 3, 1, 2)
"""How an ONNX NCHW value views a canonical NHWC tensor."""


@onnx_handler('Conv')
def _conv(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    """Translate ONNX Conv into the canonical conv2d contract.

    Only what translation itself needs is checked here. What the operation must satisfy to make
    sense is the family's contract, and what a kernel can actually run is its variant's.
    """
    if len(node.input) not in (2, 3):
        raise ValueError(f'{node_name}: Conv must have 2 or 3 inputs.')
    x_name, w_name = node.input[:2]
    # ONNX Conv is NCHW; aie4ml activations are canonically NHWC. Export channels-last
    # (input [N,H,W,C] -> Transpose(0,3,1,2) -> Conv ...) so the transpose folds into the view.
    ctx.require_order(x_name, NCHW_OVER_NHWC, node_name)
    x = ctx.source_for(x_name, node_name)
    weight = ctx.parameter_source_for(w_name, node_name)
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise ValueError(
            f'{node_name}: Conv takes a rank-4 activation and rank-4 weights, got {x.shape} and {weight.shape}.'
        )
    cout, _cin_g, kh, kw = (int(d) for d in weight.shape)
    metadata = {
        **window_attributes(node, node_name, (kh, kw)),
        'groups': int(attr(node, 'group', 1)),
        'layer_class': 'Conv2D',
        'source_class': 'Conv',
        'source_layer': node_name,
    }

    # Canonical conv weights: [kh, kw, Cin/groups, Cout]. The variant expands the groups when it
    # packs, so a future depthwise kernel can consume the compact form unchanged.
    w_canonical = np.asarray(weight.data, dtype=np.float64).transpose(2, 3, 1, 0)
    w_param = ctx.param_tensor(f'{node_name}_weight', w_canonical, weight.precision)

    out_name = node.output[0]
    batch, onnx_cout, out_h, out_w = (int(d) for d in ctx.output_shape(out_name, node_name))
    if onnx_cout != cout:
        raise ValueError(f'{node_name}: ONNX output has {onnx_cout} channels but the weights produce {cout}.')
    out_shape = (batch, out_h, out_w, cout)  # the same tensor, in canonical order

    ctx.emit_with_bias(
        'conv2d',
        node_name,
        inputs=[x, w_param],
        roles=['lhs', 'rhs'],
        output=(out_name, out_shape, None),
        metadata=metadata,
        directives=directives,
        bias_name=node.input[2] if len(node.input) == 3 else None,
        channels=cout,
        source_class='Conv',
        order=NCHW_OVER_NHWC,
    )
