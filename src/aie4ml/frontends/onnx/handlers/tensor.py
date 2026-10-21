# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Shape/view ops: Transpose, Flatten/Reshape, Slice, Split, Concat."""

from __future__ import annotations

import numpy as np

from ....ir.graph import VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS
from ..context import OnnxImportContext
from ..registry import onnx_handler
from ..shapes import normalize_axis
from ..utils import attr


@onnx_handler('Transpose')
def _transpose(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) != 1:
        raise ValueError(f'{node_name}: Transpose must have exactly 1 input.')
    src = ctx.source_for(node.input[0], node_name)
    in_shape = ctx.output_shape(node.input[0], node_name)
    perm = [int(x) for x in list(attr(node, 'perm'))]
    if sorted(perm) != list(range(len(in_shape))):
        raise ValueError(f'{node_name}: invalid permutation {perm} for rank {len(in_shape)}.')
    out_name = node.output[0]

    if src.is_parameter:
        data = np.transpose(np.asarray(src.data, dtype=np.float64), axes=perm)
        ctx.bind(out_name, ctx.param_tensor(out_name, data, src.precision))
        return

    view = ctx.order_of(node.input[0]) or tuple(range(len(in_shape)))
    ctx.bind(out_name, src)
    ctx.set_order(out_name, [view[axis] for axis in perm], node_name)


@onnx_handler('Flatten', 'Reshape')
def _flatten(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    """Flatten one sample to [1, K], or regroup the rows, the last axis kept: (G*T, C) as (G, T, C), say. The
    canonical tensor is unchanged, so the op only records which row order the ONNX value implies -- a consumer's
    weight rows follow that order.
    """
    src_name = node.input[0]
    src = ctx.source_for(src_name, node_name)
    out_name = node.output[0]
    out_shape = tuple(int(d) for d in ctx.output_shape(out_name, node_name))
    logical = tuple(int(d) for d in src.shape)
    if len(out_shape) == 2 and int(out_shape[0]) == 1 and int(np.prod(logical[1:])) == int(out_shape[1]):
        view = VIEW_FLATTEN_2D
    elif out_shape[-1] == logical[-1] and ctx.order_of(src_name) is None:
        view = VIEW_REGROUP_ROWS
    else:
        raise NotImplementedError(
            f'{node_name}: a {node.op_type} to {out_shape} neither flattens one sample to [1, K] nor regroups the '
            f'rows of {logical}, its last axis kept.'
        )
    # The flattened row is canonical; `axis_order` records how the view ravelled the axes, so
    # the fold can hand that order to whoever reduces over the row.
    ctx.set_order(out_name, None, node_name)

    ctx.emit(
        'reshape',
        node_name,
        inputs=[src],
        outputs=[(out_name, out_shape, src.precision)],
        roles=['lhs'],
        metadata={
            'view': view,
            'axis_order': ctx.order_of(src_name) or tuple(range(len(logical))),
            'layer_class': node.op_type,
            'source_layer': node_name,
        },
        directives=directives,
    )


@onnx_handler('Slice', 'Split')
def _slice_split(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    op_type = node.op_type
    src_name = node.input[0]
    src_shape = ctx.output_shape(src_name, node_name)

    if op_type == 'Slice':
        if len(node.input) not in (3, 4, 5):
            raise ValueError(f'{node_name}: Slice must have 3 to 5 inputs.')
        for name in node.input[1:]:
            if name and name not in ctx.initializers:
                raise NotImplementedError(f'{node_name}: Slice parameters must be constant initializers.')
        starts = np.asarray(ctx.initializers[node.input[1]], dtype=np.int64).reshape(-1)
        ends = np.asarray(ctx.initializers[node.input[2]], dtype=np.int64).reshape(-1)
        axes = (
            np.asarray(ctx.initializers[node.input[3]], dtype=np.int64).reshape(-1)
            if len(node.input) >= 4 and node.input[3]
            else np.arange(starts.size, dtype=np.int64)
        )
        steps = (
            np.asarray(ctx.initializers[node.input[4]], dtype=np.int64).reshape(-1)
            if len(node.input) >= 5 and node.input[4]
            else np.ones(starts.size, dtype=np.int64)
        )
        if starts.size != 1 or ends.size != 1 or axes.size != 1 or steps.size != 1:
            raise NotImplementedError(f'{node_name}: Slice currently supports exactly one sliced axis.')
        if int(steps[0]) != 1:
            raise NotImplementedError(f'{node_name}: Slice currently supports only unit steps.')
        axis = normalize_axis(int(axes[0]), len(src_shape), node_name, 'Slice')
        start = max(0, min(int(src_shape[axis]), int(starts[0])))
        end = max(start, min(int(src_shape[axis]), int(ends[0])))
        ranges = [(start, end - start)]
    else:
        if len(node.input) not in (1, 2):
            raise ValueError(f'{node_name}: Split must have 1 or 2 inputs.')
        axis = normalize_axis(int(attr(node, 'axis', 0)), len(src_shape), node_name, 'Split')
        if len(node.input) == 2:
            split_name = node.input[1]
            if split_name not in ctx.initializers:
                raise NotImplementedError(f'{node_name}: Split sizes must be a constant initializer.')
            sizes = [int(x) for x in np.asarray(ctx.initializers[split_name], dtype=np.int64).reshape(-1)]
        else:
            sizes = [int(x) for x in list(attr(node, 'split', []))]
            if not sizes:
                if int(src_shape[axis]) % len(node.output) != 0:
                    raise ValueError(f'{node_name}: equal Split does not divide axis {axis} exactly.')
                sizes = [int(src_shape[axis]) // len(node.output) for _ in node.output]
        if len(sizes) != len(node.output) or any(size <= 0 for size in sizes):
            raise ValueError(f'{node_name}: Split sizes must be positive and match output count.')
        if sum(sizes) != int(src_shape[axis]):
            raise ValueError(f'{node_name}: Split sizes must cover axis {axis} exactly.')
        offset = 0
        ranges = []
        for size in sizes:
            ranges.append((offset, size))
            offset += size

    source = ctx.source_for(src_name, node_name)
    for out_name in node.output:
        ctx.propagate_order(src_name, out_name)
    outputs = [(out_name, ctx.canonical_shape(out_name, node_name), source.precision) for out_name in node.output]
    ctx.emit(
        op_type.lower(),
        node_name,
        inputs=[source],
        outputs=outputs,
        roles=['lhs'],
        metadata={
            'axis': ctx.canonical_axis(src_name, axis),
            'slices': [{'start': start, 'extent': extent} for start, extent in ranges],
            'layer_class': op_type,
            'source_class': op_type,
            'source_layer': node_name,
        },
        directives=directives,
    )


@onnx_handler('Concat')
def _concat(ctx: OnnxImportContext, node, node_name: str, directives: dict) -> None:
    if len(node.input) < 1:
        raise ValueError(f'{node_name}: Concat must have at least one input.')
    sources = [ctx.source_for(name, node_name) for name in node.input]
    if any(src.is_parameter for src in sources):
        raise ValueError(f'{node_name}: Concat currently supports activation tensors only.')

    shapes = [ctx.output_shape(name, node_name) for name in node.input]
    rank = len(shapes[0])
    if rank < 1:
        raise ValueError(f'{node_name}: Concat does not accept scalar inputs.')
    if any(len(shape) != rank for shape in shapes):
        raise ValueError(f'{node_name}: Concat inputs must have the same rank.')
    axis = normalize_axis(int(attr(node, 'axis', -1)), rank, node_name, 'Concat')
    orders = {ctx.order_of(name) or tuple(range(rank)) for name in node.input}
    if len(orders) != 1:
        raise ValueError(
            f'{node_name}: Concat inputs view their tensors in different axis orders {sorted(orders)}, so axis '
            f'{axis} is canonical axis {sorted({ctx.canonical_axis(name, axis) for name in node.input})} of them; '
            'concatenating them is not one concat of canonical tensors.'
        )

    prefix, suffix = tuple(shapes[0][:axis]), tuple(shapes[0][axis + 1 :])
    for shape in shapes[1:]:
        if tuple(shape[:axis]) != prefix or tuple(shape[axis + 1 :]) != suffix:
            raise ValueError(f'{node_name}: Concat input shapes disagree outside axis {axis}: {shapes}.')

    precision = sources[0].precision
    for src in sources[1:]:
        if src.precision != precision:
            raise ValueError(f'{node_name}: Concat inputs must use identical precision contracts.')

    ctx.propagate_order(node.input[0], node.output[0])
    ctx.emit(
        'concat',
        node_name,
        inputs=sources,
        outputs=[(node.output[0], ctx.canonical_shape(node.output[0], node_name), precision)],
        metadata={
            'axis': ctx.canonical_axis(node.input[0], axis),
            'layer_class': 'Concat',
            'source_class': 'Concat',
            'source_layer': node_name,
        },
        directives=directives,
    )
