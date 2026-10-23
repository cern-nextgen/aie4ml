"""Op fusion on aie4ml logical IR."""

from __future__ import annotations

import numpy as np

from ..aie_types import QuantIntent
from ..ir import get_backend_context
from ..ir.attributes import FusedActivation, FusedPool, OutputView
from ..op_impls.family_registry import fuses, get_family_resolver_registry
from .base import AIEPass


def _row_permutation(shape, axis_order):
    """Canonical row -> the row the view's own order put it at, or None when the two agree."""
    shape = [int(d) for d in shape]
    axis_order = [int(a) for a in axis_order]
    if axis_order == sorted(axis_order):
        return None
    view_shape = [shape[axis] for axis in axis_order]
    rows = np.arange(int(np.prod(view_shape[1:]))).reshape(view_shape[1:])
    canonical_axes = [axis_order.index(axis) - 1 for axis in range(1, len(shape))]
    return rows.transpose(canonical_axes).reshape(-1)


class FuseOps(AIEPass):
    """Fuse each op into the op producing its input, where that op's family fuses it: a ReLU (the fused_activation
    attribute), a pool (fused_pool) and a reshape the producer writes as a view (output_view); a linear activation, a
    cast, goes.

    One walk in graph order, so each producer is final by the time its consumer is reached and a chain folds whole in
    any order: a ReLU, a pool, a ReLU, a flatten; or a reshape regrouping rows, then the pool over them. A ReLU nothing
    fuses stays, and is refused before the design search; a pool or a reshape nothing fuses is refused here.
    """

    def __init__(self):
        self.name = 'fuse_ops'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False
        for node in list(graph.nodes):
            if node.op_type == 'activation':
                changed = self._fuse_activation(graph, node) or changed
            elif node.op_type == 'pool':
                changed = self._fuse_pool(graph, node) or changed
            elif node.op_type == 'reshape':
                changed = self._fold_reshape(graph, node) or changed
        return changed

    def _fuse_activation(self, graph, node) -> bool:
        """The output tensor of the op fusing a ReLU carries the post-activation precision."""
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            return False
        activation = (node.metadata.get('activation', '') or '').lower()
        if activation in ('linear', ''):  # a pure precision cast, whatever precedes it
            graph.remove_node(node, mode='contract')
            return True
        producer = node.inputs[0].producer
        if activation != 'relu' or producer is None or len(producer.outputs) != 1 or not fuses(producer, 'relu'):
            return False
        pool = producer.attributes.get('fused_pool')
        if pool is not None and pool.kind != 'max':  # its ReLU runs before the pool: commutes with a max only
            return False
        producer.attributes['fused_activation'] = FusedActivation('relu')
        graph.remove_node(node, mode='contract')
        return True

    def _fuse_pool(self, graph, pool) -> bool:
        """The fused_pool attribute holds the window and the precision the pool reads its input in: a max selects one of
        those values, so its output keeps that precision; an average or a sum rounds a new value into its own."""
        source, pooled = pool.inputs[0], pool.outputs[0]
        producer = source.producer
        kind = pool.metadata['kind']
        if not fuses(producer, f'{kind}_pool'):
            raise NotImplementedError(
                f'{pool.name}: its {kind} pool runs fused into the op producing its input, but '
                f'{producer.op_type if producer else "the graph input"} does not fuse it.'
            )
        if len(producer.outputs) != 1 or len(source.consumers) != 1 or source.name in graph.output_tensor_names:
            raise NotImplementedError(
                f'{pool.name}: {producer.name} writes only the pooled tensor, but {source.name!r} is also '
                f'read elsewhere ({len(source.consumers)} consumers, or a graph output).'
            )
        if pool.directives:
            raise ValueError(
                f'{pool.name}: a fused pool runs inside {producer.name}; direct that layer instead, not the pool.'
            )
        if 'fused_pool' in producer.attributes:
            raise NotImplementedError(f'{pool.name}: {producer.name} fuses one pool, and already has one.')
        if kind == 'max':
            if not keeps_values(source.precision, pooled.precision):
                raise NotImplementedError(
                    f'{pool.name}: a fused max pool keeps its input quantization, but its output is quantized '
                    f'{pooled.precision}, not {source.precision}.'
                )
            pooled.precision = source.precision  # the producer rounds and saturates what the pool then selects
        window = {k: tuple(int(v) for v in pool.metadata[k]) for k in ('kernel_shape', 'strides', 'dilations', 'pads')}
        producer.attributes['fused_pool'] = FusedPool(kind=kind, **window, input_precision=source.precision)
        graph.remove_node(pool, mode='contract')
        return True

    def _fold_reshape(self, graph, node) -> bool:
        """Fold a view of an output into the producing op: it writes the view directly.

        The producing family states which views it can emit, so nothing here names an op type.
        A view may also change the order of the rows a consumer reduces over; the consuming
        family adopts that order into its constants.
        """
        source, out_tv = node.inputs[0], node.outputs[0]
        producer = source.producer
        kind = node.metadata['view']
        if not fuses(producer, kind):
            raise NotImplementedError(
                f'{node.name}: {kind} must be written by its producer, but '
                f'{producer.op_type if producer else "the graph input"} does not emit that view.'
            )
        if len(source.consumers) != 1:
            raise NotImplementedError(
                f'{node.name}: {source.name!r} has {len(source.consumers)} consumers, so its producer '
                f'cannot write {kind} for all of them.'
            )

        if not keeps_values(source.precision, out_tv.precision):
            raise NotImplementedError(
                f'{node.name}: {kind} only rearranges values, but {out_tv.name!r} is quantized {out_tv.precision}, '
                f'not {source.precision}.'
            )
        out_tv.precision = source.precision  # the producer's rounding is what the view carries

        order = _row_permutation(source.shape, node.metadata['axis_order'])
        if order is not None:
            for consumer in list(out_tv.consumers):
                get_family_resolver_registry().get(consumer.op_type).reorder_reduction_rows(consumer, out_tv, order)

        producer.attributes['output_view'] = OutputView(kind)
        graph.remove_node(node, mode='contract')
        return True


def keeps_values(source, result) -> bool:
    """Whether an op that only moves or selects values (a reshape, a max pool) can carry `source`'s quantization into
    `result`: they must share a representation; how `result` would round or saturate never applies."""
    if isinstance(source, QuantIntent) and isinstance(result, QuantIntent):
        return (source.width, source.frac, source.signed) == (result.width, result.frac, result.signed)
    return source == result
