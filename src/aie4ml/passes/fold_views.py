from __future__ import annotations

import numpy as np

from ..ir import TraitInstance, get_backend_context
from ..op_impls import get_family_resolver_registry
from .base import AIEPass
from .utils import keeps_values


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


_VIEW_OPS = ('concat', 'slice', 'split')


class FoldViewOps(AIEPass):
    """Lower semantic view ops into explicit tensor view traits."""

    def __init__(self):
        self.name = 'fold_view_ops'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for node in list(graph.nodes):
            if node.op_type == 'transpose':
                changed = self._fold_transpose(graph, node) or changed
            elif node.op_type == 'concat':
                changed = self._fold_concat(node) or changed
            elif node.op_type in ('slice', 'split'):
                changed = self._fold_slice(node) or changed
            elif node.op_type == 'reshape':
                changed = self._fold_reshape(graph, node) or changed

        return changed

    def _io_view_data(self, node):
        trait = node.traits.get('io_view')
        if trait is None:
            trait = TraitInstance('io_view', {'inputs': {}, 'outputs': {}})
            node.add_trait(trait)
        trait.data.setdefault('inputs', {})
        trait.data.setdefault('outputs', {})
        return trait.data

    def _fold_transpose(self, graph, node) -> bool:
        in_tv = node.inputs[0]
        out_tv = node.outputs[0]

        if out_tv.name in graph.output_tensor_names:
            raise NotImplementedError(
                f'{node.name}: transpose feeds graph output {out_tv.name!r}. A transpose folds into '
                'its consumers as an io_view; a graph output has no consumer kernel to carry the view, '
                'and the output boundary emits the producer buffer order as-is.'
            )

        perm = node.metadata.get('perm')
        if perm is None:
            raise ValueError(f'{node.name}: missing transpose permutation metadata.')
        perm = [int(x) for x in perm]

        rank = len(in_tv.shape)
        if len(out_tv.shape) != rank:
            raise ValueError(f'{node.name}: transpose rank mismatch between input and output.')
        if sorted(perm) != list(range(rank)):
            raise ValueError(f'{node.name}: invalid permutation {perm} for rank {rank}.')
        if (node.metadata.get('data_format', 'channels_last') or '').lower() != 'channels_last':
            raise ValueError(f'{node.name}: only channels_last transpose is supported.')

        views = [c.name for c in out_tv.consumers if c.op_type in _VIEW_OPS]
        if views:
            raise NotImplementedError(
                f'{node.name}: transpose feeds the view op(s) {views}, whose axes would then name the untransposed '
                'tensor; a view op must be recorded on the canonical tensor.'
            )

        # buffer_order is derived (the fixed axis reversal); only perm is view data.
        in_view = {
            'perm': [int(p) for p in perm],
        }

        for consumer in out_tv.consumers:
            cons_view = self._io_view_data(consumer)
            cons_view['inputs'][in_tv.name] = dict(in_view)
            cons_view['inputs'].pop(out_tv.name, None)
            if out_tv.name in consumer.roles:
                consumer.roles[in_tv.name] = consumer.roles.pop(out_tv.name)

        graph.remove_node(node, mode='bypass')
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
        registry = get_family_resolver_registry()
        resolver = registry.find(producer.op_type) if producer is not None else None
        if resolver is None or kind not in resolver.supported_output_views:
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
                registry.get(consumer.op_type).reorder_reduction_rows(consumer, out_tv, order)

        producer.add_trait(TraitInstance('output_view', {'kind': kind}))
        graph.remove_node(node, mode='contract')
        return True

    def _fold_concat(self, node) -> bool:
        if len(node.outputs) != 1:
            raise ValueError(f'{node.name}: concat must have exactly one output.')
        if not node.inputs:
            raise ValueError(f'{node.name}: concat must have at least one input.')

        output = node.outputs[0]
        axis = int(node.metadata.get('axis', -1))
        rank = len(output.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(f'{node.name}: concat axis {node.metadata.get("axis")} is out of range for rank {rank}.')

        offset = 0
        slices = []
        for tensor in node.inputs:
            if len(tensor.shape) != rank:
                raise ValueError(f'{node.name}: concat input {tensor.name!r} rank does not match output rank.')
            extent = int(tensor.shape[axis])
            slices.append({'input': tensor.name, 'start': int(offset), 'extent': extent})
            offset += extent

        if int(output.shape[axis]) != int(offset):
            raise ValueError(
                f'{node.name}: concat output axis extent {output.shape[axis]} does not match input sum {offset}.'
            )

        node.add_trait(
            TraitInstance(
                'concat_view',
                {
                    'kind': 'concat',
                    'axis': axis,
                    'output': output.name,
                    'slices': slices,
                },
            )
        )
        return True

    def _fold_slice(self, node) -> bool:
        if len(node.inputs) != 1:
            raise ValueError(f'{node.name}: {node.op_type} view must have exactly one input.')
        if not node.outputs:
            raise ValueError(f'{node.name}: {node.op_type} view must have at least one output.')

        source = node.inputs[0]
        axis = int(node.metadata['axis'])
        rank = len(source.shape)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError(f'{node.name}: {node.op_type} axis {axis} is out of range for rank {rank}.')

        slices = list(node.metadata.get('slices', ()))
        if len(slices) != len(node.outputs):
            raise ValueError(f'{node.name}: {node.op_type} slice metadata does not match output count.')
        normalized = []
        for output, item in zip(node.outputs, slices):
            start = int(item['start'])
            extent = int(item['extent'])
            if start < 0 or extent <= 0 or start + extent > int(source.shape[axis]):
                raise ValueError(f'{node.name}: invalid {node.op_type} range [{start}, {start + extent}).')
            if int(output.shape[axis]) != extent:
                raise ValueError(f'{node.name}: {node.op_type} output {output.name!r} extent mismatch.')
            normalized.append({'output': output.name, 'start': start, 'extent': extent})

        node.add_trait(
            TraitInstance(
                'slice_view',
                {
                    'kind': node.op_type,
                    'axis': axis,
                    'source': source.name,
                    'slices': normalized,
                },
            )
        )
        return True
