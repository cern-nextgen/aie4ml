from __future__ import annotations

from ..ir import get_backend_context
from ..ir.attributes import FoldedView, IOView
from ..ir.graph import FOLDED_VIEWS, ViewPart
from ..op_impls.utils.io import view_perm
from .base import AIEPass


class FoldViewOps(AIEPass):
    """Lower semantic view ops into explicit tensor view attributes."""

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

        return changed

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
        if (node.metadata.get('data_format') or 'channels_last').lower() != 'channels_last':  # unset: channels-last
            raise ValueError(f'{node.name}: only channels_last transpose is supported.')

        views = [c.name for c in out_tv.consumers if c.op_type in FOLDED_VIEWS]
        if views:
            raise NotImplementedError(
                f'{node.name}: transpose feeds the view op(s) {views}, whose axes would then name the untransposed '
                'tensor; a view op must be recorded on the canonical tensor.'
            )

        # A transpose folded into this one already permutes its input: its consumers read the two composed.
        upstream = view_perm(node, in_tv, 'inputs')
        if upstream is not None:
            perm = [int(upstream[axis]) for axis in perm]

        for consumer in out_tv.consumers:
            view = consumer.attributes.get('io_view', IOView())
            inputs = {name: order for name, order in view.inputs.items() if name != out_tv.name}
            if perm != sorted(perm):  # one keeping every axis in place is no view; buffer_order is derived
                inputs[in_tv.name] = tuple(perm)
            consumer.attributes['io_view'] = IOView(inputs, dict(view.outputs))
            if out_tv.name in consumer.roles:
                consumer.roles[in_tv.name] = consumer.roles.pop(out_tv.name)

        graph.remove_node(node, mode='bypass')
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
        parts = []
        for tensor in node.inputs:
            if len(tensor.shape) != rank:
                raise ValueError(f'{node.name}: concat input {tensor.name!r} rank does not match output rank.')
            extent = int(tensor.shape[axis])
            parts.append(ViewPart(tensor.name, int(offset), extent))
            offset += extent

        if int(output.shape[axis]) != int(offset):
            raise ValueError(
                f'{node.name}: concat output axis extent {output.shape[axis]} does not match input sum {offset}.'
            )

        node.attributes['folded_view'] = FoldedView('concat', axis, {output.name: tuple(parts)})
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
        outputs = {}
        for output, item in zip(node.outputs, slices):
            start = int(item['start'])
            extent = int(item['extent'])
            if start < 0 or extent <= 0 or start + extent > int(source.shape[axis]):
                raise ValueError(f'{node.name}: invalid {node.op_type} range [{start}, {start + extent}).')
            if int(output.shape[axis]) != extent:
                raise ValueError(f'{node.name}: {node.op_type} output {output.name!r} extent mismatch.')
            outputs[output.name] = (ViewPart(source.name, start, extent),)

        node.attributes['folded_view'] = FoldedView(node.op_type, axis, outputs)
        return True
