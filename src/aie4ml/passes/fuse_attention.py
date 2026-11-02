"""Fuse each attention head, Q·Kᵀ -> Softmax -> ·V, into one 'attention' op."""

from __future__ import annotations

from typing import Optional, Tuple

from ..aie_types import QuantIntent
from ..ir import get_backend_context
from ..ir.graph import OpNode, input_tensor_for_role
from ..op_impls.registry import get_op_impl_registry
from .base import AIEPass

ATTENTION = 'attention'


class FuseAttention(AIEPass):
    """Replace a matmul reading K transposed, the Softmax over its rows and the matmul of the probabilities with V
    by one 'attention' op reading Q, K and V, where an attention variant implements it on the device.

    Runs after FoldScale and FoldViewOps: the logits' prescale is the scores matmul's output_scale, a temperature the
    Softmax's input_scale, and K's transpose the scores matmul's io_view. The op keeps the probabilities unnormalised
    where the graph rounds them to uint8, so its output is within about a code of the QDQ model's. It takes the
    Softmax's name and its directives: a 'parallelism' there directs the attention; a Softmax directed otherwise (its
    approximation, layout, microtiling), or a matmul directed at all, stays as it is.
    """

    def __init__(self):
        self.name = 'fuse_attention'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False
        for softmax in [node for node in graph.nodes if node.op_type == 'softmax']:
            chain = _head(graph, softmax)
            if chain is None:
                continue
            fused = _attention_node(*chain)
            if any(v.matches(fused, ctx.device, {}) for v in get_op_impl_registry().candidates(ATTENTION)):
                _replace(graph, chain, fused)
                changed = True
        return changed


def _single_reader(graph, tensor, reader_type: str) -> Optional[OpNode]:
    """The one op reading `tensor`, when it is of `reader_type` and the tensor is no graph output."""
    if len(tensor.consumers) != 1 or tensor.name in graph.output_tensor_names:
        return None
    reader = tensor.consumers[0]
    return reader if reader.op_type == reader_type else None


def _int(precision, width: int, signed: bool, frac: Optional[int] = None) -> bool:
    return (
        isinstance(precision, QuantIntent)
        and int(precision.width) == width
        and bool(precision.signed) == signed
        and (frac is None or int(precision.frac) == frac)
    )


def _head(graph, softmax: OpNode) -> Optional[Tuple[OpNode, OpNode, OpNode]]:
    """(scores, softmax, context) of one head, or None where the three do not form one."""
    logits = softmax.inputs[0]
    scores = logits.producer
    if scores is None or scores.op_type != 'matmul' or _single_reader(graph, logits, 'softmax') is not softmax:
        return None
    context = _single_reader(graph, softmax.outputs[0], 'matmul')
    if context is None or scores.directives or context.directives or set(softmax.directives) - {'parallelism'}:
        return None
    q, k = input_tensor_for_role(scores, 'lhs'), input_tensor_for_role(scores, 'rhs')
    p, v = input_tensor_for_role(context, 'lhs'), input_tensor_for_role(context, 'rhs')
    views = scores.attributes.get('io_view')
    if p is not softmax.outputs[0] or views is None or tuple(views.inputs.get(k.name, ())) != (1, 0):
        return None
    if q.name in views.inputs or context.attributes.get('io_view') is not None:
        return None
    if set(scores.attributes) - {'io_view', 'output_scale'} or set(softmax.attributes) - {'input_scale'}:
        return None
    if set(context.attributes) - {'output_scale'}:
        return None
    if not all(_int(t.precision, 8, True) for t in (q, k, v, context.outputs[0])):
        return None
    if not (_int(logits.precision, 8, True) or _int(logits.precision, 32, True)):  # quantized, or exact products
        return None
    if not _int(p.precision, 8, False) or int(p.precision.frac) > 8:  # uint8 probabilities the graph never clips
        return None
    if len(q.shape) != 2 or tuple(q.shape) != tuple(k.shape) or tuple(v.shape) != tuple(k.shape):
        return None
    return scores, softmax, context


def _scale(node: OpNode, name: str) -> float:
    scale = node.attributes.get(name)
    return 1.0 if scale is None else float(scale.scale)


def _attention_node(scores: OpNode, softmax: OpNode, context: OpNode) -> OpNode:
    q, k = input_tensor_for_role(scores, 'lhs'), input_tensor_for_role(scores, 'rhs')
    v = input_tensor_for_role(context, 'rhs')
    return OpNode(
        name=softmax.name,
        op_type=ATTENTION,
        inputs=[q, k, v],
        outputs=list(context.outputs),
        metadata={
            'layer_class': 'Attention',
            'source_layer': softmax.metadata.get('source_layer', softmax.name),
            'fused': [scores.name, softmax.name, context.name],
            'logit_frac': int(scores.outputs[0].precision.frac),
            'logit_bits': int(scores.outputs[0].precision.width),
            'probability_frac': int(softmax.outputs[0].precision.frac),
            'score_scale': _scale(scores, 'output_scale'),
            'temperature': _scale(softmax, 'input_scale'),
            'output_scale': _scale(context, 'output_scale'),
        },
        directives=dict(softmax.directives),
        roles={q.name: 'q', k.name: 'k', v.name: 'v'},
    )


def _replace(graph, chain: Tuple[OpNode, OpNode, OpNode], fused: OpNode) -> None:
    scores, softmax, context = chain
    for tensor in fused.inputs:
        tensor.consumers = [node for node in tensor.consumers if node not in chain] + [fused]
    for tensor in fused.outputs:
        tensor.producer = fused
    for tensor in (scores.outputs[0], softmax.outputs[0]):
        del graph.tensors[tensor.name]
    at = graph.nodes.index(context)
    graph.nodes[at] = fused
    for node in (scores, softmax):
        graph.nodes.remove(node)
