"""Fuse each attention head, Q·Kᵀ [+ U] -> Softmax -> ·V, into one 'attention' op."""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

from ..aie_types import QuantIntent, RoundingMode, SaturationMode
from ..ir import get_backend_context
from ..ir.graph import OpNode, has_input_role, input_tensor_for_role
from ..op_impls.family_registry import fusions_of
from ..op_impls.registry import get_op_impl_registry
from .base import AIEPass

ATTENTION = 'attention'


class FuseAttention(AIEPass):
    """Fuse a matmul reading K transposed, the Softmax over its rows and the matmul with V, and an Add of a logits bias
    U before the Softmax, into one 'attention' op where a variant implements it. Runs after FoldScale and FoldViewOps,
    which leave the scales and K's transpose as attributes. The op takes the Softmax's name and directives; a Softmax
    directed beyond 'parallelism', or a directed matmul or Add, stays as it is."""

    def __init__(self):
        self.name = 'fuse_attention'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False
        for softmax in [node for node in graph.nodes if node.op_type == 'softmax']:
            head = _head(graph, softmax)
            if head is None:
                continue
            fused = _attention_node(head)
            if not _implemented(fused, ctx.device):
                continue
            _replace(graph, head, fused)
            changed = True
        return changed


def absorb_projections(model_or_ctx) -> bool:
    """Fold into each attention the undirected Dense ops projecting its Q, K and V from one x (`_projections`), where a
    variant implements the result. The design search weighs it (`pipeline.SEARCHED_REWRITES`)."""
    ctx = get_backend_context(model_or_ctx)
    graph = ctx.ir.logical
    changed = False
    for attention in [node for node in graph.nodes if node.op_type == ATTENTION and has_input_role(node, 'q')]:
        denses = _projections(graph, attention)
        head = None if denses is None else _head_node(attention, denses)
        if head is not None and _implemented(head, ctx.device):
            _absorb(graph, attention, denses, head)
            changed = True
    return changed


def _implemented(node: OpNode, device) -> bool:
    variants = get_op_impl_registry().candidates(ATTENTION)
    return any(fusions_of(node) <= v.supported_fusions and v.matches(node, device, {}) for v in variants)


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


class _Head(NamedTuple):
    scores: OpNode
    softmax: OpNode
    context: OpNode
    add: Optional[OpNode]  # the Add of the bias, if any
    bias: Optional[object]  # its U

    @property
    def nodes(self):
        return [node for node in (self.scores, self.add, self.softmax) if node is not None] + [self.context]


def _head(graph, softmax: OpNode) -> Optional[_Head]:
    """The ops of one head, or None where they do not form one."""
    logits = softmax.inputs[0]
    if _single_reader(graph, logits, 'softmax') is not softmax:
        return None
    add, bias, reader = logits.producer, None, softmax
    if add is not None and add.op_type == 'add':  # logits + U
        if add.directives or add.attributes:
            return None
        lhs, rhs = input_tensor_for_role(add, 'lhs'), input_tensor_for_role(add, 'rhs')
        logits, bias = (lhs, rhs) if lhs.producer is not None and lhs.producer.op_type == 'matmul' else (rhs, lhs)
        if bias.is_parameter or not (_int(bias.precision, 8, True) or _int(bias.precision, 16, True)):
            return None
        if not _exact_or_int8(softmax.inputs[0]) or tuple(bias.shape) != (logits.shape[0], logits.shape[0]):
            return None
        reader = add
    else:
        add = None
    scores = logits.producer
    if scores is None or scores.op_type != 'matmul' or _single_reader(graph, logits, reader.op_type) is not reader:
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
    if not all(_int(t.precision, 8, True) for t in (q, k, v, context.outputs[0])) or not _exact_or_int8(logits):
        return None
    if not _int(p.precision, 8, False) or int(p.precision.frac) > 8:  # uint8 probabilities the graph never clips
        return None
    if len(q.shape) != 2 or tuple(q.shape) != tuple(k.shape) or tuple(v.shape) != tuple(k.shape):
        return None
    return _Head(scores, softmax, context, add, bias)


def _exact_or_int8(logits) -> bool:
    """Quantized int8 logits, or exact int32 ones (exact products, or their sum with a bias)."""
    return _int(logits.precision, 8, True) or _int(logits.precision, 32, True)


def _scale(node: OpNode, name: str) -> float:
    scale = node.attributes.get(name)
    return 1.0 if scale is None else float(scale.scale)


def _attention_node(head: _Head) -> OpNode:
    scores, softmax, context = head.scores, head.softmax, head.context
    q, k = input_tensor_for_role(scores, 'lhs'), input_tensor_for_role(scores, 'rhs')
    v = input_tensor_for_role(context, 'rhs')
    logits = softmax.inputs[0]  # the Softmax's input format: the logits', or their sum's with the bias
    bias = [] if head.bias is None else [head.bias]
    return OpNode(
        name=softmax.name,
        op_type=ATTENTION,
        inputs=[q, k, v, *bias],
        outputs=list(context.outputs),
        metadata={
            'layer_class': 'Attention',
            'source_layer': softmax.metadata.get('source_layer', softmax.name),
            'fused': [node.name for node in head.nodes],
            'logit_frac': int(logits.precision.frac),
            'logit_bits': int(logits.precision.width),
            'probability_frac': int(softmax.outputs[0].precision.frac),
            'score_scale': _scale(scores, 'output_scale'),
            'temperature': _scale(softmax, 'input_scale'),
            'output_scale': _scale(context, 'output_scale'),
        },
        directives=dict(softmax.directives),
        roles={q.name: 'q', k.name: 'k', v.name: 'v', **({head.bias.name: 'bias'} if bias else {})},
    )


def _replace(graph, head: _Head, fused: OpNode) -> None:
    nodes = head.nodes
    for tensor in fused.inputs:
        tensor.consumers = [node for node in tensor.consumers if node not in nodes] + [fused]
    for tensor in fused.outputs:
        tensor.producer = fused
    for node in nodes[:-1]:
        del graph.tensors[node.outputs[0].name]
    graph.nodes[graph.nodes.index(head.context)] = fused
    for node in nodes[:-1]:
        graph.nodes.remove(node)


def _projections(graph, attention: OpNode) -> Optional[Dict[str, OpNode]]:
    """The Dense writing each of the head's Q, K and V, where all three read one int8 x through int8 weights, carry
    only a bias and an output scale, round half to even with saturation, and feed nothing else."""
    denses = {}
    for role in ('q', 'k', 'v'):
        tensor = input_tensor_for_role(attention, role)
        dense = tensor.producer
        if dense is None or dense.op_type != 'dense' or dense.directives or set(dense.attributes) - {'output_scale'}:
            return None
        if len(tensor.consumers) != 1 or tensor.name in graph.output_tensor_names:
            return None
        if tensor.precision.rounding != RoundingMode.RND_CONV or tensor.precision.saturation != SaturationMode.SAT:
            return None
        if not _int(input_tensor_for_role(dense, 'rhs').precision, 8, True):
            return None
        denses[role] = dense
    x = {input_tensor_for_role(dense, 'lhs').name: input_tensor_for_role(dense, 'lhs') for dense in denses.values()}
    if len(x) != 1 or not _int(next(iter(x.values())).precision, 8, True):
        return None
    return denses


def _head_node(attention: OpNode, denses: Dict[str, OpNode]) -> Optional[OpNode]:
    """The attention reading x, the projections' weights and biases, and the logits' bias if any; or None where two
    share a constant."""
    x = input_tensor_for_role(denses['q'], 'lhs')
    inputs, roles = [x], {x.name: 'x'}
    for role, dense in denses.items():
        for kind, name in (('rhs', f'w{role}'), ('bias', f'b{role}')):
            if has_input_role(dense, kind):
                tensor = input_tensor_for_role(dense, kind)
                inputs.append(tensor)
                roles[tensor.name] = name
    if has_input_role(attention, 'bias'):
        u = input_tensor_for_role(attention, 'bias')
        inputs.append(u)
        roles[u.name] = 'bias'
    if len(roles) != len(inputs):
        return None
    projections = {
        role: {'frac': int(dense.outputs[0].precision.frac), 'output_scale': _scale(dense, 'output_scale')}
        for role, dense in denses.items()
    }
    return OpNode(
        name=attention.name,
        op_type=ATTENTION,
        inputs=inputs,
        outputs=list(attention.outputs),
        metadata={
            **attention.metadata,
            'fused': [dense.name for dense in denses.values()] + list(attention.metadata['fused']),
            'projections': projections,
        },
        directives=dict(attention.directives),
        roles=roles,
    )


def _absorb(graph, attention: OpNode, denses: Dict[str, OpNode], head: OpNode) -> None:
    absorbed = [*denses.values(), attention]
    for tensor in head.inputs:
        tensor.consumers = [node for node in tensor.consumers if node not in absorbed] + [head]
    for tensor in head.outputs:
        tensor.producer = head
    for dense in denses.values():
        del graph.tensors[dense.outputs[0].name]
        graph.nodes.remove(dense)
    graph.nodes[graph.nodes.index(attention)] = head
