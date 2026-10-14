"""Activation fusion pass on aie4ml logical IR."""

from ..ir import TraitInstance, get_backend_context
from ..op_impls.family_registry import get_family_resolver_registry
from .base import AIEPass


class FuseActivationCasts(AIEPass):
    """Fuse relu into a family that lists it (adds fused_activation trait) and contract linear activations out.

    After this pass, no activation nodes remain in the IR and output tensors carry post-activation precision.
    """

    def __init__(self):
        self.name = 'fuse_activation_casts'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for act_node in list(graph.nodes):
            if act_node.op_type != 'activation' or len(act_node.inputs) != 1 or len(act_node.outputs) != 1:
                continue

            activation = (act_node.metadata.get('activation', '') or '').lower()

            if activation == 'relu':
                in_tensor = act_node.inputs[0]
                producer = in_tensor.producer
                if producer is None or len(producer.outputs) != 1:
                    continue
                resolver = get_family_resolver_registry().find(producer.op_type)
                if resolver is None or 'relu' not in resolver.supported_fusions:
                    continue
                producer.add_trait(TraitInstance('fused_activation', {'activation': 'relu'}))
                graph.remove_node(act_node, mode='contract')
                changed = True

            elif activation in ('linear', ''):
                # Pure precision cast — eliminate regardless of what precedes it.
                graph.remove_node(act_node, mode='contract')
                changed = True

        return changed
