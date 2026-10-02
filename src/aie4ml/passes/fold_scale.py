# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..ir import TraitInstance, get_backend_context
from .base import AIEPass

# Single source of truth for scale-absorption capability. Value legality is enforced downstream in the resolver.
_SCALE_ABSORBERS = {
    'dense': 'output_scale',
    'matmul': 'output_scale',
    'softmax': 'input_scale',
}


class FoldScale(AIEPass):
    """Fold constant `scale` nodes into an adjacent op that absorbs them."""

    def __init__(self):
        self.name = 'fold_scale'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for scale_node in list(graph.nodes):
            if scale_node.op_type != 'scale':
                continue
            if len(scale_node.inputs) != 1 or len(scale_node.outputs) != 1:
                raise ValueError(f'{scale_node.name}: scale must have exactly one input and one output.')

            scale = float(scale_node.metadata['scale'])
            if scale <= 0.0:
                raise ValueError(f'{scale_node.name}: scale must be positive, got {scale}.')

            in_tensor = scale_node.inputs[0]
            consumers = list(scale_node.outputs[0].consumers)
            producer = in_tensor.producer

            # A scale before its quantizer scales what is quantized: only the producer's requantization applies it
            # exactly. One after reads quantized codes, which a consumer can scale as it reads them.
            consumer = consumers[0] if len(consumers) == 1 and not scale_node.metadata.get('before_quantizer') else None
            if consumer is not None and _SCALE_ABSORBERS.get(consumer.op_type) == 'input_scale':
                _accumulate_scale(consumer, 'input_scale', scale)
                graph.remove_node(scale_node, mode='bypass')
                changed = True
                continue

            if producer is not None and _SCALE_ABSORBERS.get(producer.op_type) == 'output_scale':
                if len(in_tensor.consumers) != 1:
                    raise NotImplementedError(
                        f'{scale_node.name}: cannot fold output scale because {in_tensor.name!r} '
                        f'has {len(in_tensor.consumers)} consumers.'
                    )
                _accumulate_scale(producer, 'output_scale', scale)
                graph.remove_node(scale_node, mode='contract')
                changed = True
                continue

            raise NotImplementedError(
                f'{scale_node.name}: no adjacent op absorbs this scale '
                f'(producer={producer.op_type if producer else None}, '
                f'consumers={[c.op_type for c in consumers]}); there is no standalone scale kernel.'
            )

        return changed


def _accumulate_scale(node, trait_name, scale):
    existing = node.traits.get(trait_name)
    combined = scale * float(existing.data['scale']) if existing is not None else scale
    node.add_trait(TraitInstance(trait_name, {'scale': combined}))
