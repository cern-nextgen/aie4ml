# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from ..ir import get_backend_context
from ..ir.attributes import Scale
from ..ir.graph import has_input_role
from ..op_impls.family_registry import fuses
from .base import AIEPass


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
            if fuses(consumer, 'input_scale'):  # its resolution checks the value
                _accumulate_scale(consumer, 'input_scale', scale)
                graph.remove_node(scale_node, mode='bypass')
                changed = True
                continue

            # A requantization scales by a shift: a power of two (a producer applies no multiplier).
            if fuses(producer, 'output_scale') and math.log2(scale).is_integer():
                if len(in_tensor.consumers) != 1:
                    raise NotImplementedError(
                        f'{scale_node.name}: cannot fold output scale because {in_tensor.name!r} '
                        f'has {len(in_tensor.consumers)} consumers.'
                    )
                _accumulate_scale(producer, 'output_scale', scale)
                graph.remove_node(scale_node, mode='contract')
                changed = True
                continue

            # A scale on an operand of a product scales the product: a consumer linear in what it reads (no bias, no
            # activation) takes it on its output. The pre-scale tensor keeps the range (`_pin_back`), so a power of two
            # leaves every code as it was.
            linear = consumers[0] if len(consumers) == 1 and _linear(consumers[0]) else None
            if fuses(linear, 'output_scale'):
                _accumulate_scale(linear, 'output_scale', scale)
                graph.remove_node(scale_node, mode='bypass')
                changed = True
                continue

            raise NotImplementedError(
                f'{scale_node.name}: no adjacent op absorbs this scale '
                f'(producer={producer.op_type if producer else None}, '
                f'consumers={[c.op_type for c in consumers]}); there is no standalone scale kernel.'
            )

        return changed


def _linear(node) -> bool:
    """Whether `node` is linear in each input: no bias added, no activation applied."""
    return not has_input_role(node, 'bias') and 'fused_activation' not in node.attributes


def _accumulate_scale(node, name, scale):
    existing = node.attributes.get(name)
    node.attributes[name] = Scale(scale * existing.scale if existing is not None else scale)
