# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from ..ir import get_backend_context
from ..ir.graph import has_input_role
from ..op_impls.family_registry import get_family_resolver_registry
from .base import AIEPass


class FoldBias(AIEPass):
    """Fold a trailing `add(output, const)` into the producing op's bias.

    The ONNX frontend emits bias as a separate `add` (a standalone MatMul+Add, Gemm's C input,
    Conv's B input); this pass is the single place that fuses it, keeping the frontend free of
    fusion decisions.
    """

    def __init__(self):
        self.name = 'fold_bias'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False

        for add in list(graph.nodes):
            if add.op_type != 'add' or len(add.inputs) != 2 or len(add.outputs) != 1:
                continue
            producer_out, bias = _match_biasless_producer(add.inputs[0], add.inputs[1])
            if producer_out is None:
                _refuse_unfused_bias(add)
                continue

            producer = producer_out.producer
            n_out = int(producer_out.shape[-1])
            bias_size = int(np.asarray(bias.data).size)
            if bias_size != n_out:
                raise ValueError(
                    f'{add.name}: a fused bias needs exactly {n_out} elements, one per output channel, '
                    f'got {bias_size}.'
                )
            if len(producer_out.consumers) != 1:
                raise ValueError(
                    f'{add.name}: cannot fold bias because {producer.op_type} output {producer_out.name!r} '
                    f'has {len(producer_out.consumers)} consumers.'
                )

            _fold(graph, producer, producer_out, bias, add)
            changed = True

        return changed


def _match_biasless_producer(lhs, rhs):
    if _is_biasless_producer(lhs) and rhs.is_parameter:
        return lhs, rhs
    if _is_biasless_producer(rhs) and lhs.is_parameter:
        return rhs, lhs
    return None, None


def _is_biasless_producer(tensor) -> bool:
    producer = tensor.producer
    if producer is None or has_input_role(producer, 'bias'):
        return False
    resolver = get_family_resolver_registry().find(producer.op_type)
    return resolver is not None and 'bias' in resolver.supported_fusions


def _refuse_unfused_bias(add) -> None:
    """A constant broadcast over the rows is a bias, which no kernel adds on its own: refused by name where the op
    producing the other operand does not fuse one."""
    for tensor, const in ((add.inputs[0], add.inputs[1]), (add.inputs[1], add.inputs[0])):
        producer = tensor.producer
        if producer is None or not const.is_parameter or tensor.is_parameter:
            continue
        resolver = get_family_resolver_registry().find(producer.op_type)
        fuses = resolver is not None and 'bias' in resolver.supported_fusions
        if np.asarray(const.data).size < np.prod(tensor.shape) and not fuses:
            raise NotImplementedError(
                f'{add.name}: a constant added along the rows is a bias, which runs fused into the op producing '
                f'{tensor.name!r}, and {producer.name} ({producer.op_type}) does not fuse one.'
            )


def _fold(graph, producer, producer_out, bias, add) -> None:
    y = add.outputs[0]

    producer.inputs.append(bias)
    bias.consumers = [c for c in bias.consumers if c is not add]
    bias.consumers.append(producer)
    producer.roles[bias.name] = 'bias'

    producer.outputs = [y]
    y.producer = producer
    graph.tensors.pop(producer_out.name, None)

    graph.nodes.remove(add)
    add.inputs.clear()
    add.outputs.clear()
