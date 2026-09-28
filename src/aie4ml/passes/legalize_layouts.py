"""Insert the layout conversions ops ask for as kernel graphs of their own.

Resolution picks each op's kernel and says what layout it reads. Where an input arrives in another
one -- a strided conv's column-grouped frame, which neither the boundary nor a producing kernel
writes -- the op names a conversion, and this pass makes it an execution instance: a kernel with its own
ports, placement and staging, writing an execution-only value that the op then reads instead. Only
the execution graph changes; the logical graph keeps the model's semantics untouched.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from ..ir import get_backend_context
from ..ir.graph import ExecutionInput, ExecutionInstance, ExecutionValue, OpNode
from .base import AIEPass
from .resolve import output_contracts

log = logging.getLogger(__name__)


def convert_inputs(inst: ExecutionInstance, sources, device) -> list[ExecutionInstance]:
    """The converter kernels `inst` asks for, then `inst` reading their outputs instead; `sources` are the values it
    reads. Pure: `inst` is left as it is, so the parallelism search builds the same kernels this pass inserts."""
    kernels = []
    for conversion in inst.variant.input_conversions(inst.node, inst.config, sources):
        source = inst.input(conversion.source)
        variant, config = conversion.variant, conversion.config
        # The converter implements no logical op: its node only names it.
        node = OpNode(name=conversion.name, op_type=variant.op_type)
        variant.validate_config(node, config, device)
        routes = inst.io_route.get('inputs', {})
        view = inst.port_views[conversion.source]
        kernels.append(
            ExecutionInstance(
                node=node,
                variant=variant,
                ports=variant.build_ports(node, config),
                io_route={
                    'inputs': {conversion.source: routes.get(conversion.source, 'auto')},
                    'outputs': {conversion.target: 'direct'},
                },
                port_views={conversion.source: view, conversion.target: view},
                config=config,
                graph_header=variant.graph_header,
                graph_name=variant.graph_name,
                param_template=variant.param_template,
                inputs=(source,),
                outputs=(conversion.target,),
            )
        )
        # The op now reads the converted value, straight from the converter.
        inst = replace(
            inst,
            inputs=tuple(
                ExecutionInput(conversion.target, item.role, shared_memory=conversion.shared_memory)
                if item.tensor == conversion.source
                else item
                for item in inst.inputs
            ),
            io_route={
                **inst.io_route,
                'inputs': {**{k: v for k, v in routes.items() if k != conversion.source}, conversion.target: 'direct'},
            },
            port_views={
                **{k: v for k, v in inst.port_views.items() if k != conversion.source},
                conversion.target: view,
            },
        )
    return kernels + [inst]


class LegalizeLayouts(AIEPass):
    def __init__(self):
        self.name = 'legalize_layouts'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        execution = ctx.ir.execution
        changed = False
        for inst in list(execution):
            *converters, reader = convert_inputs(
                inst, {item.tensor: execution.values[item.tensor] for item in inst.inputs}, ctx.device
            )
            for converter in converters:
                execution.insert_before(inst.name, converter)
                execution.add_value(ExecutionValue(converter.outputs[0], producer=converter.name))
                execution.tensor_contracts.update(output_contracts(converter))
                log.info(
                    '%s: reads %s through %s, a kernel on a tile of its own',
                    inst.name,
                    converter.inputs[0].tensor,
                    converter.name,
                )
            if converters:
                execution.instances[inst.name] = reader
                changed = True
        execution.verify()
        return changed
