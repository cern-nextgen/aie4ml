from __future__ import annotations

import logging
from dataclasses import replace

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..ir.graph import (
    ExecutionInput,
    ExecutionInstance,
    ExecutionValue,
    ExecutionView,
    OpNode,
    TensorContract,
    input_role,
)
from ..op_impls.family_registry import get_family_resolver_registry
from ..op_impls.utils.io import check_io_view, resolve_io_route, view_perm
from .base import AIEPass

log = logging.getLogger(__name__)


def output_contracts(inst: ExecutionInstance) -> dict[str, TensorContract]:
    """The TensorContracts an instance publishes for its outputs, which its consumers adopt."""
    contracts = {}
    for tensor in inst.outputs:
        contract = inst.variant.output_staging_contract(inst.node, inst.config, tensor)
        if contract is None:
            continue
        contracts[tensor] = TensorContract(
            contract=contract,
            port_staging=inst.staging[tensor],
            inner_shards=inst.variant.output_inner_shards(inst.node, inst.config, tensor),
            view=inst.port_views[tensor],
        )
    return contracts


def resolve_instance(node, device, input_contracts, choice=None) -> ExecutionInstance:
    """The kernel graph that implements `node`, given the contracts its inputs arrive in and, when given, directives
    to resolve it under over its own (a schedule). Pure: nothing is registered."""
    directives = {**(node.directives or {}), **(choice or {})}
    io_route = resolve_io_route(node)  # user intents

    check_io_view(node, device.generation)
    config, variant = (
        get_family_resolver_registry().get(node.op_type).resolve(node, device, directives, input_contracts)
    )
    _check_views(node, config, variant)
    variant.validate_config(node, config, device)
    _check_fits_array(node, config.parallelism, device)
    ports = variant.build_ports(node, config)
    variant.validate_ports(node, ports, device)

    inputs = tuple(ExecutionInput(t.name, input_role(node, t.name)) for t in node.inputs if not t.is_parameter)
    outputs = tuple(t.name for t in node.outputs)
    return ExecutionInstance(
        node=node,
        variant=variant,
        ports=ports,
        io_route=io_route,
        port_views={name: config.io_views[name] for name in (*(item.tensor for item in inputs), *outputs)},
        config=config,
        staging=port_staging(node, variant, config, ports),
        inputs=inputs,
        outputs=outputs,
    )


def _check_fits_array(node, parallelism, device) -> None:
    """A kernel graph larger than the array placement searches can never be placed."""
    columns, rows = int(device.columns) - int(device.column_start), int(device.rows) - int(device.row_start)
    width, height = int(parallelism.cas_length), int(parallelism.cas_num)
    if width > columns or height > rows:
        raise ConfigRefused(
            f'{node.name}: its kernel graph spans {width}x{height} tiles (columns x rows), beyond '
            f'the {columns}x{rows} tiles of the {device.platform} array; split it the other way.'
        )


def port_staging(node, variant, config, ports) -> dict[str, tuple]:
    """Each bound tensor's staging, port by port, as the variant describes it: what transport reads of the kernel.

    Transport reads a port's staging as where its buffer sits in the op's frame (`offset`) and which element of
    the tensor that is (`logical_origin`), so all the ports of one tensor place the frame at one origin."""
    staging = {}
    for bindings, describe in (
        (ports.inputs, variant.describe_input_staging),
        (ports.outputs, variant.describe_output_staging),
    ):
        for tensor, binding in bindings.items():
            staging[tensor] = tuple(describe(node, config, tensor, port) for port in range(int(binding.count)))
            origins = {tuple(o - b for o, b in zip(s.logical_origin, s.offset)) for s in staging[tensor]}
            if len(origins) > 1:
                raise RuntimeError(
                    f"{node.name}: the ports of {tensor!r} place its frame at {sorted(origins)}; each port's "
                    '`logical_origin` must be its `offset` from one frame origin.'
                )
    return staging


def _check_views(node, config, variant) -> None:
    """The variant views each tensor in the order the node's `io_view` gives it, as a folded transpose left it. A
    transposed view holds each block in the tensor's order, not the view's: only a kernel that transposes the
    blocks it loads may read one, and only in blocks. Refuse rather than feed a kernel permuted data.
    """
    for direction, tensors in (('inputs', node.inputs), ('outputs', node.outputs)):
        for tensor in tensors:
            perm = view_perm(node, tensor, direction)
            view = config.io_views.get(tensor.name)
            if perm is not None and (view is None or view.perm != perm):
                raise NotImplementedError(
                    f'{node.name}: {variant.variant_id} does not read {tensor.name!r} through the io_view perm '
                    f'{perm} a folded transpose gave it.'
                )
    for name, view in config.io_views.items():
        if not view.is_transposed:
            continue
        if not variant.kernel_transposes_microtile:
            raise NotImplementedError(
                f'{node.name}: {name!r} is a transposed view, but {variant.variant_id} does not '
                'transpose the microtile on load, so the kernel would read permuted data.'
            )
        if view.microtile is None:
            raise NotImplementedError(
                f'{node.name}: {name!r} is a transposed view staged in whole rows; the DMA needs a '
                'microtiled staging to walk the grid in view order.'
            )


def _folded_views(node):
    """(value, view) for each output of a folded slice, split or concat."""
    folded = node.attributes['folded_view']
    views = [
        (name, ExecutionView(folded.kind, node.name, folded.axis, parts)) for name, parts in folded.outputs.items()
    ]
    if sorted(name for name, _ in views) != sorted(tensor.name for tensor in node.outputs):
        raise ValueError(f'{node.name}: its view names {sorted(name for name, _ in views)}, not its outputs.')
    return views


def logical_values(logical) -> dict[str, ExecutionValue]:
    """The values execution moves, read off the logical graph: its inputs, the views folding left without a
    kernel, and every other node's outputs, which the kernel graph of that node's name writes."""
    values = {name: ExecutionValue(name) for name in logical.input_tensor_names}
    for node in logical:
        if node.is_folded_view:
            values.update({name: ExecutionValue(name, view=view) for name, view in _folded_views(node)})
        else:
            values.update({t.name: ExecutionValue(t.name, producer=node.name) for t in node.outputs})
    return values


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
        ports = variant.build_ports(node, config)
        kernels.append(
            ExecutionInstance(
                node=node,
                variant=variant,
                ports=ports,
                io_route={
                    'inputs': {conversion.source: routes.get(conversion.source, 'auto')},
                    'outputs': {conversion.target: 'direct'},
                },
                port_views={conversion.source: conversion.source_view, conversion.target: view},
                config=config,
                staging=port_staging(node, variant, config, ports),
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


class Resolve(AIEPass):
    """Lower each logical node, in graph order, to the execution instance its family resolves, preceded by the
    converter kernels it asks for (`convert_inputs`): the execution graph it leaves is complete. The logical graph
    is left untouched."""

    def __init__(self):
        self.name = 'resolve'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        ctx.ir.execution.clear()
        execution = ctx.ir.execution
        # Producer before consumer (the logical order), so each node reads its inputs' published contracts.
        for node in ctx.ir.logical:
            if node.is_folded_view:
                continue
            inputs = {
                t.name: execution.tensor_contracts[t.name] for t in node.inputs if t.name in execution.tensor_contracts
            }
            inst = resolve_instance(node, ctx.device, inputs, node.schedule)
            execution.add(inst)
            execution.tensor_contracts.update(output_contracts(inst))

        # From here on transport reads these values, never the logical tensors.
        execution.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
        execution.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
        execution.values = {}
        for value in logical_values(ctx.ir.logical).values():
            execution.add_value(value)

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
        return True
