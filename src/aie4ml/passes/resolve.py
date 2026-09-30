from __future__ import annotations

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..ir.graph import (
    ExecutionInput,
    ExecutionInstance,
    ExecutionValue,
    ExecutionView,
    TensorContract,
    ViewPart,
    input_role,
)
from ..op_impls import get_family_resolver_registry
from ..op_impls.utils.io import check_io_view, resolve_io_route
from .base import AIEPass


def output_contracts(inst: ExecutionInstance) -> dict[str, TensorContract]:
    """The TensorContracts an instance publishes for its outputs, which its consumers adopt."""
    contracts = {}
    for tensor in inst.outputs:
        contract = inst.variant.output_staging_contract(inst.node, inst.config, tensor)
        if contract is None:
            continue
        ports = range(int(inst.variant.output_port_count(inst.node, inst.config)))
        contracts[tensor] = TensorContract(
            contract=contract,
            port_staging=tuple(
                inst.variant.describe_output_staging(inst.node, inst.config, tensor, port) for port in ports
            ),
            inner_shards=inst.variant.output_inner_shards(inst.node, inst.config, tensor),
            view=inst.port_views[tensor],
        )
    return contracts


def resolve_instance(node, device, input_contracts, parallelism=None) -> ExecutionInstance:
    """The kernel graph that implements `node`, given the contracts its inputs arrive in and, when given, the
    parallelism to resolve it under instead of its own directive. Pure: nothing is registered."""
    directives = dict(node.directives or {})
    if parallelism is not None:
        directives['parallelism'] = dict(parallelism)
    directives['io_route'] = resolve_io_route(node)  # user intents

    check_io_view(node, device.generation)
    config, variant = (
        get_family_resolver_registry().get(node.op_type).resolve(node, device, directives, input_contracts)
    )
    _check_transposed_views(node, config, variant)
    variant.validate_config(node, config, device)
    _check_fits_array(node, variant.footprint(node, config), device)
    ports = variant.build_ports(node, config)
    variant.validate_ports(node, ports, device)

    inputs = tuple(ExecutionInput(t.name, input_role(node, t.name)) for t in node.inputs if not t.is_parameter)
    outputs = tuple(t.name for t in node.outputs)
    _check_port_frames(node, config, variant, ports)
    return ExecutionInstance(
        node=node,
        variant=variant,
        ports=ports,
        io_route=dict(config.io_route),
        port_views={name: config.io_views[name] for name in (*(item.tensor for item in inputs), *outputs)},
        config=config,
        graph_header=variant.graph_header,
        graph_name=variant.graph_name,
        param_template=variant.param_template,
        inputs=inputs,
        outputs=outputs,
    )


def _check_fits_array(node, footprint, device) -> None:
    """A kernel graph larger than the array placement searches can never be placed."""
    columns, rows = int(device.columns) - int(device.column_start), int(device.rows) - int(device.row_start)
    if footprint.width > columns or footprint.height > rows:
        raise ConfigRefused(
            f'{node.name}: its kernel graph spans {footprint.width}x{footprint.height} tiles (columns x rows), beyond '
            f'the {columns}x{rows} tiles of the {device.platform} array; split it the other way.'
        )


def _check_port_frames(node, config, variant, ports) -> None:
    """Transport reads a port's staging as where its buffer sits in the op's frame (`offset`) and which element of
    the tensor that is (`logical_origin`), so all the ports of one tensor place the frame at one origin."""
    for bindings, describe in (
        (ports.inputs, lambda tensor, port: variant.describe_input_staging(node, config, tensor, port, None)),
        (ports.outputs, lambda tensor, port: variant.describe_output_staging(node, config, tensor, port)),
    ):
        for tensor, binding in bindings.items():
            origins = set()
            for port in range(int(binding.count)):
                staging = describe(tensor, port)
                origins.add(tuple(int(o) - int(b) for o, b in zip(staging['logical_origin'], staging['offset'])))
            if len(origins) > 1:
                raise RuntimeError(
                    f"{node.name}: the ports of {tensor!r} place its frame at {sorted(origins)}; each port's "
                    '`logical_origin` must be its `offset` from one frame origin.'
                )


def _check_transposed_views(node, config, variant) -> None:
    """A folded transpose needs both halves: the DMA walks the microtile grid in view order and
    the kernel transposes each block on load. Refuse rather than feed a kernel permuted data.
    """

    for name, view in (getattr(config, 'io_views', None) or {}).items():
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
    """(value, view) for each output of a folded slice, split or concat, checked against the trait's schema."""

    def trait(name, keys, part_keys):
        data = node.traits[name].data
        if set(data) != keys or any(set(item) != part_keys for item in data['slices']):
            raise ValueError(f'{node.name}: malformed {name} {data}.')
        return data

    if 'concat_view' in node.traits:
        data = trait('concat_view', {'kind', 'axis', 'output', 'slices'}, {'input', 'start', 'extent'})
        parts = tuple(ViewPart(str(s['input']), int(s['start']), int(s['extent'])) for s in data['slices'])
        views = [(str(data['output']), ExecutionView('concat', node.name, int(data['axis']), parts))]
    else:
        data = trait('slice_view', {'kind', 'axis', 'source', 'slices'}, {'output', 'start', 'extent'})
        views = []
        for s in data['slices']:
            part = ViewPart(str(data['source']), int(s['start']), int(s['extent']))
            views.append((str(s['output']), ExecutionView(node.op_type, node.name, int(data['axis']), (part,))))
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


class Resolve(AIEPass):
    """Resolve logical nodes into family-owned execution instances."""

    def __init__(self):
        self.name = 'resolve'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        ctx.ir.logical.verify()
        ctx.ir.execution.clear()
        execution = ctx.ir.execution
        # Producer before consumer (the logical order), so each node reads its inputs' published contracts.
        for node in ctx.ir.logical:
            if node.is_folded_view:
                continue
            inputs = {
                t.name: execution.tensor_contracts[t.name] for t in node.inputs if t.name in execution.tensor_contracts
            }
            # the design search's choice where it made one; otherwise the node's own directive
            inst = resolve_instance(node, ctx.device, inputs, ctx.ir.optimizer.get('parallelism', {}).get(node.name))
            execution.add(inst)
            execution.tensor_contracts.update(output_contracts(inst))

        # From here on transport reads these values, never the logical tensors.
        execution.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
        execution.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
        execution.values = {}
        for value in logical_values(ctx.ir.logical).values():
            execution.add_value(value)
        return True
