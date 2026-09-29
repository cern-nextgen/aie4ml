from __future__ import annotations

from ...errors import ConfigRefused
from ...ir import get_backend_context
from ...ir.graph import ROUTE_MODES
from ...op_impls.utils.tensor_view import staging_tile_shape
from ..base import AIEPass
from .boundary import direct_boundary_access
from .descriptors import rebase_descriptor_offset
from .legality import direct_transport_failure, memtile_staging_failure, uses_stream
from .model import Connection, TransportDecision


class ClassifyTransportEntries(AIEPass):
    """Choose direct or memtile realization without performing memtile legalization."""

    def __init__(self):
        self.name = 'classify_transport_entries'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        entries = ctx.ir.physical.plan['_memory_plan_state']['entries']
        changed = False
        for entry in entries:
            self._validate_entry(entry)
            leg = entry.consumers[0] if entry.consumers else Connection(entry.logical_tensor, entry.producer, None)
            decision = classify_connection(leg, ctx.ir.execution, ctx.device.has_memtile)
            changed = changed or entry.decision != decision
            entry.decision = decision
        return changed

    @staticmethod
    def _validate_entry(entry) -> None:
        if len(entry.consumers) > 1:
            raise RuntimeError(
                f'{entry.logical_tensor}: classification requires fanout to be split into single-consumer entries.'
            )
        if entry.consumers and entry.graph_output:
            raise RuntimeError(
                f'{entry.logical_tensor}: classification requires graph-output and consumer legs to be separate.'
            )
        if entry.producer.node is not None and not entry.graph_output and not entry.consumers:
            raise RuntimeError(f'{entry.logical_tensor}: internal transport entry has no consumer.')


def classify_connection(leg: Connection, execution, has_memtile) -> TransportDecision:
    """How one transport leg is realised; `leg.consumer` is None for a graph output. Reads only the resolved
    instances and contracts in `execution`, so the parallelism search decides every leg by this rule too."""
    tensor, producer, consumer = leg.logical_tensor, leg.producer, leg.consumer
    shards_failure = _inner_shards_failure(leg, execution)
    if shards_failure is not None:
        raise ConfigRefused(
            f'{tensor}: {producer.node.name} stores its inner axis shard by shard, but {shards_failure}.'
        )
    route = _route_policy(leg, execution)
    is_boundary = producer.node is None or consumer is None
    has_memtile = bool(has_memtile)

    restage_failure = memtile_staging_failure(execution, leg.endpoints())
    if route == 'memtile' and restage_failure is not None:
        raise ConfigRefused(f'{tensor}: io_route=memtile needs re-staging, but {restage_failure}.')
    can_memtile = has_memtile and restage_failure is None

    if uses_stream(execution, leg.endpoints()):
        # A stream port has no buffer for a memory tile to stage, so the leg is
        # point-to-point (or PLIO) and must already agree on its element order.
        if route == 'memtile':
            raise ConfigRefused(f'{tensor}: io_route=memtile requested on a stream port.')
        if not is_boundary:
            failure = _direct_failure(leg, execution)
            if failure is not None:
                raise ConfigRefused(f'{tensor}: point-to-point ports cannot connect directly: {failure}.')
        return TransportDecision('direct', True)

    if is_boundary:
        if route == 'memtile' and not has_memtile:
            raise ConfigRefused(f'{tensor}: io_route=memtile requested on a device without memory tiles.')
        if route == 'direct' or not can_memtile:
            realization = 'direct'
        elif route == 'memtile':
            realization = 'memtile'
        else:  # a memory tile adds a stage and streams the padding it zero-fills; take it only where it must
            realization = 'direct' if _direct_boundary_failure(leg, execution) is None else 'memtile'
        return TransportDecision(realization, True if realization == 'direct' else None)

    direct_failure = _direct_failure(leg, execution)
    staging_compatible = direct_failure is None
    if route == 'direct':
        if not staging_compatible:
            raise ConfigRefused(
                f'{tensor}: io_route=direct requested but point-to-point transport '
                f'is not staging-compatible: {direct_failure}.'
            )
        realization = 'direct'
    elif route == 'memtile':
        if not has_memtile:
            raise ConfigRefused(f'{tensor}: io_route=memtile requested on a device without memory tiles.')
        realization = 'memtile'
    else:
        if staging_compatible:
            realization = 'direct'
        elif not can_memtile:
            raise ConfigRefused(
                f'{tensor}: AIE1 cannot directly connect this transport: {direct_failure}; '
                'relay/relayout is not implemented.'
            )
        else:
            realization = 'memtile'
    return TransportDecision(realization, staging_compatible)


def _direct_boundary_failure(leg: Connection, execution) -> str | None:
    """Why a PLIO cannot feed or drain this graph-boundary leg's kernel ports through the tile's own DMA, or None.
    That DMA writes only the logical elements: padding rows may hold anything, since no row reads another, but an
    input padded along its inner axis is summed across it, so only a memory tile, which zero-fills, may feed it."""
    output = leg.consumer is None
    endpoint = leg.producer if output else leg.consumer
    inst = execution.get(endpoint.node.name)
    if output:
        binding, element = inst.ports.outputs[endpoint.tensor], inst.variant.output_precision(inst.config)
    else:
        role = inst.input(endpoint.tensor).role
        binding, element = inst.ports.inputs[endpoint.tensor], inst.variant.input_precision(inst.config, role)
    for port in endpoint.selected_ports(binding.count):
        if output:
            staging = inst.variant.describe_output_staging(endpoint.node, inst.config, endpoint.tensor, port, None)
        else:
            staging = inst.variant.describe_input_staging(endpoint.node, inst.config, endpoint.tensor, port, None, None)
            rebase_descriptor_offset(staging, endpoint.offset_base)
            inner = int(staging['inner_dimension'])
            if int(staging['io_tiling_dimension'][inner]) < staging_tile_shape(staging)[inner]:
                return f'{endpoint.node.name}.{endpoint.group} pads its inner axis'
        try:
            direct_boundary_access(staging, staging['io_tiling_dimension'], element_bits=element.width, output=output)
        except ConfigRefused as refusal:
            return str(refusal)
    return None


def _inner_shards_failure(leg: Connection, execution) -> str | None:
    """Why this leg would read a tensor stored shard by shard (TensorContract.inner_shards) as if it were in logical
    order, or None: only a consumer that adopted that order may read it, and only whole."""
    contract = execution.tensor_contracts.get(leg.producer.tensor)
    shards = contract.inner_shards if contract is not None else None
    if shards is None:
        return None
    if leg.consumer is None:
        return 'a graph output leaves in logical order'
    if leg.consumer.tensor != leg.producer.tensor:
        return f'{leg.consumer.node.name} reads it through the view {leg.consumer.tensor!r}'
    inst = execution.get(leg.consumer.node.name)
    if inst.variant.input_inner_shards(inst.node, inst.config, leg.consumer.tensor) != shards:
        return f'{leg.consumer.node.name} reads it in logical order'
    return None


def _direct_failure(leg: Connection, execution) -> str | None:
    consumer = leg.consumer
    if execution.get(consumer.node.name).port_views[consumer.tensor].perm is not None:
        return f'consumer {consumer.node.name}.{consumer.group} applies an input permutation'
    return direct_transport_failure(execution, leg.logical_tensor, leg.producer, consumer)


def _route_policy(leg: Connection, execution) -> str:
    modes = set()
    if leg.producer.node is not None:
        producer_mode = execution.get(leg.producer.node.name).io_route.get('outputs', {}).get(leg.producer.tensor)
        if producer_mode:
            modes.add(str(producer_mode))

    if leg.consumer is not None:
        consumer_mode = execution.get(leg.consumer.node.name).io_route.get('inputs', {}).get(leg.consumer.tensor)
        if consumer_mode:
            modes.add(str(consumer_mode))

    bad = [mode for mode in modes if mode not in ROUTE_MODES]
    if bad:
        raise ValueError(f'{leg.logical_tensor}: unsupported io_route mode(s) {bad}.')
    if 'memtile' in modes:
        return 'memtile'
    if modes == {'direct'}:
        return 'direct'
    return 'auto'
