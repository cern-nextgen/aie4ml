from __future__ import annotations

from ...errors import ConfigRefused
from ...ir import get_backend_context
from ...ir.graph import ROUTE_MODES
from ..base import AIEPass
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
        realization = 'direct' if route == 'direct' or (route == 'auto' and not can_memtile) else 'memtile'
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
