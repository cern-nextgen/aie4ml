from __future__ import annotations

import copy

from ...op_impls.common_types import PORT_KIND_BUFFER, PORT_KIND_STREAM
from ...op_impls.utils import STORAGE_LAYOUT_INNER_BLOCKED
from ...op_impls.utils.io import normalized_staging
from .descriptors import localize_descriptor
from .model import Endpoint


def endpoint_port_kind(execution, endpoint: Endpoint) -> str:
    """The ADF port kind behind a kernel endpoint; a graph boundary (PLIO) takes either kind."""
    if endpoint.node is None:
        return PORT_KIND_BUFFER
    inst = execution.get(endpoint.node.name)
    if inst is None:
        raise RuntimeError(f'{endpoint.tensor}: endpoint {endpoint.node.name!r} has no resolved execution instance.')
    bindings = inst.ports.outputs if endpoint.tensor in inst.ports.outputs else inst.ports.inputs
    return bindings[endpoint.tensor].kind


def uses_stream(execution, endpoints) -> bool:
    """Whether any kernel endpoint of a transport leg is a stream port."""
    return any(endpoint_port_kind(execution, endpoint) == PORT_KIND_STREAM for endpoint in endpoints)


def memtile_staging_failure(execution, endpoints) -> str | None:
    """Why a memory tile cannot re-stage this entry, or None when it can.

    Storage encoding and routing are separate concerns: this answers only whether the memtile
    pass knows how to shard the layout the endpoints use.
    """
    for endpoint in endpoints:
        if endpoint.node is None:
            continue
        inst = execution.get(endpoint.node.name)
        if endpoint.tensor in inst.ports.outputs:
            desc = inst.variant.describe_output_staging(endpoint.node, inst.config, endpoint.tensor, 0, None)
        else:
            desc = inst.variant.describe_input_staging(endpoint.node, inst.config, endpoint.tensor, 0, None, None)
        if desc.get('storage_layout') == STORAGE_LAYOUT_INNER_BLOCKED:
            return (
                f'{endpoint.node.name}.{endpoint.group} stages an inner-blocked buffer, which memtile '
                'sharding does not implement'
            )
        if 'transfer_bytes' in desc:
            return (
                f'{endpoint.node.name}.{endpoint.group} frames each inference as a padded transfer, which '
                'memtile staging does not implement'
            )
    return None


def direct_transport_failure(
    execution,
    logical_tensor: str,
    producer: Endpoint,
    consumer: Endpoint,
) -> str | None:
    if producer.node is None or consumer.node is None:
        return 'direct transport requires resolved kernel endpoints'
    producer_inst = execution.get(producer.node.name)
    consumer_inst = execution.get(consumer.node.name)
    if producer_inst is None or consumer_inst is None:
        raise RuntimeError(f'{logical_tensor}: direct transport legality requires resolved execution instances.')

    producer_kind = producer_inst.ports.outputs[producer.tensor].kind
    consumer_kind = consumer_inst.ports.inputs[consumer.tensor].kind
    if producer_kind != consumer_kind:
        # ADF could bridge the two through the tile DMA; refuse until a kernel needs that bridge.
        return (
            f'producer {producer.node.name}.{producer.group} is a {producer_kind} port but consumer '
            f'{consumer.node.name}.{consumer.group} is a {consumer_kind} port'
        )

    producer_ports = producer.selected_ports(producer_inst.ports.outputs[producer.tensor].count)
    consumer_ports = consumer.selected_ports(consumer_inst.ports.inputs[consumer.tensor].count)
    if len(producer_ports) != len(consumer_ports):
        return f'producer ports {producer_ports} do not match consumer ports {consumer_ports}'

    tc = execution.tensor_contracts.get(producer.tensor)
    if tc is not None:
        if any(int(port) < 0 or int(port) >= len(tc.port_staging) for port in producer_ports):
            return f'producer ports {producer_ports} exceed the published staging contract'

    for p_port, c_port in zip(producer_ports, consumer_ports):
        src_desc = producer_inst.variant.describe_output_staging(
            producer.node, producer_inst.config, producer.tensor, int(p_port), None
        )
        if producer.offset_base:
            src_desc = copy.deepcopy(src_desc)
            localize_descriptor(src_desc, producer.offset_base, producer.buffer_dimension)
        dst_desc = consumer_inst.variant.describe_input_staging(
            consumer.node,
            consumer_inst.config,
            consumer.tensor,
            int(c_port),
            None,
            producer.node,
        )
        dst_desc = copy.deepcopy(dst_desc)
        localize_descriptor(dst_desc, consumer.offset_base, src_desc['buffer_dimension'])
        src_staging = normalized_staging(src_desc)
        dst_staging = normalized_staging(dst_desc)
        if src_staging != dst_staging:
            keys = sorted(
                key for key in set(src_staging) | set(dst_staging) if src_staging.get(key) != dst_staging.get(key)
            )
            return (
                f'staging mismatch at {producer.node.name}.{producer.group}[{p_port}] -> '
                f'{consumer.node.name}.{consumer.group}[{c_port}] ({", ".join(keys)})'
            )
    return None
