from __future__ import annotations

from typing import Optional, Tuple

from ...op_impls.common_types import PORT_KIND_BUFFER, PORT_KIND_STREAM
from .layout import port_layout
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
    pass knows how to shard the layout the endpoints use. A memory tile holds the tensor itself and a
    producer's DMA writes all its buffer holds, so padding a producer holds inside the tensor would land
    on another port's data.
    """
    for endpoint in endpoints:
        if endpoint.node is None:
            continue
        inst = execution.get(endpoint.node.name)
        if endpoint.tensor in inst.ports.outputs:
            ports = range(int(inst.ports.outputs[endpoint.tensor].count))
            stagings = [
                inst.variant.describe_output_staging(endpoint.node, inst.config, endpoint.tensor, port)
                for port in ports
            ]
        else:
            stagings = [inst.variant.describe_input_staging(endpoint.node, inst.config, endpoint.tensor, 0, None)]
        if 'transfer_bytes' in stagings[0]:
            return (
                f'{endpoint.node.name}.{endpoint.group} frames each inference as a padded transfer, which '
                'memtile staging does not implement'
            )
        if endpoint.tensor in inst.ports.outputs:
            for port, staging in enumerate(stagings):
                if port_layout(staging).pads_within():
                    return (
                        f'{endpoint.node.name}.{endpoint.group}[{port}] holds padding inside the tensor, which '
                        "its DMA would write over another port's data"
                    )
    return None


def direct_transport_failure(execution, logical_tensor: str, producer: Endpoint, consumer: Endpoint) -> str | None:
    return direct_pairing(execution, logical_tensor, producer, consumer)[0]


def direct_pairing(
    execution, logical_tensor: str, producer: Endpoint, consumer: Endpoint
) -> Tuple[Optional[str], Tuple[int, ...]]:
    """(why the leg cannot be direct or None, the producer port each consumer port reads in consumer-port order).
    Port i reads producer port i where the counts agree; with fewer producer ports, each consumer port reads the
    producer port whose buffer it holds whole, a producer port read by several being broadcast, and every producer
    port is read."""
    if producer.node is None or consumer.node is None:
        return 'direct transport requires resolved kernel endpoints', ()
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
        ), ()

    producer_ports = producer.selected_ports(producer_inst.ports.outputs[producer.tensor].count)
    consumer_ports = consumer.selected_ports(consumer_inst.ports.inputs[consumer.tensor].count)
    if len(producer_ports) > len(consumer_ports):
        return f'producer ports {producer_ports} do not match consumer ports {consumer_ports}', ()

    tc = execution.tensor_contracts.get(producer.tensor)
    if tc is not None:
        if any(int(port) < 0 or int(port) >= len(tc.port_staging) for port in producer_ports):
            return f'producer ports {producer_ports} exceed the published staging contract', ()

    def mismatch(p_port, c_port) -> Optional[str]:
        """Why one buffer cannot serve both ports: some position holds a different element, as data or padding."""
        written = producer_inst.variant.describe_output_staging(
            producer.node, producer_inst.config, producer.tensor, int(p_port)
        )
        read = consumer_inst.variant.describe_input_staging(
            consumer.node, consumer_inst.config, consumer.tensor, int(c_port), producer.node
        )
        where = f'{producer.node.name}.{producer.group}[{p_port}] -> {consumer.node.name}.{consumer.group}[{c_port}]'
        if written.get('transfer_bytes') != read.get('transfer_bytes'):
            return f'staging mismatch at {where}: the ports frame each inference as different transfers'
        source = port_layout(written).shifted(producer.offset_base).canonical()
        target = port_layout(read).shifted(consumer.offset_base).canonical()
        if source.data != target.data:
            return f'staging mismatch at {where}: the ports hold different parts of the tensor as data'
        if source != target:
            return f'staging mismatch at {where}: the buffers hold the tensor in different orders'
        return None

    one_to_one = len(producer_ports) == len(consumer_ports)
    paired = []
    for index, c_port in enumerate(consumer_ports):
        candidates = [producer_ports[index]] if one_to_one else producer_ports
        problems = [mismatch(p_port, c_port) for p_port in candidates]
        if all(problems):
            return problems[0], ()
        paired.append(int(candidates[problems.index(None)]))
    unread = sorted(set(int(p) for p in producer_ports) - set(paired))
    if unread:
        return f'producer ports {unread} are read by no consumer port', ()
    return None, tuple(paired)
