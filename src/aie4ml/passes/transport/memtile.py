from __future__ import annotations

import copy

import numpy as np

from ...errors import ConfigRefused
from ...ir import get_backend_context
from ..base import AIEPass
from .boundary import graph_input_port_descs, graph_input_writer_port_descs
from .layout import port_layout
from .legality import uses_stream
from .model import EdgeEntry, GraphInputSpec, TransportDecision, TransportUnit


class LegalizeMemtilePortLimits(AIEPass):
    """Number the graph-input ports, and split each memory-tile leg into units within a memory tile's port limits
    (`memtile_units`)."""

    def __init__(self):
        self.name = 'legalize_memtile_port_limits'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        execution = ctx.ir.execution
        max_in = int(ctx.device.max_mem_in_ports)
        max_out = int(ctx.device.max_mem_out_ports)
        rewritten = []
        changed = False
        next_graph_input_port = 0

        for entry in state['entries']:
            self._validate_classified_entry(entry)
            if entry.producer.node is None:
                descriptors = graph_input_port_descs(entry, execution, next_graph_input_port)
                next_graph_input_port += len(descriptors)
                stream = uses_stream(execution, entry.endpoints())
                entry.graph_input = GraphInputSpec(
                    descriptors, graph_input_writer_port_descs(descriptors, stream=stream)
                )

            if entry.decision.realization == 'direct':
                if entry.unit is None:  # a graph boundary's: kernel-to-kernel legs are paired by classification
                    entry.unit = TransportUnit(_writer_ports(entry, execution), _consumer_ports(entry, execution))
                rewritten.append(entry)
                continue
            if max_in <= 0 or max_out <= 0:
                raise RuntimeError(
                    f'{entry.logical_tensor}: memory-tile transport selected on a device without usable '
                    'memory-tile ports.'
                )
            units = memtile_units(entry, execution, max_in, max_out)
            changed = changed or len(units) > 1
            rewritten.extend(units)

        state['entries'] = rewritten
        return changed

    @staticmethod
    def _validate_classified_entry(entry) -> None:
        if entry.decision is None:
            raise RuntimeError(f'{entry.logical_tensor}: missing transport decision; run classification first.')
        if len(entry.consumers) > 1 or (entry.consumers and entry.graph_output):
            raise RuntimeError(f'{entry.logical_tensor}: memtile legalization requires one independent transport leg.')


def memtile_units(entry, execution, max_in: int, max_out: int) -> list:
    """The memory-tile leg `entry` split into units within a memory tile's port limits, each a copy carrying its
    TransportUnit where it splits: its writers in port order, each reader in the unit whose writers hold all the data
    it reads. Refused where a unit would still pass a limit, or a reader needs data of two units (a relay is not
    implemented). Reads only `execution` and the entry (a graph input's numbered `graph_input`), so the design search
    judges each memory-tile leg by this rule too."""
    producer_ports = _writer_ports(entry, execution)
    consumer_ports = _consumer_ports(entry, execution)
    reader_count = len(consumer_ports) if entry.consumers else len(producer_ports)
    units = max(-(-len(producer_ports) // max_in), -(-reader_count // max_out))
    size = -(-len(producer_ports) // units)
    writer_chunks = [producer_ports[start : start + size] for start in range(0, len(producer_ports), size)]
    reader_chunks = _reader_chunks(entry, execution, writer_chunks, consumer_ports)
    count = len(writer_chunks)
    legal = []
    for index, (writers, readers) in enumerate(zip(writer_chunks, reader_chunks)):
        unit = copy.copy(entry) if count > 1 else entry
        unit.unit = TransportUnit(tuple(writers), tuple(readers), index=index, count=count)
        _validate_limits(unit, max_in, max_out)
        legal.append(unit)
    return legal


def check_memtile_leg(leg, execution, device) -> None:
    """Refuse a leg the classifier sends through a memory tile where `memtile_units` cannot split it within the
    device's port limits, as the pipeline would after placement."""
    if leg.producer.node is None:
        inst = execution.get(leg.consumer.node.name)
        writers = len(leg.consumer.selected_ports(inst.ports.inputs[leg.consumer.tensor].count))
    else:
        writers = execution.get(leg.producer.node.name).ports.outputs[leg.producer.tensor].count
    entry = EdgeEntry(
        leg.logical_tensor,
        leg.producer,
        writers,
        consumers=[leg] if leg.consumer is not None else [],
        graph_output=leg.consumer is None,
        decision=TransportDecision('memtile', None),
    )
    if leg.producer.node is None:
        descriptors = graph_input_port_descs(entry, execution, 0)
        entry.graph_input = GraphInputSpec(descriptors, graph_input_writer_port_descs(descriptors, stream=False))
    memtile_units(entry, execution, int(device.max_mem_in_ports), int(device.max_mem_out_ports))


def _reader_chunks(entry, execution, writer_chunks, consumer_ports):
    """The consumer ports each unit serves: a reader goes to the first unit whose writers hold all its data. A
    graph output's host readers follow their producer ports, so they need no assignment."""
    if not entry.consumers:
        return [() for _ in writer_chunks]
    if len(writer_chunks) == 1:
        return [tuple(consumer_ports)]
    written = [[_writer_data(entry, execution, port) for port in chunk] for chunk in writer_chunks]
    consumer = entry.single_consumer()
    inst = execution.get(consumer.node.name)
    chunks = [[] for _ in writer_chunks]
    for port in consumer_ports:
        staging = inst.variant.describe_input_staging(
            consumer.node, inst.config, consumer.tensor, port, entry.producer.node
        )
        read = port_layout(staging).shifted(consumer.offset_base).data
        unit = next((index for index, windows in enumerate(written) if _covers(windows, read)), None)
        if unit is None:
            raise ConfigRefused(
                f'{entry.logical_tensor}: {consumer.node.name} port {port} reads data its producer writes into '
                'different memory-tile units; a relay is not implemented.'
            )
        chunks[unit].append(port)
    return [tuple(chunk) for chunk in chunks]


def _writer_data(entry, execution, port):
    """The tensor coordinates one writer holds as data: a graph-input port's, the part its consumer port reads."""
    if entry.producer.node is None:
        return port_layout(entry.graph_input.port_descriptors[int(port)]).data
    producer = entry.producer
    inst = execution.get(producer.node.name)
    staging = inst.variant.describe_output_staging(producer.node, inst.config, producer.tensor, port)
    return port_layout(staging).shifted(producer.offset_base).data


def _validate_limits(entry, max_in: int, max_out: int) -> None:
    if len(entry.unit.producer_ports) > max_in:
        raise ConfigRefused(f'{entry.logical_tensor}: shard exceeds memtile in-port limit {max_in}.')
    output_count = len(entry.unit.consumer_ports) if entry.consumers else len(entry.unit.producer_ports)
    if output_count > max_out:
        raise ConfigRefused(f'{entry.logical_tensor}: shard exceeds memtile out-port limit {max_out}.')


def _writer_ports(entry, execution):
    """The ports writing the leg: a graph input's numbered ones, else its producer's."""
    if entry.producer.node is None:
        return tuple(entry.graph_input.port_descriptors)
    inst = execution.get(entry.producer.node.name)
    return entry.producer.selected_ports(inst.ports.outputs[entry.producer.tensor].count)


def _consumer_ports(entry, execution):
    if not entry.consumers:
        return ()
    consumer = entry.single_consumer()
    inst = execution.get(consumer.node.name)
    return consumer.selected_ports(inst.ports.inputs[consumer.tensor].count)


def _covers(windows, window) -> bool:
    """Whether the boxes `windows` together hold every coordinate of the box `window`."""
    held = np.zeros([max(0, hi - lo) for lo, hi in window], dtype=bool)
    for box in windows:
        cut = []
        for (lo, hi), (first, end) in zip(box, window):
            start = min(max(lo, first), end) - first
            cut.append(slice(start, max(start, min(hi, end) - first)))
        held[tuple(cut)] = True
    return bool(held.all())
