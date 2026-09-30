from __future__ import annotations

import copy

import numpy as np

from ...errors import ConfigRefused
from ...ir import get_backend_context
from ..base import AIEPass
from .boundary import graph_input_port_descs, graph_input_writer_port_descs
from .layout import port_layout
from .legality import uses_stream
from .model import GraphInputSpec, TransportUnit


class LegalizeMemtilePortLimits(AIEPass):
    """Number the graph-input ports, and split each memory-tile leg into units within a memory tile's port limits:
    its writers in port order, each reader in the unit whose writers hold all the data it reads."""

    def __init__(self):
        self.name = 'legalize_memtile_port_limits'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        state = ctx.ir.physical.plan['_memory_plan_state']
        max_in = int(ctx.device.max_mem_in_ports)
        max_out = int(ctx.device.max_mem_out_ports)
        rewritten = []
        changed = False
        next_graph_input_port = 0

        for entry in state['entries']:
            self._validate_classified_entry(entry)
            producer_ports = self._producer_port_ids(entry, ctx)
            consumer_ports = self._consumer_port_ids(entry, ctx)
            if entry.producer.node is None:
                descriptors = graph_input_port_descs(entry, ctx, next_graph_input_port)
                next_graph_input_port += len(descriptors)
                stream = uses_stream(ctx.ir.execution, entry.endpoints())
                entry.graph_input = GraphInputSpec(
                    descriptors, graph_input_writer_port_descs(descriptors, stream=stream)
                )
                producer_ports = tuple(descriptors)

            if entry.decision.realization == 'direct':
                entry.unit = TransportUnit(producer_ports, consumer_ports)
                rewritten.append(entry)
                continue
            if max_in <= 0 or max_out <= 0:
                raise RuntimeError(
                    f'{entry.logical_tensor}: memory-tile transport selected on a device without usable '
                    'memory-tile ports.'
                )

            reader_count = len(consumer_ports) if entry.consumers else len(producer_ports)
            units = max(-(-len(producer_ports) // max_in), -(-reader_count // max_out))
            size = -(-len(producer_ports) // units)
            writer_chunks = [producer_ports[start : start + size] for start in range(0, len(producer_ports), size)]
            reader_chunks = self._reader_chunks(entry, ctx, writer_chunks, consumer_ports)
            count = len(writer_chunks)
            changed = changed or count > 1
            for index, (writers, readers) in enumerate(zip(writer_chunks, reader_chunks)):
                legal = copy.copy(entry) if count > 1 else entry
                legal.unit = TransportUnit(tuple(writers), tuple(readers), index=index, count=count)
                self._validate_limits(legal, max_in, max_out)
                rewritten.append(legal)

        state['entries'] = rewritten
        return changed

    @staticmethod
    def _validate_classified_entry(entry) -> None:
        if entry.decision is None:
            raise RuntimeError(f'{entry.logical_tensor}: missing transport decision; run classification first.')
        if len(entry.consumers) > 1 or (entry.consumers and entry.graph_output):
            raise RuntimeError(f'{entry.logical_tensor}: memtile legalization requires one independent transport leg.')

    def _reader_chunks(self, entry, ctx, writer_chunks, consumer_ports):
        """The consumer ports each unit serves: a reader goes to the first unit whose writers hold all its data. A
        graph output's host readers follow their producer ports, so they need no assignment."""
        if not entry.consumers:
            return [() for _ in writer_chunks]
        if len(writer_chunks) == 1:
            return [tuple(consumer_ports)]
        written = [[self._writer_data(entry, ctx, port) for port in chunk] for chunk in writer_chunks]
        consumer = entry.single_consumer()
        inst = ctx.ir.execution.get(consumer.node.name)
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

    @staticmethod
    def _writer_data(entry, ctx, port):
        """The tensor coordinates one writer holds as data: a graph-input port's, the part its consumer port reads."""
        if entry.producer.node is None:
            return port_layout(entry.graph_input.port_descriptors[int(port)]).data
        producer = entry.producer
        inst = ctx.ir.execution.get(producer.node.name)
        staging = inst.variant.describe_output_staging(producer.node, inst.config, producer.tensor, port)
        return port_layout(staging).shifted(producer.offset_base).data

    @staticmethod
    def _validate_limits(entry, max_in: int, max_out: int) -> None:
        if len(entry.unit.producer_ports) > max_in:
            raise ConfigRefused(f'{entry.logical_tensor}: shard exceeds memtile in-port limit {max_in}.')
        output_count = len(entry.unit.consumer_ports) if entry.consumers else len(entry.unit.producer_ports)
        if output_count > max_out:
            raise ConfigRefused(f'{entry.logical_tensor}: shard exceeds memtile out-port limit {max_out}.')

    @staticmethod
    def _producer_port_ids(entry, ctx):
        if entry.producer.node is None:
            return tuple(range(entry.producer_port_count))
        inst = ctx.ir.execution.get(entry.producer.node.name)
        return entry.producer.selected_ports(inst.ports.outputs[entry.producer.tensor].count)

    @staticmethod
    def _consumer_port_ids(entry, ctx):
        if not entry.consumers:
            return ()
        consumer = entry.single_consumer()
        inst = ctx.ir.execution.get(consumer.node.name)
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
