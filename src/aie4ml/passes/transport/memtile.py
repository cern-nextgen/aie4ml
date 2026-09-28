from __future__ import annotations

import copy

from ...errors import ConfigRefused
from ...ir import get_backend_context
from ..base import AIEPass
from .boundary import (
    graph_input_full_descriptor,
    graph_input_port_descs,
    graph_input_unit_box,
    graph_input_writer_port_descs,
)
from .descriptors import localize_descriptor, rebase_descriptor_offset
from .legality import uses_stream
from .model import GraphInputSpec, TransportUnit


class LegalizeMemtilePortLimits(AIEPass):
    """Assign physical ports and split memtile transports into legal one-stage shards."""

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

            if entry.decision.realization == 'direct':
                if entry.producer.node is None:
                    port_base = next_graph_input_port
                    next_graph_input_port += len(producer_ports)
                    descriptors = graph_input_port_descs(entry, ctx, port_base)
                    entry.graph_input = GraphInputSpec(
                        descriptors,
                        graph_input_writer_port_descs(
                            descriptors, stream=uses_stream(ctx.ir.execution, entry.endpoints())
                        ),
                    )
                    producer_ports = tuple(port_base + index for index in range(len(producer_ports)))
                entry.unit = TransportUnit(producer_ports, consumer_ports)
                rewritten.append(entry)
                continue

            if max_in <= 0 or max_out <= 0:
                raise RuntimeError(
                    f'{entry.logical_tensor}: memory-tile transport selected on a device without usable '
                    'memory-tile ports.'
                )

            p = len(producer_ports)
            c = len(consumer_ports) if entry.consumers else p
            units = max((p + max_in - 1) // max_in, (c + max_out - 1) // max_out)
            port_base = next_graph_input_port if entry.producer.node is None else 0
            if entry.producer.node is None:
                next_graph_input_port += p

            if entry.producer.node is None:
                descriptors = graph_input_port_descs(entry, ctx, port_base)
                graph_input = GraphInputSpec(descriptors, graph_input_writer_port_descs(descriptors))
            else:
                graph_input = None

            if units == 1:
                entry.graph_input = graph_input
                entry.unit = self._single_unit(entry, ctx, producer_ports, consumer_ports, port_base)
                self._validate_limits(entry, max_in, max_out)
                rewritten.append(entry)
                continue

            changed = True
            if entry.producer.ports is not None:
                raise ConfigRefused(f'{entry.logical_tensor}: sharded slice transport is not implemented.')
            self._validate_one_stage_ratio(entry, p, c)

            p_chunks = self._split_ports_serial(p, units)
            if entry.producer.node is None:
                p_chunks = [[port_base + port for port in chunk] for chunk in p_chunks]
            else:
                p_chunks = [[producer_ports[port] for port in chunk] for chunk in p_chunks]

            c_chunks = self._consumer_chunks(entry, ctx, p, c, p_chunks, consumer_ports, max_out)
            if graph_input is None:
                shard_dim, port_stride, _ = self._shard_params(entry, ctx)
                unit_sizes = [len(chunk) * int(port_stride) for chunk in p_chunks]
                unit_bases = []
                base = 0
                for size in unit_sizes:
                    unit_bases.append(base)
                    base += int(size)
                unit_boxes = None
            else:
                unit_boxes = [graph_input_unit_box(graph_input.port_descriptors, chunk) for chunk in p_chunks]

            for index, (p_ports, c_ports) in enumerate(zip(p_chunks, c_chunks)):
                legal = copy.copy(entry)
                legal.graph_input = graph_input
                if graph_input is not None:
                    offset_base, buffer_dimension = unit_boxes[index]
                    legal.unit = TransportUnit(
                        tuple(p_ports),
                        tuple(c_ports),
                        producer_tensor_port_base=port_base,
                        offset_base=tuple(offset_base),
                        buffer_dimension=tuple(buffer_dimension),
                        index=index,
                        count=units,
                    )
                else:
                    legal.unit = TransportUnit(
                        tuple(p_ports),
                        tuple(c_ports),
                        dimension=shard_dim,
                        port_stride=port_stride,
                        dimension_base=unit_bases[index],
                        dimension_size=unit_sizes[index],
                        index=index,
                        count=units,
                    )
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

    def _single_unit(self, entry, ctx, producer_ports, consumer_ports, port_base: int) -> TransportUnit:
        if entry.producer.node is None:
            full = graph_input_full_descriptor(entry, ctx)
            return TransportUnit(
                tuple(port_base + index for index in range(len(producer_ports))),
                tuple(consumer_ports),
                producer_tensor_port_base=port_base,
                offset_base=tuple(0 for _ in full['offset']),
                buffer_dimension=tuple(full['buffer_dimension']),
            )
        shard_dim, port_stride, full_dim = self._shard_params(entry, ctx)
        return TransportUnit(
            tuple(producer_ports),
            tuple(consumer_ports),
            dimension=shard_dim,
            port_stride=port_stride,
            dimension_size=full_dim,
        )

    @staticmethod
    def _validate_one_stage_ratio(entry, producer_count: int, consumer_count: int) -> None:
        if entry.consumers and consumer_count < producer_count:
            raise ConfigRefused(
                f'{entry.logical_tensor}: shard transport requires relay; one-stage sharding cannot contract '
                f'producer_ports={producer_count} to consumer_ports={consumer_count}.'
            )
        if entry.consumers and consumer_count % producer_count != 0:
            raise ConfigRefused(
                f'{entry.logical_tensor}: shard transport requires relay; one-stage sharding cannot regroup '
                f'producer_ports={producer_count} to consumer_ports={consumer_count}.'
            )

    def _consumer_chunks(self, entry, ctx, p, c, p_chunks, consumer_ports, max_out):
        if not entry.consumers:
            return [[] for _ in p_chunks]
        c_per_p = c // p
        per_shard_out = max((len(chunk) * c_per_p for chunk in p_chunks), default=0)
        if entry.producer.node is not None and per_shard_out > max_out:
            raise ConfigRefused(
                f'{entry.logical_tensor}: shard transport requires relay; one-stage sharding cannot realize '
                f'producer_ports={p} -> consumer_ports={c} under memtile out-port limit {max_out}.'
            )
        if entry.producer.node is not None:
            return self._consumer_chunks_for_internal_shard(entry, ctx, p_chunks, consumer_ports, c_per_p)

        chunks = []
        start = 0
        for p_ports in p_chunks:
            size = len(p_ports) * c_per_p
            chunks.append(list(consumer_ports[start : start + size]))
            start += size
        if start != c:
            raise RuntimeError(f'{entry.logical_tensor}: consumer shard assignment mismatch ({start} != {c}).')
        return chunks

    @staticmethod
    def _split_ports_serial(count: int, units: int):
        chunk_size = (count + units - 1) // units
        return [list(range(start, min(start + chunk_size, count))) for start in range(0, count, chunk_size)]

    def _consumer_chunks_for_internal_shard(self, entry, ctx, p_chunks, consumer_ports, c_per_p: int):
        consumer = entry.single_consumer()
        inst = ctx.ir.execution.get(consumer.node.name)
        shard_dim, port_stride, full_dim = self._shard_params(entry, ctx)
        producer_ports = self._producer_port_ids(entry, ctx)
        producer_groups = {port: index for index, port in enumerate(producer_ports)}
        port_groups = {}

        for c_port in consumer_ports:
            desc = inst.variant.describe_input_staging(
                consumer.node, inst.config, consumer.tensor, c_port, None, entry.producer.node
            )
            rebase_descriptor_offset(desc, consumer.offset_base)
            offset = int(desc['offset'][shard_dim])
            extent = int(desc['io_tiling_dimension'][shard_dim])
            if offset < 0 or extent <= 0 or offset + extent > full_dim:
                raise ConfigRefused(
                    f'{entry.logical_tensor}: shard transport requires relay; consumer port {c_port} range '
                    f'[{offset}, {offset + extent}) is invalid for producer shard dim{shard_dim} extent {full_dim}.'
                )
            group = offset // port_stride
            if group < 0 or group >= len(producer_ports) or offset + extent > (group + 1) * port_stride:
                raise ConfigRefused(
                    f'{entry.logical_tensor}: shard transport requires relay; consumer port {c_port} range '
                    f'[{offset}, {offset + extent}) crosses producer shard boundary for slice {group}.'
                )
            port_groups.setdefault(group, []).append(c_port)

        chunks = []
        for p_ports in p_chunks:
            chunk = []
            for p_port in p_ports:
                chunk.extend(port_groups.get(producer_groups[p_port], []))
            chunk.sort()
            expected = len(p_ports) * c_per_p
            if len(chunk) != expected:
                raise ConfigRefused(
                    f'{entry.logical_tensor}: shard transport requires relay; expected {expected} consumer ports '
                    f'for producer slice set {p_ports}, got {len(chunk)}.'
                )
            chunks.append(chunk)
        if sorted(port for chunk in chunks for port in chunk) != sorted(consumer_ports):
            raise RuntimeError(f'{entry.logical_tensor}: consumer shard assignment is incomplete.')
        return chunks

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

    def _shard_params(self, entry, ctx):
        if entry.producer.node is not None:
            inst = ctx.ir.execution.get(entry.producer.node.name)
            ports = self._producer_port_ids(entry, ctx)
            d0 = inst.variant.describe_output_staging(
                entry.producer.node, inst.config, entry.producer.tensor, ports[0], None
            )
            d1 = (
                inst.variant.describe_output_staging(
                    entry.producer.node, inst.config, entry.producer.tensor, ports[1], None
                )
                if len(ports) > 1
                else None
            )
            localize_descriptor(d0, entry.producer.offset_base, entry.producer.buffer_dimension)
            if d1 is not None:
                localize_descriptor(d1, entry.producer.offset_base, entry.producer.buffer_dimension)
        else:
            consumer = entry.single_consumer()
            inst = ctx.ir.execution.get(consumer.node.name)
            ports = self._consumer_port_ids(entry, ctx)
            d0 = inst.variant.describe_input_staging(consumer.node, inst.config, consumer.tensor, ports[0], None, None)
            d1 = (
                inst.variant.describe_input_staging(consumer.node, inst.config, consumer.tensor, ports[1], None, None)
                if len(ports) > 1
                else None
            )

        shard_dim = int(d0['slice_dimension'])
        port_stride = (
            int(d1['offset'][shard_dim] - d0['offset'][shard_dim])
            if d1 is not None
            else int(d0['buffer_dimension'][shard_dim])
        )
        full_dim = int(d0['buffer_dimension'][shard_dim])
        if full_dim != port_stride * len(ports):
            raise ConfigRefused(
                f'{entry.logical_tensor}: shard transport is not legal on dim{shard_dim}; expected '
                f'buffer_dimension[{shard_dim}] == port_stride * ports ({full_dim} != {port_stride} * {len(ports)}).'
            )
        return shard_dim, port_stride, full_dim
