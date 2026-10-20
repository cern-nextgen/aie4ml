from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...errors import ConfigRefused
from ...ir import ExecutionView, OpNode
from ...op_impls.utils.tensor_view import map_view_axis
from .model import Connection, EdgeEntry, Endpoint


class TransportCollector:
    """Expand the execution graph's connectivity, views included, into semantic transport entries.

    It reads the execution IR only: entries, the values they read and write, and the graph boundary.
    """

    def __init__(self, execution):
        self.execution = execution
        self.producers: Dict[str, Tuple[OpNode, str]] = {
            tname: (inst.node, inst.ports.outputs[tname].group) for inst in execution for tname in inst.outputs
        }

    def collect(self) -> List[EdgeEntry]:
        return self._group_edges(self._collect_connections())

    def input_connections(self, inst) -> List[Connection]:
        """The legs into one instance: from a kernel, through a folded view, or from the graph boundary."""
        return [leg for item in inst.inputs for leg in self.item_connections(inst, item)]

    def item_connections(self, inst, item) -> List[Connection]:
        """The legs into one input of an instance."""
        tname = item.tensor
        cg = inst.ports.inputs[tname].group
        view = self.execution.values[tname].view
        if view is not None:
            if view.kind == 'concat':
                return self._concat_connections(inst.node, tname, cg, view)
            return [self._slice_connection(inst.node, tname, cg, view)]
        if tname in self.producers:
            p, pg = self.producers[tname]
            return [Connection(tname, Endpoint(p, tname, pg), Endpoint(inst.node, tname, cg))]
        return [Connection(tname, Endpoint(None, tname, 'graph_input'), Endpoint(inst.node, tname, cg))]

    def ports_read(self, value: str, source: str) -> Tuple[int, ...]:
        """The ports of `source`'s producer that reading `value` reads: all of them for `source` itself or a concat
        over it, those a slice of it covers, none where `value` is not read from `source`."""
        inst = self._kernel_inst(self.producers[source][0])
        every = tuple(range(int(inst.ports.outputs[source].count)))
        view = self.execution.values[value].view
        if view is None:
            return every if value == source else ()
        if source not in view.sources:
            return ()
        if view.kind == 'concat':
            return every
        part = view.parts[0]
        return self._slice_producer_ports(inst.node, source, view.axis, part.start, part.extent)[0]

    def reads(self, tensor: str) -> List[Tuple[str, str]]:
        """Every read of `tensor`: (the value read -- `tensor` itself or a view over it -- and the instance reading
        it), once per input."""
        return [
            (item.tensor, inst.name)
            for inst in self.execution
            for item in inst.inputs
            if item.tensor == tensor or tensor in getattr(self.execution.values[item.tensor].view, 'sources', ())
        ]

    def port_readers(self, tensor: str, port: int, reads: Optional[List[Tuple[str, str]]] = None) -> List[str]:
        """Who reads port `port` of `tensor`'s producer, of `reads` (`reads`, this graph's by default): each reader
        once per input reading it, whole or through a view covering it, and the graph boundary where `tensor` is a
        graph output."""
        reads = self.reads(tensor) if reads is None else reads
        readers = [reader for value, reader in reads if port in self.ports_read(value, tensor)]
        return readers + (['the graph boundary'] if tensor in self.execution.graph_outputs else [])

    def output_connections(self, inst, read) -> List[Connection]:
        """The graph-boundary legs out of one instance: its graph outputs, and any output nothing in `read` reads."""
        return [
            Connection(tname, Endpoint(inst.node, tname, inst.ports.outputs[tname].group), None)
            for tname in inst.outputs
            if tname in self.execution.graph_outputs or tname not in read
        ]

    def _collect_connections(self) -> List[Connection]:
        for name in self.execution.graph_outputs:
            view = self.execution.values[name].view
            if view is not None:
                raise NotImplementedError(f'{name}: {view.kind}-backed graph outputs are not implemented.')

        connections: List[Connection] = []
        read: set[str] = set()
        for inst in self.execution:
            connections.extend(self.input_connections(inst))
            for item in inst.inputs:
                view = self.execution.values[item.tensor].view
                read.update(view.sources if view is not None else (item.tensor,))
        for inst in self.execution:
            connections.extend(self.output_connections(inst, read))
        return connections

    def _slice_connection(
        self,
        consumer: OpNode,
        slice_tensor: str,
        consumer_group: str,
        view: ExecutionView,
    ) -> Connection:
        part = view.parts[0]
        source_name = part.source
        producer, producer_group = self._kernel_source(
            slice_tensor,
            source_name,
            view_kind='slice',
        )
        ports, offset_base = self._slice_producer_ports(
            producer,
            source_name,
            view.axis,
            part.start,
            part.extent,
        )
        return Connection(
            slice_tensor,
            Endpoint(
                producer,
                source_name,
                producer_group,
                ports=ports,
                offset_base=offset_base,
            ),
            Endpoint(consumer, slice_tensor, consumer_group),
        )

    def _slice_producer_ports(
        self, producer: OpNode, source_tensor: str, axis: int, start: int, extent: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        inst = self._kernel_inst(producer)
        view = inst.port_views[source_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, axis)
        total_ports = int(inst.ports.outputs[source_tensor].count)
        end = int(start) + int(extent)
        selected = []
        ranges = []
        first_desc = None
        for port in range(total_ports):
            desc = inst.staging[source_tensor][port]
            first_desc = desc if first_desc is None else first_desc
            port_start, port_end = self._descriptor_axis_range(desc, axis_dim)
            overlaps = port_start < end and port_end > start
            if overlaps and not (port_start >= start and port_end <= end):
                raise ConfigRefused(
                    f'{source_tensor}: slice range [{start}, {end}) crosses producer port {port} '
                    f'range [{port_start}, {port_end}); packed slice/relay is not implemented.'
                )
            if overlaps:
                selected.append(port)
                ranges.append((port_start, port_end))
        ordered_ranges = sorted(ranges)
        contiguous = all(left[1] == right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:]))
        if not selected or ordered_ranges[0][0] != start or ordered_ranges[-1][1] != end or not contiguous:
            raise ConfigRefused(
                f'{source_tensor}: slice range [{start}, {end}) does not align exactly with producer ports.'
            )

        base = tuple(int(start) if dim == axis_dim else 0 for dim in range(len(first_desc['buffer_dimension'])))
        return tuple(selected), base

    def _concat_connections(
        self,
        consumer: OpNode,
        concat_tensor: str,
        consumer_group: str,
        concat_view: ExecutionView,
    ) -> List[Connection]:
        ports_by_source = self._concat_consumer_ports(consumer, concat_tensor, concat_view)
        conns: List[Connection] = []
        for part in concat_view.parts:
            source_name = part.source
            ports = tuple(ports_by_source[source_name])
            if not ports:
                continue
            offset_base = self._concat_consumer_offset_base(consumer, concat_tensor, concat_view, part.start)
            producer, producer_group = self._kernel_source(
                concat_tensor,
                source_name,
                view_kind='concat',
            )
            conns.append(
                Connection(
                    concat_tensor,
                    Endpoint(producer, source_name, producer_group),
                    Endpoint(
                        consumer,
                        concat_tensor,
                        consumer_group,
                        ports=ports,
                        offset_base=offset_base,
                    ),
                )
            )
        return conns

    def _concat_consumer_offset_base(
        self, consumer: OpNode, concat_tensor: str, concat_view: ExecutionView, start: int
    ) -> Tuple[int, ...]:
        inst = self._kernel_inst(consumer)
        view = inst.port_views[concat_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, concat_view.axis)
        return tuple(int(start) if dim == axis_dim else 0 for dim in range(view.rank))

    def _concat_consumer_ports(
        self, consumer: OpNode, concat_tensor: str, concat_view: ExecutionView
    ) -> Dict[str, List[int]]:
        inst = self._kernel_inst(consumer)
        if inst is None:
            raise RuntimeError(f'{concat_tensor}: concat consumer {consumer.name!r} is not resolved.')
        total_ports = int(inst.ports.inputs[concat_tensor].count)
        slices = [(part.source, part.start, part.start + part.extent) for part in concat_view.parts]
        out: Dict[str, List[int]] = {name: [] for name, _, _ in slices}
        axis = concat_view.axis
        view = inst.port_views[concat_tensor]
        axis_dim = self._view_axis_to_buffer_dim(view, axis)
        for port in range(total_ports):
            desc = inst.staging[concat_tensor][port]
            start, end = self._descriptor_axis_range(desc, axis_dim)

            owners = [name for name, lo, hi in slices if start >= lo and end <= hi]
            if len(owners) != 1:
                raise ConfigRefused(
                    f'{concat_tensor}: concat consumer port {port} axis {axis} range [{start}, {end}) does not '
                    'fit exactly inside one concat input slice; packed concat/relay is not implemented.'
                )
            out[owners[0]].append(int(port))
        return out

    @staticmethod
    def _descriptor_axis_range(desc: Dict[str, Any], dim: int) -> Tuple[int, int]:
        start = int(desc['offset'][dim])
        extent = int(desc['io_tiling_dimension'][dim])
        if extent <= 0:
            raise RuntimeError(f'invalid descriptor extent {extent} on dim{dim}.')
        return start, start + extent

    @staticmethod
    def _view_axis_to_buffer_dim(view, axis: int) -> int:
        rank = int(view.rank)
        normalized = int(axis)
        if normalized < 0:
            normalized += rank
        if normalized < 0 or normalized >= rank:
            raise ValueError(f'view axis {axis} is out of range for rank {rank}.')
        real_axis = map_view_axis(view, normalized)
        return int(view.buffer_order.index(int(real_axis)))

    # -------------------------------------------------------------------------
    # Group edges
    # -------------------------------------------------------------------------

    def _group_edges(self, connections: Iterable[Connection]) -> List[EdgeEntry]:
        grouped: Dict[Tuple[str, str, str, str], EdgeEntry] = {}

        for c in connections:
            consumer = c.consumer
            consumer_tensor = consumer.tensor if consumer is not None else c.logical_tensor
            consumer_group = consumer.group if consumer is not None else 'graph_output'
            key = (c.producer.tensor, c.producer.group, consumer_tensor, consumer_group)
            if key not in grouped:
                grouped[key] = EdgeEntry(
                    logical_tensor=c.logical_tensor,
                    producer=c.producer,
                    producer_port_count=self._producer_port_count(c.producer),
                )

            e = grouped[key]
            if e.producer != c.producer:
                raise RuntimeError(f'{c.logical_tensor}: inconsistent producer endpoint for grouped edge.')

            if c.consumer is None:
                e.graph_output = True
            else:
                e.consumers.append(c)
                if c.producer.node is None:
                    e.producer_port_count = max(
                        e.producer_port_count,
                        self._consumer_port_count(c.consumer),
                    )

        return list(grouped.values())

    def _kernel_source(
        self,
        logical_tensor: str,
        source_name: str,
        *,
        view_kind: str,
    ) -> Tuple[OpNode, str]:
        if source_name in self.producers:
            return self.producers[source_name]

        value = self.execution.values.get(source_name)
        if value is None:
            raise ValueError(f'{logical_tensor}: {view_kind} source tensor {source_name!r} does not exist.')
        if value.view is not None:
            raise NotImplementedError(
                f'{logical_tensor}: chained view transport through {value.view.kind} '
                f'{value.view.node!r} is not implemented.'
            )
        if value.producer is not None:
            raise RuntimeError(
                f'{logical_tensor}: {view_kind} source producer {value.producer!r} has no resolved execution output.'
            )
        if source_name not in self.execution.graph_inputs:
            raise RuntimeError(
                f'{logical_tensor}: {view_kind} source tensor {source_name!r} has no producer and is not a '
                'declared graph input.'
            )
        raise NotImplementedError(
            f'{logical_tensor}: {view_kind} input {source_name!r} is a graph input; '
            f'{view_kind}-backed graph-input legs are not implemented.'
        )

    def _kernel_inst(self, node):
        return self.execution.get(node.name) if node else None

    def _producer_port_count(self, endpoint: Endpoint):
        if endpoint.node is None:
            return 1
        total = self._kernel_inst(endpoint.node).ports.outputs[endpoint.tensor].count
        return len(endpoint.selected_ports(total))

    def _consumer_port_count(self, endpoint: Endpoint):
        total = self._kernel_inst(endpoint.node).ports.inputs[endpoint.tensor].count
        return len(endpoint.selected_ports(total))
