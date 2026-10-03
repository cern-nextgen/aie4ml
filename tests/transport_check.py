"""An executable model of a physical plan: tagged tensor elements pushed through every emitted walk.

Each kernel port declares, through its staging, which tensor element sits at each position of its buffer. This model
fills every producer buffer that way and moves the elements through the plan as the ADF tiling parameters say --
direct edges, ordered merges, memory tiles, and the host's files as the simulator packs them. Every consumer buffer
must then hold the elements it declares, and zeros where it asks for them, and every graph output must reassemble its
tensor. It shares no code with the walk generator, so it checks the transport against the ops' declarations rather
than against itself.
"""

from __future__ import annotations

import numpy as np
from aie4ml.op_impls.common_types import PORT_KIND_STREAM
from aie4ml.passes.transport.collect import TransportCollector
from aie4ml.passes.transport.layout import port_layout
from aie4ml.passes.utils import sanitize_identifier
from aie4ml.simulation import _extract_port_tile, _framed, _insert_port_tile, build_io_layout

UNWRITTEN = -1  # a memory-tile position no writer reached
PADDING = -2  # a producer buffer position that holds no element of the tensor


def check_transport(model) -> None:
    """Raise AssertionError naming every port that does not receive the elements it declares."""
    _Plan(model).check()


def walk(desc):
    """The flat buffer positions an ADF tiling parameter visits, in stream order, and which of them read as zeros."""
    dims = [int(d) for d in desc['buffer_dimension']]
    rank = len(dims)
    coords = np.array([desc['offset']], dtype=np.int64)
    loops = [
        (int(s['dimension']), int(s['wrap']), int(s['stride'])) for s in reversed(desc.get('tile_traversal') or [])
    ]
    loops += [(axis, int(desc['tiling_dimension'][axis]), 1) for axis in reversed(range(rank))]
    for axis, count, step in loops:  # slowest first
        steps = np.zeros((count, rank), dtype=np.int64)
        steps[:, axis] = np.arange(count) * step
        coords = (coords[:, None, :] + steps[None, :, :]).reshape(-1, rank)
    zeros = np.zeros(len(coords), dtype=bool)
    if desc.get('boundary_dimension') is not None:
        low = np.array(desc.get('boundary_offset') or [0] * rank)
        zeros = ((coords < low) | (coords >= low + np.array(desc['boundary_dimension']))).any(axis=1)
    inside = ((coords >= 0) & (coords < np.array(dims))).all(axis=1)
    if not (inside | zeros).all():
        raise AssertionError(f'a walk leaves its buffer {dims}: {desc}')
    flat = np.zeros(len(coords), dtype=np.int64)
    flat[inside] = np.ravel_multi_index(coords[inside].T, dims, order='F')
    return flat, zeros


def _positions(layout):
    """The tensor coordinate of each position of a kernel buffer, in memory order, and which are its data."""
    rank = len(layout.start)
    coords = np.array([layout.start], dtype=np.int64)
    for loop in layout.loops:
        steps = np.zeros((loop.count, rank), dtype=np.int64)
        steps[:, loop.axis] = np.arange(loop.count) * loop.step
        coords = (coords[:, None, :] + steps[None, :, :]).reshape(-1, rank)
    data = np.ones(len(coords), dtype=bool)
    for axis, (lo, hi) in enumerate(layout.data):
        data &= (coords[:, axis] >= lo) & (coords[:, axis] < hi)
    return coords, data


class _Tags:
    """A distinct positive tag for each element of a tensor whose coordinates span `bounds`."""

    def __init__(self, bounds):
        self.low = np.array([lo for lo, _ in bounds])
        self.shape = [max(1, hi - lo) for lo, hi in bounds]

    def of(self, coords):
        return 1 + np.ravel_multi_index((coords - self.low).T, self.shape, order='F')

    def tensor(self):
        """The tagged tensor in numpy order (buffer order reversed), as the host holds it."""
        coords = np.indices(self.shape).reshape(len(self.shape), -1).T + self.low
        return self.of(coords).reshape(self.shape).T


class _Plan:
    def __init__(self, model):
        from aie4ml.ir import get_backend_context

        self.model = model
        self.ctx = get_backend_context(model)
        self.plan = self.ctx.ir.physical.plan
        self.execution = self.ctx.ir.execution
        self.instances = {sanitize_identifier(inst.name): inst for inst in self.execution}
        self.io = {(port['direction'], int(port['port'])): port for port in self.plan.get('io_ports', ())}
        self.layout = build_io_layout(model)
        self.legs = {}  # (consumer id, group, port) -> (producer endpoint, consumer endpoint)
        self.outputs = {}  # (producer id, group) -> producer endpoint
        for entry in TransportCollector(self.execution).collect():
            if entry.graph_output:
                self.outputs[(sanitize_identifier(entry.producer.node.name), entry.producer.group)] = entry.producer
            for leg in entry.consumers:
                consumer = leg.consumer
                binding = self.instances[sanitize_identifier(consumer.node.name)].ports.inputs[consumer.tensor]
                for port in consumer.selected_ports(binding.count):
                    self.legs[(sanitize_identifier(consumer.node.name), consumer.group, port)] = (
                        leg.producer,
                        consumer,
                    )
        self.failures = []
        self.host_out = {}  # graph output tensor -> host array being reassembled

    # -- the ports' declarations ------------------------------------------------------------------------

    def _producer(self, endpoint, port):
        inst = self.execution.get(endpoint.node.name)
        staging = inst.variant.describe_output_staging(endpoint.node, inst.config, endpoint.tensor, port)
        return port_layout(staging).shifted(endpoint.offset_base)

    def _consumer(self, endpoint, producer, port):
        inst = self.execution.get(endpoint.node.name)
        staging = inst.variant.describe_input_staging(endpoint.node, inst.config, endpoint.tensor, port, producer.node)
        return port_layout(staging).shifted(endpoint.offset_base), 'boundary_dimension' in staging

    def _tags(self, producer, host_port=None):
        """The tags of a leg's tensor: the span of its producer's data, or the graph input's extent."""
        if producer.node is None:
            staging = self.io[('input', host_port)]['staging']
            return _Tags([(0, int(extent)) for extent in staging['io_boundary_dimension']])
        inst = self.execution.get(producer.node.name)
        windows = [self._producer(producer, port).data for port in range(inst.ports.outputs[producer.tensor].count)]
        return _Tags([(min(w[a][0] for w in windows), max(w[a][1] for w in windows)) for a in range(len(windows[0]))])

    def _filled(self, producer, port, tags):
        coords, data = _positions(self._producer(producer, port))
        values = np.full(len(coords), PADDING, dtype=np.int64)
        values[data] = tags.of(coords[data])
        return values

    def _receive(self, where, consumer_id, group, port, values, *, zero_filled):
        """Check what reached a consumer port: its data, and -- where the transport zero-fills -- its padding."""
        producer, consumer = self.legs[(consumer_id, group, port)]
        layout, zeros = self._consumer(consumer, producer, port)
        coords, data = _positions(layout)
        if len(values) != len(coords):
            self.failures.append(f'{where}: {len(values)} elements reach a buffer of {len(coords)}')
            return
        tags = self._leg_tags[(consumer_id, group, port)]
        wrong = np.flatnonzero(values[data] != tags.of(coords[data]))
        if len(wrong):
            self.failures.append(f'{where}: {len(wrong)} of {int(data.sum())} data positions hold other elements')
        if zero_filled and zeros and (values[~data] != 0).any():
            self.failures.append(f'{where}: padding it reads as zeros holds {int((values[~data] != 0).sum())} values')

    # -- the plan ----------------------------------------------------------------------------------------

    def check(self):
        self._leg_tags = {}
        for key, (producer, _consumer) in self.legs.items():
            self._leg_tags[key] = self._tags(producer, self._graph_port(key) if producer.node is None else None)
        for edge in self.plan.get('direct_edges', ()):
            self._direct_edge(edge)
        for merge in self.plan.get('merges', ()):
            self._merge(merge)
        for buffer in self.plan.get('buffers', ()):
            self._memory_tile(buffer)
        for tensor, ports in self.layout.outputs.items():
            want = self._tags(self.outputs[self._output_key(ports[0])]).tensor()
            have = self.host_out.get(tensor)
            if have is None or have.shape[1:] != want.shape or (have[0] != want).any():
                self.failures.append(f'graph output {tensor}: the host does not reassemble the tensor')
        if self.failures:
            raise AssertionError('transport check failed:\n  ' + '\n  '.join(self.failures))

    def _graph_port(self, key):
        """The graph-input port feeding a consumer port."""
        for port in self.plan.get('io_ports', ()):
            if port['direction'] != 'input':
                continue
            for target in self._targets(port):
                if target == key:
                    return int(port['port'])
        raise AssertionError(f'{key}: no graph-input port feeds it')

    def _targets(self, io_port):
        endpoint = io_port['endpoint']
        if endpoint.startswith('buffer_'):
            buffer = next(b for b in self.plan['buffers'] if endpoint.split('.')[0] == b['name'])
            return [
                (r['target_endpoint']['op_impl_id'], r['target_endpoint']['group'], r['target_endpoint']['port'])
                for r in buffer['readers']
                if r['target_type'] == 'op_impl'
            ]
        name, rest = endpoint.split('.', 1)
        group, port = rest.rstrip(']').split('[')
        return [(name, group, int(port))]

    def _output_key(self, io_port):
        for (producer_id, group), producer in self.outputs.items():
            if producer.tensor == io_port.tensor:
                return (producer_id, group)
        raise AssertionError(f'{io_port.tensor}: no producer of a graph output')

    def _host_stream(self, port):
        io = next(p for p in self.layout.inputs[self.io[('input', port)]['tensor']] if p.port == port)
        tags = _Tags([(0, int(extent)) for extent in io.staging['io_boundary_dimension']])
        return _framed(_extract_port_tile(tags.tensor()[None], io), io).astype(np.int64)

    def _host_receive(self, graph_port, values):
        io = next(p for ports in self.layout.outputs.values() for p in ports if p.port == graph_port)
        per = int(np.prod(io.numpy_tile_shape))
        out = self.host_out.setdefault(io.tensor, np.zeros((1, *io.numpy_boundary_shape), dtype=np.int64))
        _insert_port_tile(out, values[:per].reshape(1, *io.numpy_tile_shape), io)

    def _direct_edge(self, edge):
        """A direct edge hands the consumer the producer's buffer; a PLIO the host's tile, through the kernel port's
        access pattern -- or in order, for a stream port, which carries its buffer as the kernel holds it."""
        source, target = edge['source'], edge['target']
        where = f'{source} -> {target}'
        if source.startswith('ifm['):
            graph_port = int(source[4:-1])
            consumer_id, group, port = self._targets({'endpoint': target})[0]
            producer, consumer = self.legs[(consumer_id, group, port)]
            layout, _ = self._consumer(consumer, producer, port)
            buffer = np.full(len(_positions(layout)[0]), UNWRITTEN, dtype=np.int64)
            stream_port = self.instances[consumer_id].ports.inputs[consumer.tensor].kind == PORT_KIND_STREAM
            flat, zeros = self._access(self.io[('input', graph_port)]['descriptor'], len(buffer), stream_port)
            stream = self._host_stream(graph_port)
            if len(stream) < len(flat):
                self.failures.append(f'{where}: {len(stream)} elements for a walk of {len(flat)}')
                return
            buffer[flat[~zeros]] = stream[: len(flat)][~zeros]
            self._receive(where, consumer_id, group, port, buffer, zero_filled=False)
            return
        producer_id, rest = source.split('.', 1)
        group, port = rest.rstrip(']').split('[')
        if target.startswith('ofm['):
            graph_port = int(target[4:-1])
            producer = self.outputs[(producer_id, group)]
            values = self._filled(producer, int(port), self._tags(producer))
            stream_port = self.instances[producer_id].ports.outputs[producer.tensor].kind == PORT_KIND_STREAM
            flat, zeros = self._access(self.io[('output', graph_port)]['descriptor'], len(values), stream_port)
            self._host_receive(graph_port, np.where(zeros, 0, values[flat]))
            return
        consumer_id, cgroup, cport = self._targets({'endpoint': target})[0]
        producer, _consumer = self.legs[(consumer_id, cgroup, cport)]
        values = self._filled(producer, int(port), self._leg_tags[(consumer_id, cgroup, cport)])
        self._receive(where, consumer_id, cgroup, cport, values, zero_filled=False)

    def _merge(self, merge):
        """An ordered packet merge hands each reader its writers' buffers one after another, in writer order."""
        for target in merge['readers']:
            consumer_id, group, port = self._targets({'endpoint': target})[0]
            producer, _consumer = self.legs[(consumer_id, group, port)]
            tags = self._leg_tags[(consumer_id, group, port)]
            ports = [int(writer.rstrip(']').split('[')[1]) for writer in merge['writers']]
            values = np.concatenate([self._filled(producer, p, tags) for p in ports])
            self._receive(f"{merge['name']} -> {target}", consumer_id, group, port, values, zero_filled=False)

    @staticmethod
    def _access(desc, size, stream):
        """The kernel buffer positions a PLIO's DMA visits: its access pattern, or all of them in order on a stream."""
        return (np.arange(size), np.zeros(size, dtype=bool)) if stream else walk(desc)

    def _memory_tile(self, buffer):
        memory = np.full(int(np.prod(buffer['dimension'])), UNWRITTEN, dtype=np.int64)
        readers = [r for r in buffer['readers'] if r['target_type'] == 'op_impl']
        if readers:
            key = tuple(readers[0]['target_endpoint'][k] for k in ('op_impl_id', 'group', 'port'))
            producer, _ = self.legs[key]
            tags = self._leg_tags[key]
        else:
            source = buffer['writers'][0]['source_endpoint']
            producer = self.outputs[(source['op_impl_id'], source['group'])]
            tags = self._tags(producer)
        for writer in buffer['writers']:
            flat, zeros = walk(writer['descriptor'])
            if zeros.any():
                self.failures.append(f'{writer["target"]}: a write walk zero-fills')
            if writer['source_type'] == 'plio':
                stream = self._host_stream(int(writer['source_endpoint']['port']))
            else:
                stream = self._filled(producer, int(writer['source_endpoint']['port']), tags)
            if len(stream) < len(flat):
                self.failures.append(f'{writer["target"]}: {len(stream)} elements for a walk of {len(flat)}')
                continue
            memory[flat] = stream[: len(flat)]
        for reader in buffer['readers']:
            flat, zeros = walk(reader['descriptor'])
            values = np.where(zeros, 0, memory[flat])
            if reader['target_type'] == 'plio':
                self._host_receive(int(reader['target_endpoint']['port']), values)
                continue
            target = reader['target_endpoint']
            self._receive(
                reader['source'], target['op_impl_id'], target['group'], int(target['port']), values, zero_filled=True
            )
