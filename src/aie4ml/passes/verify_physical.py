"""The physical plan's invariants, checked before any code is generated.

Exactly the kernel graphs are placed, each buffer where its graph lists it or, an input, in a memory its reading core
reaches, and every edge the execution graph requires in shared memory is realised
as one on each of its ports: a direct hand-over per port, legal by the rule placement searched under
(`shared_buffer`), its two ports pinned to the same memory, recorded as `shared_memory`, and carrying no
DMA access pattern.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import zip_longest

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..op_impls.base import KERNEL_BANKS
from ..op_impls.common_types import PORT_KIND_BUFFER
from ..utils import sanitize_identifier
from .base import AIEPass
from .shared_buffer import SHARED_MEMORY, hand_over_problem, placed_buffers, placed_locations, reached
from .transport.dma_resources import KERNEL_BUFFERS, memtile_port_bds, pool_use, tile_port_bds
from .transport.model import TransportUnit
from .transport.routing import BELOW, port_tiles, route_overflow

_PORT = re.compile(r'^(?P<graph>\w+)\.(?P<group>\w+)\[(?P<port>\d+)\]$')
_PLIO = re.compile(r'^(ifm|ofm)\[\d+\]$')


def _port(endpoint: str, graphs):
    match = _PORT.match(endpoint)
    if match is None or match['graph'] not in graphs:
        raise RuntimeError(f'physical plan endpoint {endpoint!r} names no placed kernel graph port.')
    return graphs[match['graph']], match['group'], int(match['port'])


def _kernel_ports(inst, group: str, port: int, direction: str):
    return {
        f'{sanitize_identifier(inst.name)}.{kernel_port}'
        for binding in getattr(inst.ports, direction).values()
        if binding.group == group
        for kernel_port in binding.endpoints[port]
    }


def verify_physical(ctx) -> None:
    execution, physical = ctx.ir.execution, ctx.ir.physical
    if set(physical.placements) != set(execution.instances):
        raise RuntimeError(
            f'physical plan places {sorted(physical.placements)}, not the kernel graphs {sorted(execution.instances)}.'
        )
    for inst in execution:
        placement = physical.placements[inst.name]
        col, row = int(placement['col']), int(placement['row'])
        for loc, at in placed_buffers(ctx, inst):
            listed = (col + loc.rel_col, row + loc.rel_row, tuple(loc.banks))
            if at != listed and at[:2] not in reached(loc, col, row, ctx.device.alternating_horizontal):
                raise RuntimeError(
                    f'{inst.name}.{loc.port_group}[{loc.port}]: placed at {at}, neither where its graph lists it '
                    f'({listed}) nor in a memory its reading core reaches.'
                )
    graphs = {}
    for inst in execution:
        name = sanitize_identifier(inst.name)
        if graphs.setdefault(name, inst) is not inst:
            raise RuntimeError(f'{graphs[name].name} and {inst.name}: both generate the graph name {name!r}.')
    accessed = {
        access['endpoint']
        for key in ('kernel_read_accesses', 'kernel_write_accesses')
        for access in physical.plan.get(key, ())
    }
    edges = physical.plan.get('direct_edges', ())
    for inst in execution:
        for item in inst.inputs:
            if not item.shared_memory:
                continue
            group, ports = inst.ports.inputs[item.tensor].group, inst.ports.inputs[item.tensor].count
            name = sanitize_identifier(inst.name)
            legs = [
                e for e in edges if (end := _PORT.match(e['target'])) and (end['graph'], end['group']) == (name, group)
            ]
            if len(legs) != ports:
                raise RuntimeError(
                    f'{inst.name}: must read {item.tensor!r} through shared memory on each of its {ports} ports, '
                    f'but the plan hands it over directly on {len(legs)}.'
                )
            ends = [(_port(leg['source'], graphs), _port(leg['target'], graphs)) for leg in legs]
            unit = TransportUnit(tuple(source[2] for source, _ in ends), tuple(target[2] for _, target in ends))
            for leg, ((producer, p_group, p_port), (_, c_group, c_port)) in zip(legs, ends):
                problem = hand_over_problem(ctx, unit, producer, p_group, p_port, inst, c_group, c_port)
                if not problem and leg.get('realization') != SHARED_MEMORY:
                    problem = f"the plan records it as {leg.get('realization')!r}"
                kernel_ports = _kernel_ports(producer, p_group, p_port, 'outputs') | _kernel_ports(
                    inst, c_group, c_port, 'inputs'
                )
                if not problem and accessed & kernel_ports:
                    problem = f'{sorted(accessed & kernel_ports)} carry a DMA access pattern'
                if problem:
                    raise RuntimeError(f'{leg["source"]} -> {leg["target"]}: must be shared memory, but {problem}.')


def verify_memory_banks(ctx) -> None:
    """No two ops hold one core or memory bank -- a kernel's core and its tile's `KERNEL_BANKS`, a buffer's banks --
    but the buffer a shared hand-over names at both its ends; and a buffer placed away from where its graph lists it
    holds no bank its own op holds otherwise."""
    physical = ctx.ir.physical
    graphs = {sanitize_identifier(inst.name): inst for inst in ctx.ir.execution}
    holders = defaultdict(set)
    for inst in ctx.ir.execution:
        placement = physical.placements[inst.name]
        col, row = int(placement['col']), int(placement['row'])
        tiles = [
            (col + dc, row + dr) for dc in range(int(placement['width'])) for dr in range(int(placement['height']))
        ]
        held = {(c, r, 'core') for c, r in tiles} | {(c, r, bank) for c, r in tiles for bank in KERNEL_BANKS}
        moved = []
        for loc, at in placed_buffers(ctx, inst):
            banks = {(at[0], at[1], bank) for bank in at[2]}
            if at == (col + loc.rel_col, row + loc.rel_row, tuple(loc.banks)):
                held |= banks
            else:
                moved.append((loc, banks))
        for loc, banks in moved:
            if banks & held:
                raise RuntimeError(
                    f'{inst.name}.{loc.port_group}[{loc.port}]: placed on banks {sorted(banks & held)} its own op '
                    'holds.'
                )
            held |= banks
        for resource in held:
            holders[resource].add(inst.name)
    shared = {}
    for edge in physical.plan.get('direct_edges', ()):
        if edge.get('realization') == SHARED_MEMORY:
            producer, group, port = _port(edge['source'], graphs)
            consumer = _port(edge['target'], graphs)[0]
            for c, r, banks in placed_locations(ctx, producer, group, port):
                shared.update({(c, r, bank): {producer.name, consumer.name} for bank in banks})
    for resource, names in sorted(holders.items(), key=str):
        if len(names) > 1 and names != shared.get(resource):
            raise ConfigRefused(f'{sorted(names)} all hold {resource}: one core or memory bank, two ops.')


def verify_dma_resources(ctx) -> None:
    """Every DMA's BDs within its pools: each memory-tile shared buffer on its tile's pools, each compute tile's
    DMA-fed buffer ports on the tile that holds the buffer."""
    device, plan = ctx.device, ctx.ir.physical.plan
    dma = device.memtile_dma
    for buffer in plan.get('buffers', ()):
        if dma is None:
            raise RuntimeError(
                f"{buffer['name']}: a memory-tile buffer on {device.platform}, which has no memory tile."
            )
        count = int(buffer['num_buffers'])
        writers = [
            memtile_port_bds(w['descriptor'], count, device.generation, buffer['name']) for w in buffer['writers']
        ]
        readers = [
            memtile_port_bds(r['descriptor'], count, device.generation, buffer['name']) for r in buffer['readers']
        ]
        unused = [0] * dma.bd_pools
        tiles = zip_longest(pool_use(writers, dma), pool_use(readers, dma), fillvalue=unused)
        pools = [[w + r for w, r in zip(written, read)] for written, read in tiles]
        if max(map(max, pools)) > dma.bds // dma.bd_pools:
            raise ConfigRefused(
                f"{buffer['name']}: its {len(writers)} writers and {len(readers)} readers need {pools} BDs (per pool "
                f'of each memory tile they take) from memory tiles whose pools hold {dma.bds // dma.bd_pools} each.'
            )

    graphs = {sanitize_identifier(inst.name): inst for inst in ctx.ir.execution}
    shared = set()
    for edge in plan.get('direct_edges', ()):
        if edge.get('realization') == SHARED_MEMORY:
            shared |= {edge['source'], edge['target']}
    verify_tile_dma_channels(ctx, graphs, shared)

    accesses = {
        a['endpoint']: a['descriptor']
        for key in ('kernel_read_accesses', 'kernel_write_accesses')
        for a in plan.get(key, ())
    }
    used = defaultdict(int)
    for name, inst in graphs.items():
        for direction in ('inputs', 'outputs'):
            for tensor, binding in getattr(inst.ports, direction).items():
                if binding.kind != PORT_KIND_BUFFER:
                    continue
                element = (
                    inst.variant.output_precision(inst.config)
                    if direction == 'outputs'
                    else inst.variant.input_precision(inst.config, inst.input(tensor).role)
                )
                for port, endpoints in enumerate(binding.endpoints):
                    if f'{name}.{binding.group}[{port}]' in shared:
                        continue
                    owners = sorted({(col, row) for col, row, _ in placed_locations(ctx, inst, binding.group, port)})
                    for index, endpoint in enumerate(endpoints):
                        where = f'{name}.{endpoint}'
                        owner = owners[0] if len(owners) == 1 else owners[index]
                        used[owner] += tile_port_bds(
                            accesses.get(where), KERNEL_BUFFERS, device.tile_dma, device.generation, element.width
                        )
    for owner, bds in used.items():
        if bds > device.tile_dma.bds:
            raise ConfigRefused(f'tile {owner}: its DMA needs {bds} BDs, beyond its {device.tile_dma.bds}.')


def verify_tile_dma_channels(ctx, graphs, shared) -> None:
    """Every compute tile's DMA within its channels: each kernel buffer that is not shared memory takes an S2MM
    channel (input) or an MM2S channel (output) on the tile that holds it. The placer only reports an
    oversubscribed tile as a failed placement."""
    limit = int(ctx.device.tile_dma.channels)
    used = defaultdict(lambda: defaultdict(list))  # tile -> direction -> kernel ports
    for name, inst in graphs.items():
        for direction, channel in (('inputs', 'S2MM'), ('outputs', 'MM2S')):
            for binding in getattr(inst.ports, direction).values():
                if binding.kind != PORT_KIND_BUFFER:
                    continue
                for port, endpoints in enumerate(binding.endpoints):
                    if f'{name}.{binding.group}[{port}]' in shared:
                        continue
                    owners = sorted({(col, row) for col, row, _ in placed_locations(ctx, inst, binding.group, port)})
                    if len(owners) not in (1, len(endpoints)):
                        raise RuntimeError(f'{name}.{binding.group}[{port}]: its buffers sit on {owners}.')
                    for index, endpoint in enumerate(endpoints):
                        owner = owners[0] if len(owners) == 1 else owners[index]
                        used[owner][channel].append(f'{name}.{endpoint}')
    for owner, channels in sorted(used.items()):
        for channel, ports in channels.items():
            if len(ports) > limit:
                raise ConfigRefused(
                    f'tile {owner}: {len(ports)} buffers need its {channel} DMA ({", ".join(sorted(ports))}), '
                    f'which has {limit} channels.'
                )


def verify_stream_routes(ctx) -> None:
    """Every stream within the switches' ports (`transport.routing`): each hand-over not in shared memory -- a DMA
    between tiles, each memory-tile writer and reader, each merge writer, each PLIO -- from its source port to all its
    targets."""
    physical = ctx.ir.physical
    graphs = {sanitize_identifier(inst.name): inst for inst in ctx.ir.execution}
    buffers = {buffer['name'] for buffer in physical.plan.get('buffers', ())}

    def tiles(endpoint: str, direction: str):
        match = _PORT.match(endpoint)
        if match is not None and match['graph'] in graphs:
            inst, group, port = graphs[match['graph']], match['group'], int(match['port'])
            binding = next(b for b in getattr(inst.ports, direction).values() if b.group == group)
            if binding.kind == PORT_KIND_BUFFER:
                return {(col, row) for col, row, _ in placed_locations(ctx, inst, group, port)}
            placement = physical.placements[inst.name]
            return port_tiles(inst, direction, group, port, int(placement['col']), int(placement['row']))
        if _PLIO.match(endpoint) or (match is not None and match['graph'] in buffers):
            return {BELOW}
        raise RuntimeError(f'stream endpoint {endpoint!r} names no kernel graph, memory-tile buffer or PLIO port.')

    streams = {}

    def add(source: str, target: str) -> None:
        if source not in streams:
            origin = tiles(source, 'outputs')
            if len(origin) != 1:
                raise RuntimeError(f'{source}: one stream source on the tiles {sorted(origin)}.')
            streams[source] = (next(iter(origin)), set())
        streams[source][1].update(tiles(target, 'inputs'))

    for edge in physical.plan.get('direct_edges', ()):
        if edge.get('realization') != SHARED_MEMORY:
            add(edge['source'], edge['target'])
    for buffer in physical.plan.get('buffers', ()):
        for port in (*buffer['writers'], *buffer['readers']):
            add(port['source'], port['target'])
    for merge in physical.plan.get('merges', ()):  # each writer's stream counted to every reader: an upper bound
        for writer in merge['writers']:
            for reader in merge['readers']:
                add(writer, reader)
    problem = route_overflow(streams, ctx.device.stream_switch_ports, ctx.device.columns)
    if problem:
        raise ConfigRefused(f'stream routes: {problem}.')


class VerifyPhysicalPlan(AIEPass):
    def __init__(self):
        self.name = 'verify_physical_plan'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        verify_physical(ctx)
        verify_memory_banks(ctx)
        verify_dma_resources(ctx)
        verify_stream_routes(ctx)
        return False
