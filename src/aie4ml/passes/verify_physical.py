"""The physical plan's invariants, checked before any code is generated.

Exactly the kernel graphs are placed, and every edge the execution graph requires in shared memory is realised
as one on each of its ports: a direct hand-over per port, legal by the rule placement searched under
(`shared_buffer`), its two ports pinned to the same memory, recorded as `shared_memory`, and carrying no
DMA access pattern.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..op_impls.common_types import PORT_KIND_BUFFER
from .base import AIEPass
from .shared_buffer import SHARED_MEMORY, location_problem, pinned_locations, static_problem
from .transport.dma_resources import memtile_port_bds, pool_use, tile_port_bds
from .utils import sanitize_identifier

KERNEL_BUFFERS = 2  # a kernel's DMA-fed buffer port is double-buffered

_PORT = re.compile(r'^(?P<graph>\w+)\.(?P<group>\w+)\[(?P<port>\d+)\]$')


def _port(endpoint: str, graphs):
    match = _PORT.match(endpoint)
    if match is None or match['graph'] not in graphs:
        raise RuntimeError(f'physical plan endpoint {endpoint!r} names no placed kernel graph port.')
    return graphs[match['graph']], match['group'], int(match['port'])


def _pinned(ctx, inst, group: str, port: int):
    placement = ctx.ir.physical.placements[inst.name]
    return pinned_locations(inst, group, port, int(placement['col']), int(placement['row']))


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
            legs = [e for e in edges if e['tensor'] == item.tensor and _port(e['target'], graphs)[0] is inst]
            ports = inst.ports.inputs[item.tensor].count
            if len(legs) != ports:
                raise RuntimeError(
                    f'{inst.name}: must read {item.tensor!r} through shared memory on each of its {ports} ports, '
                    f'but the plan hands it over directly on {len(legs)}.'
                )
            for leg in legs:
                producer, p_group, p_port = _port(leg['source'], graphs)
                _, c_group, c_port = _port(leg['target'], graphs)
                problem = static_problem(ctx, item.tensor, producer, p_group, p_port, inst, c_group, c_port)
                problem = problem or location_problem(
                    _pinned(ctx, producer, p_group, p_port), _pinned(ctx, inst, c_group, c_port)
                )
                if not problem and leg.get('realization') != SHARED_MEMORY:
                    problem = f"the plan records it as {leg.get('realization')!r}"
                kernel_ports = _kernel_ports(producer, p_group, p_port, 'outputs') | _kernel_ports(
                    inst, c_group, c_port, 'inputs'
                )
                if not problem and accessed & kernel_ports:
                    problem = f'{sorted(accessed & kernel_ports)} carry a DMA access pattern'
                if problem:
                    raise RuntimeError(f'{leg["source"]} -> {leg["target"]}: must be shared memory, but {problem}.')


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
        pools = [w + r for w, r in zip(pool_use(writers, dma), pool_use(readers, dma))]
        if max(pools) > dma.bds // dma.bd_pools:
            raise ConfigRefused(
                f"{buffer['name']}: its {len(writers)} writers and {len(readers)} readers need {pools} BDs from a "
                f'memory tile whose pools hold {dma.bds // dma.bd_pools} each.'
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
                    owners = sorted({(col, row) for col, row, _ in _pinned(ctx, inst, binding.group, port)})
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
                    owners = sorted({(col, row) for col, row, _ in _pinned(ctx, inst, binding.group, port)})
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


class VerifyPhysicalPlan(AIEPass):
    def __init__(self):
        self.name = 'verify_physical_plan'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        verify_physical(ctx)
        verify_dma_resources(ctx)
        return False
