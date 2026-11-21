"""A lowered design's interval, first-inference latency and tile use, compile-free -- the one estimate the optimizer
ranks by and users read: `print(estimate(aie4ml.from_hls4ml(hls_model)))`.

Kernel cycles come from the generation's calibrated cost model (`ARTIFACTS`), composed along the physical plan with the
chain solver and the handoffs, every rate and depth a device fact (aie_devices.json). Interval and latency are given
at the kernels' point estimates and at the ends of their empirical ranges; hard latency claims need a measurement.
Tile use is exact. A kernel or handoff the model does not cover leaves no estimate, only the reasons.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..model import AIEModel
from ..passes.utils import sanitize_identifier
from .chain import Link, interval
from .design import Chain, Edge, Transport, graph_edges, latency
from .proxy import KernelProxy, cascade_timeline
from .specialization import installed_compiler, kernel_specialization, kernel_templates

LEVELS = ('low', 'cycles', 'high')
ARTIFACTS = Path(__file__).parent / 'artifacts'  # one calibrated cost model per generation: <generation>.json


@dataclass(frozen=True)
class Estimate:
    tiles: int
    interval: Optional[Tuple[float, float, float]]  # cycles: low, point, high
    latency: Optional[Tuple[float, float, float]]
    evidence: str
    refusals: Tuple[str, ...]
    clock_mhz: float

    def __str__(self) -> str:
        lines = [f'tiles     {self.tiles}']
        for name, cycles in (('interval', self.interval), ('latency', self.latency)):
            if cycles is not None:
                low, point, high = cycles
                lines.append(
                    f'{name:9s} {point:.0f} cycles ({low:.0f}..{high:.0f}), {point / self.clock_mhz:.3f} us '
                    f'at {self.clock_mhz:.0f} MHz'
                )
        lines += [f'refused   {reason}' for reason in self.refusals]
        return '\n'.join(lines + [f'evidence  {self.evidence}'])


def estimate(model: AIEModel, proxy: Union[KernelProxy, Path, None] = None) -> Estimate:
    """The estimate of a lowered design (aie4ml.from_onnx or aie4ml.from_hls4ml, its pipeline run) by the cost model
    `proxy`: an artifact, its path, or by default the one shipped for the design's generation."""
    if not isinstance(model, AIEModel):
        raise TypeError(
            f'estimate takes an AIEModel (aie4ml.from_onnx, aie4ml.from_hls4ml), not {type(model).__name__}.'
        )
    ctx = model.context
    if not ctx.ir.physical.plan:
        raise ValueError('the design is not lowered yet: run its pipeline (or compile it) first.')
    clock = ctx.device.aie_clock_mhz
    if not isinstance(proxy, KernelProxy):
        path = Path(proxy) if proxy is not None else ARTIFACTS / f'{ctx.device.generation.lower()}.json'
        if not path.exists():
            refusal = f'no cost model is calibrated for {ctx.device.generation} ({path}).'
            return Estimate(_tiles(ctx), None, None, '', (refusal,), clock)
        proxy = KernelProxy.load(path)
    compiler = installed_compiler(ctx.device.generation)
    if compiler is not None and compiler != proxy.compiler:
        warnings.warn(f'the cost model was calibrated with AIE compiler {proxy.compiler}; Vitis here ships {compiler}.')
    return _estimate(ctx, proxy)


def _tiles(ctx) -> int:
    return sum(
        footprint.width * footprint.height
        for inst in ctx.ir.execution
        for footprint in [inst.variant.footprint(inst.node, inst.config)]
    )


def _estimate(ctx, proxy: KernelProxy) -> Estimate:
    ir = ctx.ir
    tiles, clock = _tiles(ctx), ctx.device.aie_clock_mhz
    # Cycles are the generation's: every part of it runs the same cores; its own clocks and ports apply below.
    evidence = f'cost model for {proxy.generation} (calibrated on {proxy.part}), compiler {proxy.compiler}'
    if ctx.device.generation != proxy.generation:
        refusal = f'the cost model was calibrated for {proxy.generation}, not {ctx.device.generation}.'
        return Estimate(tiles, None, None, evidence, (refusal,), clock)
    chains: Dict[str, List[Chain]] = {level: [] for level in LEVELS}
    refusals = []
    for inst in ir.execution:
        spec = kernel_specialization(inst)
        if spec.group is None:
            refusals.append(f'{inst.name}: compile required: {inst.variant.op_type} has no cost descriptor.')
            continue
        sources = _templates(spec.kernel, spec.parameters)
        costs = {
            role: proxy.cost(spec.variant_id, sources, role, spec.group, spec.features) for role in set(spec.roles)
        }
        refused = sorted({cost.refusal for cost in costs.values() if cost.refusal})
        if refused:
            refusals += [f'{inst.name}: {reason}' for reason in refused]
            continue
        graph = sanitize_identifier(inst.name)
        for chain in range(spec.chains):
            names = [f'{graph}.kk[{chain * spec.length + position}]' for position in range(spec.length)]
            kernels = [costs[role] for role in spec.roles]
            for level in LEVELS:
                timelines = [
                    [(getattr(cost, level), 'end')]
                    if role == 'single'
                    else cascade_timeline(getattr(cost, level), *cost.cascade, role != 'first', role != 'last')
                    for cost, role in zip(kernels, spec.roles)
                ]
                chains[level].append(Chain(names, timelines, [cost.wrapper for cost in kernels]))
    if refusals:
        return Estimate(tiles, None, None, evidence, tuple(refusals), clock)
    try:
        handoffs = graph_edges(ir.execution, ir.physical)
    except NotImplementedError as error:
        return Estimate(tiles, None, None, evidence, (f'compile required: {error}',), clock)
    device = ctx.device
    if device.cascade_words_in_flight is None and any(len(c.blocks) > 1 for c in chains['cycles']):
        return Estimate(
            tiles, None, None, evidence, (f'{device.generation}: no cascade FIFO depth to time chains by.',)
        )
    link = Link(proxy.link_latency, device.cascade_words_in_flight or 0)
    stream = device.stream_switch_width_bits / 8
    boundary = min(device.plio_width_bits / 8 * device.pl_clock_mhz / device.aie_clock_mhz, stream)
    transport = Transport(stream, boundary, **proxy.transport)
    intervals = tuple(_interval(chains[level], handoffs, link, transport) for level in LEVELS)
    latencies = tuple(latency(chains[level], handoffs, link, transport) for level in LEVELS)
    return Estimate(tiles, intervals, latencies, evidence, (), clock)


def _interval(chains: Sequence[Chain], handoffs: Sequence[Edge], link: Link, transport: Transport) -> float:
    """The slowest chain's cycles per call, or the slowest handoff a DMA copies."""
    paces = [
        interval(c.timelines, c.wrappers, link) if len(c.blocks) > 1 else c.timelines[0][0][0] + c.wrappers[0]
        for c in chains
    ]
    copies = [
        edge.bytes / (transport.bytes_per_cycle if edge.source and edge.target else transport.boundary_bytes_per_cycle)
        for edge in handoffs
        if not edge.shared
    ]
    return max(paces + copies)


@lru_cache(maxsize=None)
def _templates(kernel: str, parameters: str) -> str:
    return kernel_templates(kernel, parameters)
