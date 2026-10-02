"""Choose every layer's parallelism for the whole graph at once (AIEConfig `Optimize`), before resolution.

Resolution and the transport classifier stay the only judges of legality; user directives are constraints and are
never rewritten -- the choice lives in `ctx.ir.optimizer`, which Resolve and placement read. Designs are ranked by
an estimate of their interval (the slowest kernel or leg) and latency (the critical path), in cycles (`estimate`):
'performance' takes the lowest interval, then latency -- each to within INTERVAL_TOLERANCE, which the estimate cannot
tell apart -- then the fewest memory-tile legs and tiles; 'resource' the fewest tiles, then memory-tile legs, then the
lowest latency.

It takes the layers in an order that keeps few tensors alive, since a state holds a layout per live tensor.
'performance' first finds the lowest interval a design within the budget reaches, keeping only each state's
fewest-tile design, then searches the designs within it. The search is bounded, not exhaustive: per state it keeps
the designs no other beats on tiles, latency and memory-tile legs, at most MAX_PARTIALS designs per layer, and builds
at most MAX_PLACEMENT_TRIALS, so it reports a search limit unless it discarded nothing.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..ir.graph import ExecutionIR, ExecutionValue, OpNode
from ..op_impls import get_family_resolver_registry
from .base import AIEPass, run_aie_passes
from .estimate import kernel_cycles, leg_cycles, shareable
from .legalize_layouts import convert_inputs
from .resolve import logical_values, output_contracts, resolve_instance
from .transport.classify import classify_connection
from .transport.collect import TransportCollector

log = logging.getLogger(__name__)

MODES = ('performance', 'resource')
MAX_PARTIALS = 50_000  # partial designs kept past a layer; beyond it the search narrows
MAX_PLACEMENT_TRIALS = 16
INTERVAL_TOLERANCE = 0.1  # cycle estimates this close the search cannot tell apart


@dataclass(frozen=True)
class _Design:
    """The layers resolved so far: what they cost, the instance chosen for each, and which of their outputs later
    layers still read."""

    memtile: int
    tiles: int
    interval: int  # estimated cycles: the slowest kernel or leg so far
    latency: int  # estimated cycles: the last kernel or graph output so far to finish
    chosen: Tuple[Tuple[str, Any], ...]  # (layer, instance)
    live: Tuple[Tuple[str, Any], ...]  # (tensor, producing instance)
    ready: Tuple[Tuple[str, int], ...]  # (live tensor, estimated cycle it is written by)


def _step(cycles: int) -> int:
    """`cycles` on a geometric grid INTERVAL_TOLERANCE apart."""
    return int(math.log1p(cycles) / math.log1p(INTERVAL_TOLERANCE))


def _fewest_live(layers: List[OpNode], reads: Dict[str, Tuple[str, ...]]) -> List[OpNode]:
    """The layers in an order their reads allow that keeps few tensors alive between them. A search state holds a
    layout per live tensor, so the states multiply with every tensor alive at once: attention's Q, K and V, taken in
    model order, are all alive before Q.Kᵀ consumes two of them. Next is always the ready layer that adds the fewest
    live tensors less those it reads last, the earliest in model order on a tie."""
    producer = {t.name: node.name for node in layers for t in node.outputs}
    readers = defaultdict(int)  # layers not yet taken that read each tensor
    for node in layers:
        for tensor in set(reads[node.name]):
            readers[tensor] += 1
    order: List[OpNode] = []
    taken: set = set()
    while len(order) < len(layers):
        ready = [
            node
            for node in layers
            if node.name not in taken and all(t not in producer or producer[t] in taken for t in reads[node.name])
        ]

        def growth(node: OpNode) -> int:
            freed = sum(readers[t] == 1 for t in set(reads[node.name]))
            return sum(readers[t.name] > 0 for t in node.outputs) - freed

        node = min(ready, key=growth)
        order.append(node)
        taken.add(node.name)
        for tensor in set(reads[node.name]):
            readers[tensor] -= 1
    return order


class ChooseParallelism(AIEPass):
    def __init__(self):
        self.name = 'choose_parallelism'

    def transform(self, model_or_ctx) -> bool:
        ctx = get_backend_context(model_or_ctx)
        mode = ctx.aie_config.get('Optimize', 'resource')
        if mode not in MODES:
            raise ValueError(f'AIEConfig Optimize={mode!r}; expected one of {list(MODES)}.')
        device = ctx.device
        max_tiles = int(
            ctx.aie_config.get('MaxTiles', (device.columns - device.column_start) * (device.rows - device.row_start))
        )
        if max_tiles < 1:
            raise ValueError(f'AIEConfig MaxTiles={max_tiles} must be positive.')

        ctx.ir.optimizer = {}  # a previous run's choice must not steer this one
        print(
            f"[aie4ml] Searching for a '{mode}' design within {max_tiles} AIE tiles; this can take a minute...",
            flush=True,
        )
        search = _Search(ctx, mode, max_tiles)
        design, built, trials = search.best_buildable()
        narrowed = f', narrowed after {len(search.narrowed)} layers' if search.narrowed else ''
        print(
            f'[aie4ml] Design found: {design.tiles} AIE tiles, the best of {search.compared} compared{narrowed}.',
            flush=True,
        )
        ctx.ir.optimizer = {
            'mode': mode,
            'max_tiles': max_tiles,
            'tiles': design.tiles,
            'memtile_legs': design.memtile,
            'designs_tried': trials,
            'narrowed_after': sorted(search.narrowed),
            'parallelism': search.parallelism(design),
            # The trial already placed the design: pinning its places spares the placer a second search.
            'placement': {
                name: {key: placed[key] for key in ('col', 'row')}
                for name, placed in built.ir.physical.placements.items()
            },
        }
        log.info('choose_parallelism: %s', ctx.ir.optimizer)
        return True


class _Search:
    def __init__(self, ctx, mode: str, max_tiles: int):
        self.ctx = ctx
        self.mode = mode
        self.max_tiles = max_tiles
        self.values = logical_values(ctx.ir.logical)
        # what each layer's legs read: its inputs, or the sources of a folded view it reads
        self.reads: Dict[str, Tuple[str, ...]] = {}
        layers = [node for node in ctx.ir.logical if not node.is_folded_view]
        for node in layers:
            views = [self.values[t.name].view for t in node.inputs if not t.is_parameter]
            inputs = [t.name for t in node.inputs if not t.is_parameter]
            self.reads[node.name] = tuple(
                source for name, view in zip(inputs, views) for source in (view.sources if view else (name,))
            )
        self.layers: List[OpNode] = _fewest_live(layers, self.reads)
        registry = get_family_resolver_registry()
        self.options: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self.choosing = set()  # the layers the search picks a parallelism for; the rest resolve as they stand
        for node in self.layers:
            # A user's parallelism directive constrains the candidates to those agreeing with it; one that agrees
            # with none is resolved as given, so resolution says what is wrong with it.
            asked = dict(node.directives.get('parallelism') or {})
            offered = [
                option
                for option in registry.get(node.op_type).parallelism_candidates(node, ctx.device)
                if all(option.get(key) == value for key, value in asked.items())
                # a candidate whose own kernels are past the budget cannot be part of a design
                and int(option['cas_num']) * int(option['cas_length']) <= max_tiles
            ]
            if offered:
                self.choosing.add(node.name)
            self.options[node.name] = offered or [None]
        self.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
        self.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
        # after which layer each tensor is read no more: a tensor stays in the state until then
        self.last_read: Dict[str, int] = {
            tensor: position for position, node in enumerate(self.layers) for tensor in self.reads[node.name]
        }
        # how many layers read each tensor: a tensor more than one reads is copied to each, never shared
        self.readers: Dict[str, int] = defaultdict(int)
        for tensors in self.reads.values():
            for tensor in set(tensors):
                self.readers[tensor] += 1
        self.refusals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.intervals: set = set()  # every layer's estimated interval met, the bounds a performance search tries
        self.truncated = False  # whether the program has discarded a design since last reset
        self.narrowed: set = set()  # the layers past which the search kept only MAX_PARTIALS designs
        self._resolved: Dict[Any, Any] = {}
        self._legs_cache: Dict[Any, Any] = {}
        self.compared = 0  # distinct complete designs within the budget that best_buildable ranked
        self._contracts: Dict[int, Dict[str, Any]] = {}
        self._signatures: Dict[Tuple[int, str], str] = {}

    # -- what an instance publishes -----------------------------------------------------------------

    def _output_contracts(self, inst) -> Dict[str, Any]:
        if id(inst) not in self._contracts:
            self._contracts[id(inst)] = output_contracts(inst)
        return self._contracts[id(inst)]

    def _signature(self, inst, tensor: str) -> str:
        """What resolution and transport read of `tensor`'s producer: its ports, route, view, published contract
        and each port's staging."""
        key = (id(inst), tensor)
        if key not in self._signatures:
            binding = inst.ports.outputs[tensor]
            contract = self._output_contracts(inst).get(tensor)
            stagings = [
                inst.variant.describe_output_staging(inst.node, inst.config, tensor, port)
                for port in range(int(binding.count))
            ]
            self._signatures[key] = json.dumps(
                [
                    repr(binding),
                    inst.io_route.get('outputs', {}).get(tensor),
                    repr(inst.port_views[tensor]),
                    None if contract is None else (contract.contract, contract.port_staging, contract.inner_shards),
                    stagings,
                ],
                sort_keys=True,
                default=str,
            )
        return self._signatures[key]

    # -- one layer and its legs ----------------------------------------------------------------------

    def _resolve(self, node: OpNode, option, live: Dict[str, Any]):
        """(kernels, tiles, cycles) for `node` under `option` -- the layout converters it needs, then its own
        instance, and each one's estimated cycles -- or None when resolution refuses it; per input arrival."""
        inputs = {t.name: live[t.name] for t in node.inputs if t.name in live}
        arrival = tuple((t, self._signature(i, t)) for t, i in inputs.items())
        key = (node.name, json.dumps(option, sort_keys=True), arrival)
        if key in self._resolved:
            return self._resolved[key]
        contracts = {t: self._output_contracts(i)[t] for t, i in inputs.items() if t in self._output_contracts(i)}
        try:
            inst = resolve_instance(node, self.ctx.device, contracts, option)
            kernels = convert_inputs(
                inst, {item.tensor: self.values[item.tensor] for item in inst.inputs}, self.ctx.device
            )
            for converter in kernels[:-1]:
                target = converter.outputs[0]
                self.values.setdefault(target, ExecutionValue(target, producer=converter.name))
            tiles = sum(self._area(kernel.variant.footprint(kernel.node, kernel.config)) for kernel in kernels)
            work = inst.variant.work(node, inst.config) if node.name in self.choosing else None
            cycles = tuple(kernel_cycles(k, work if k is inst else None, self.ctx.device) for k in kernels)
            resolved = (tuple(kernels), tiles, cycles)
        except ConfigRefused as refusal:
            self.refusals[node.name][str(refusal)] += 1
            resolved = None
        self._resolved[key] = resolved
        return resolved

    @staticmethod
    def _area(footprint) -> int:
        return footprint.width * footprint.height

    def _legs(self, node: OpNode, kernels, live: Dict[str, Any]):
        """(memtile, arrivals, departure) for `node`'s kernels: how many of their legs a memory tile carries, the
        (source tensor, latency, interval) legs into each kernel, and the (latency, interval) its graph-boundary
        outputs take to leave -- or None when the transport classifier refuses a leg."""
        producers = {id(live[t]): live[t] for t in self.reads[node.name] if t in live}
        key = (id(kernels[-1]), tuple(sorted(producers)))
        if key not in self._legs_cache:
            instances = [*producers.values(), *kernels]
            execution = ExecutionIR(
                instances={inst.name: inst for inst in instances},
                tensor_contracts={t: c for inst in instances for t, c in self._output_contracts(inst).items()},
                values=self.values,
                graph_inputs=self.graph_inputs,
                graph_outputs=self.graph_outputs,
            )
            collector = TransportCollector(execution)
            memtile = 0

            def cost(leg):
                nonlocal memtile
                realization = classify_connection(leg, execution, self.ctx.device).realization
                memtile += realization == 'memtile'
                readers = self.readers.get(leg.producer.tensor, 0)
                shared = shareable(leg, realization, execution, readers)
                return (leg.producer.tensor, *leg_cycles(leg, realization, shared, execution, self.ctx.device))

            try:
                arrivals = tuple(tuple(cost(leg) for leg in collector.input_connections(kernel)) for kernel in kernels)
                leaving = [cost(leg)[1:] for leg in collector.output_connections(kernels[-1], self.last_read)]
                departure = tuple(max(cc, default=0) for cc in zip(*leaving)) or (0, 0)
                self._legs_cache[key] = (memtile, arrivals, departure)
            except ConfigRefused as refusal:
                self.refusals[node.name][str(refusal)] += 1
                self._legs_cache[key] = None
        return self._legs_cache[key]

    # -- the dynamic program ------------------------------------------------------------------------

    def designs(self, bound: Optional[int] = None, fewest: bool = False) -> List[_Design]:
        """The complete designs whose estimated interval stays within `bound`, as the program keeps them; with
        `fewest`, only each state's fewest-tile one, which tells whether any design completes within the budget."""
        frontier = {(): [_Design(0, 0, 0, 0, (), (), ())]}
        for position, node in enumerate(self.layers):
            grown: Dict[Any, List[_Design]] = defaultdict(list)
            for designs in frontier.values():
                live = dict(designs[0].live)  # every design of a state sees the same arrivals
                for option in self.options[node.name]:
                    resolved = self._resolve(node, option, live)
                    if resolved is None:
                        continue
                    kernels, tiles, cycles = resolved
                    legs = self._legs(node, kernels, live)
                    if legs is None:
                        continue
                    memtile, arrivals, departure = legs
                    slowest = max(*cycles, *(cc for legs_in in arrivals for *_, cc in legs_in), departure[1])
                    self.intervals.add(slowest)
                    if bound is not None and slowest > bound:
                        continue
                    inst = kernels[-1]
                    kept = {t: i for t, i in live.items() if self.last_read[t] > position}
                    kept.update({t: inst for t in inst.outputs if self.last_read.get(t, -1) > position})
                    state = tuple(sorted((t, self._signature(i, t)) for t, i in kept.items()))
                    for design in designs:
                        total = design.tiles + tiles
                        if total > self.max_tiles:
                            continue
                        ready = dict(design.ready)
                        for kernel, legs_in, kernel_cc in zip(kernels, arrivals, cycles):
                            start = max((0 if src in self.graph_inputs else ready[src]) + cc for src, cc, _ in legs_in)
                            ready.update({t: start + kernel_cc for t in kernel.outputs})
                        finish = max(ready[t] for t in inst.outputs)
                        grown[state].append(
                            _Design(
                                design.memtile + memtile,
                                total,
                                max(design.interval, slowest),
                                max(design.latency, finish + departure[0]),
                                design.chosen + ((node.name, inst),),
                                tuple(kept.items()),
                                tuple((t, ready[t]) for t in kept),
                            )
                        )
            frontier = {state: self._kept(found, fewest) for state, found in grown.items()}
            if sum(len(found) for found in frontier.values()) > MAX_PARTIALS:
                frontier = self._narrow(node, frontier)
        return [design for found in frontier.values() for design in found]

    def _narrow(self, node: OpNode, frontier: Dict[Any, List[_Design]]) -> Dict[Any, List[_Design]]:
        """MAX_PARTIALS of the designs: each state's fewest-tile one, which can be completed within the budget
        whenever any of its state's can, so narrowing loses no feasible design, then the best of the others."""
        if len(frontier) > MAX_PARTIALS:
            raise RuntimeError(
                f'choose_parallelism: over {MAX_PARTIALS} search states at {node.name}; give some layers a '
                'parallelism directive to narrow the search.'
            )
        self.narrowed.add(node.name)
        self.truncated = True
        kept = {state: [min(found, key=lambda design: design.tiles)] for state, found in frontier.items()}
        others = sorted(
            ((state, design) for state, found in frontier.items() for design in found if design is not kept[state][0]),
            key=lambda item: self._rank(item[1]),
        )
        for state, design in others[: MAX_PARTIALS - len(kept)]:
            kept[state].append(design)
        return kept

    def _kept(self, designs: List[_Design], fewest: bool) -> List[_Design]:
        """The designs of one state the search carries on: those no other beats on tiles, latency and memory-tile
        legs at once, latencies to within INTERVAL_TOLERANCE, or with `fewest` the fewest-tile one alone. Designs
        sharing a state have the same futures, so this keeps every trade of tiles spent here against tiles left for
        later layers, and a direct design beside a cheaper one through memory tiles: one beaten on all three only
        completes as a design beaten on all three, and the fewest-tile one completes within the budget whenever any
        does. Intervals need no such trade: `best_buildable` bounds them. One design per point of the front: ties
        placement refuses alike would spend its trials on one failure."""
        ordered = sorted(designs, key=lambda d: (d.tiles, _step(d.latency), d.memtile, *self._rank(d)))
        if fewest:
            return ordered[:1]
        kept: List[Tuple[int, int, _Design]] = []
        for design in ordered:
            latency = _step(design.latency)
            if not any(other <= latency and memtile <= design.memtile for other, memtile, _ in kept):
                kept.append((latency, design.memtile, design))
        self.truncated |= len(designs) > len(kept)
        return [design for *_, design in kept]

    def _rank(self, design: _Design) -> tuple:
        if self.mode == 'resource':
            return (design.tiles, design.memtile, design.latency, design.interval)
        # intervals, then latencies, within INTERVAL_TOLERANCE of each other the estimate cannot tell apart: of
        # those, the fewest memory-tile legs, then the fewest tiles
        return (
            _step(design.interval),
            _step(design.latency),
            design.memtile,
            design.tiles,
            design.interval,
            design.latency,
        )

    def best_buildable(self) -> Tuple[_Design, Any, int]:
        """The best-ranked design the rest of the pipeline builds, the context it was built in, and how many
        designs were tried."""
        bounds: List[Optional[int]] = [None]
        if self.mode == 'performance':
            # each grid step's widest interval met, from the lowest step any design completes within on
            bounds = []
            if self.designs(fewest=True):  # also resolves every option it meets
                widest = {}
                for interval in self.intervals:
                    widest[_step(interval)] = max(interval, widest.get(_step(interval), 0))
                bounds = [widest[step] for step in sorted(widest)]
                low, high = 0, len(bounds) - 1
                while low < high:
                    middle = (low + high) // 2
                    low, high = (low, middle) if self.designs(bounds[middle], fewest=True) else (middle + 1, high)
                bounds = bounds[low:]
        tried, compared, unbuilt = set(), set(), defaultdict(int)
        self.truncated = False
        for bound in bounds:
            complete = self.designs(bound)
            compared.update(json.dumps(self.parallelism(design), sort_keys=True) for design in complete)
            self.compared = len(compared)
            for design in sorted(complete, key=self._rank):
                key = json.dumps(self.parallelism(design), sort_keys=True)
                if key in tried:
                    continue
                tried.add(key)
                built = self._build(design)
                if not isinstance(built, str):
                    return design, built, len(tried)
                unbuilt[built] += 1
                if len(tried) == MAX_PLACEMENT_TRIALS:
                    raise RuntimeError(
                        self._failure(f'search limit: the {len(tried)} best designs cannot be built', unbuilt)
                    )
        if not tried:
            raise ConfigRefused(self._failure('no design resolves within the tile budget', unbuilt))
        if not self.truncated:
            raise ConfigRefused(
                self._failure(f'none of the {len(tried)} designs within the tile budget can be built', unbuilt)
            )
        raise RuntimeError(self._failure(f'search limit: none of the {len(tried)} designs kept can be built', unbuilt))

    def parallelism(self, design: _Design) -> Dict[str, Dict[str, Any]]:
        """The parallelism the design resolved each chosen layer to."""
        return {name: asdict(inst.config.parallelism) for name, inst in design.chosen if name in self.choosing}

    def _build(self, design: _Design):
        """The rest of the pipeline run on a copy of the context with this design: the built copy, or why the
        pipeline refuses it."""
        from ..pipeline import DEFAULT_PIPELINE

        trial = copy.deepcopy(self.ctx)
        trial.ir.optimizer = {'parallelism': self.parallelism(design)}
        rest = DEFAULT_PIPELINE[DEFAULT_PIPELINE.index(ChooseParallelism) + 1 :]
        try:
            run_aie_passes(trial, [cls() for cls in rest])
        except ConfigRefused as refusal:
            return str(refusal)
        return trial

    def _failure(self, what: str, unbuilt: Dict[str, int]) -> str:
        lines = [f'choose_parallelism ({self.mode}, MaxTiles={self.max_tiles}): {what}.']
        if what.startswith('search limit'):
            lines.append('  A design may still exist: give some layers a parallelism directive to narrow the search.')
        for name, messages in self.refusals.items():
            for message, count in sorted(messages.items(), key=lambda item: -item[1])[:3]:
                lines.append(f'  {name}: {count} option(s) refused: {message}')
        lines += [f'  {count} design(s) not built: {message}' for message, count in unbuilt.items()]
        return '\n'.join(lines)
