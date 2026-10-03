"""Choose every layer's parallelism, and the row height of its microtile blocks, for the whole graph at once (AIEConfig
`Optimize`), before resolution.

Resolution and the transport classifier stay the only judges of legality; user directives are constraints and are
never rewritten -- the choice lives in `ctx.ir.optimizer`, which Resolve and placement read. Designs are ranked by
an estimate of their interval (the slowest kernel or leg) and latency (the critical path), in cycles (`estimate`):
'performance' takes the lowest interval, then latency -- each to within INTERVAL_TOLERANCE, which the estimate cannot
tell apart -- then the fewest memory-tile legs and tiles; 'latency' the lowest latency, then interval, likewise;
'resource' the fewest tiles, then memory-tile legs, then the lowest latency.

It takes the layers in an order that keeps few tensors alive, since a state holds a layout per live tensor.
'performance' first finds the lowest interval a design within the budget reaches, keeping only each state's
fewest-tile design, then searches the designs within it. The search is bounded, not exhaustive: per state it keeps
the designs no other beats on tiles, latency and memory-tile legs, at most MAX_PARTIALS designs per layer, and builds
at most MAX_PLACEMENT_TRIALS, so it reports a search limit unless it discarded nothing. A design counts the tiles its
input buffers keep beside its kernels against the array, so it proposes none the array cannot hold.

A hand-off is direct only where producer and consumer write and read the same blocks, so the block height is chosen
for the graph, not per layer: one search with every layer's own microtile, then one per height the layers offer
(`_block_heights`), each layer offering it taking it; the best design of them all wins.
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
from .transport.memtile import check_memtile_leg
from .transport.routing import BELOW, port_tiles, route_overflow

log = logging.getLogger(__name__)

MODES = ('performance', 'latency', 'resource')
MAX_PARTIALS = 50_000  # partial designs kept past a layer; beyond it the search narrows
MAX_PLACEMENT_TRIALS = 16
INTERVAL_TOLERANCE = 0.1  # cycle estimates this close the search cannot tell apart


class SearchLimit(RuntimeError):
    """The search discarded or could not build designs within its limits; a design may still exist."""


@dataclass(frozen=True)
class _Design:
    """The layers resolved so far: what they cost, the instance chosen for each, and which of their outputs later
    layers still read."""

    memtile: int
    tiles: int
    reserved: int  # tiles beside the kernels that their input buffers keep every other op off (`_reserved_tiles`)
    interval: int  # estimated cycles: the slowest kernel or leg so far
    latency: int  # estimated cycles: the last kernel or graph output so far to finish
    chosen: Tuple[Tuple[str, Any, Any], ...]  # (layer, instance, the search's choice or None)
    live: Tuple[Tuple[str, Any], ...]  # (tensor, producing instance)
    ready: Tuple[Tuple[str, int], ...]  # (live tensor, estimated cycle it is written by)


def _step(cycles: int) -> int:
    """`cycles` on a geometric grid INTERVAL_TOLERANCE apart."""
    return int(math.log1p(cycles) / math.log1p(INTERVAL_TOLERANCE))


def _reserved_tiles(inst, shared_ports: set) -> int:
    """Tiles outside the kernel's footprint whose memory its input buffers pin, less those holding a buffer it may
    share with its producer: placement keeps every other op's tiles and buffers off them (`placement`'s conflict
    rule), so they are taken as surely as the footprint. The fewer over the anchor rows' parities (AIE1 mirrors
    its banks on odd rows)."""
    footprint = inst.variant.footprint(inst.node, inst.config)
    inputs = {binding.group for binding in inst.ports.inputs.values()}
    counts = []
    for anchor_row in (0, 1):
        outside, excused = set(), set()
        for loc in inst.variant.buffer_locations(inst.node, inst.config, anchor_row):
            col, row = loc.rel_col, loc.rel_row
            if loc.port_group not in inputs or (0 <= col < footprint.width and 0 <= row < footprint.height):
                continue
            if not (-1 <= col <= footprint.width and -1 <= row <= footprint.height):
                raise RuntimeError(
                    f'{inst.name}: buffer {loc.port_group}[{loc.port}] pinned at ({col}, {row}), beyond the tiles next '
                    'to its footprint that the design search accounts for.'
                )
            outside.add((col, row))
            if (loc.port_group, loc.port) in shared_ports:
                excused.add((col, row))
        counts.append(len(outside - excused))
    return min(counts)


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


def _block_heights(ctx) -> List[int]:
    """The row heights the layers' microtile candidates offer. A hand-off is direct only where producer and consumer
    share a block, so besides the layers' own microtiles, the search tries each height shared by every layer that
    offers it, one search apiece, rather than every layer's microtile apart, which multiplies its states."""
    registry = get_family_resolver_registry()
    return sorted(
        {
            option['microtile_m']
            for node in ctx.ir.logical
            if not node.is_folded_view
            for option in registry.get(node.op_type).microtiling_candidates(node, ctx.device)
        }
    )


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
        found, refusal, trials = None, None, 0
        for height in (None, *_block_heights(ctx)):
            search = _Search(ctx, mode, max_tiles, height)
            if height is not None and search.microtiling == {}:
                continue  # no layer offers this height: the pass would repeat the first
            try:
                design, built, tried = search.best_buildable()
            except (ConfigRefused, SearchLimit) as failure:
                refusal = refusal or failure
                continue
            trials += tried
            if found is None or search.rank(design) < found[0].rank(found[1]):
                found = (search, design, built)
        if found is None:
            raise refusal
        search, design, built = found
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
            **search.choices(design),
            # The trial already placed the design: pinning its places spares the placer a second search.
            'placement': {
                name: {key: placed[key] for key in ('col', 'row')}
                for name, placed in built.ir.physical.placements.items()
            },
        }
        log.info('choose_parallelism: %s', ctx.ir.optimizer)
        return True


class _Search:
    def __init__(self, ctx, mode: str, max_tiles: int, height: Optional[int] = None):
        self.ctx = ctx
        self.mode = mode
        self.max_tiles = max_tiles
        device = ctx.device
        # the tiles kernels and their buffers may take: the placement region, and the column west and row south of
        # it where the device has them, which only buffers may use
        columns = int(device.columns - max(0, device.column_start - 1))
        self.room = columns * int(device.rows - max(0, device.row_start - 1))
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
        self.choosing = set()  # the layers the search makes a choice for; the rest resolve as they stand
        self.microtiling: Dict[str, Dict[str, int]] = {}  # the layers this pass gives the shared block height
        for node in self.layers:
            family = registry.get(node.op_type)
            # A user's directive constrains the candidates to those agreeing with it; one that agrees with none is
            # resolved as given, so resolution says what is wrong with it.
            asked = dict(node.directives.get('parallelism') or {})
            parallelisms = [
                {'parallelism': option}
                for option in family.parallelism_candidates(node, ctx.device)
                if all(option.get(key) == value for key, value in asked.items())
                # a candidate whose own kernels are past the budget cannot be part of a design
                and int(option['cas_num']) * int(option['cas_length']) <= max_tiles
            ]
            pinned = node.directives.get('microtiling')
            microtiling = next(
                (
                    option
                    for option in family.microtiling_candidates(node, ctx.device)
                    if option['microtile_m'] == height and pinned in (None, option)
                ),
                None,
            )
            if microtiling is not None:
                self.microtiling[node.name] = microtiling
            if parallelisms or microtiling:
                self.choosing.add(node.name)
                extra = {'microtiling': microtiling} if microtiling else {}
                self.options[node.name] = [{**p, **extra} for p in parallelisms or [{}]]
            else:
                self.options[node.name] = [None]
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
        """(memtile, arrivals, departure, reserved) for `node`'s kernels: how many of their legs a memory tile
        carries, the (source tensor, latency, interval) legs into each kernel, the (latency, interval) its
        graph-boundary outputs take to leave, and the tiles their input buffers reserve (`_reserved_tiles`) -- or
        None when the transport classifier refuses a leg."""
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
            may_share = defaultdict(set)  # kernel -> its input ports a producer's buffer may coincide with
            below = defaultdict(dict)  # kernel -> its streams from or to below (a memory tile or PLIO), at (0, 0)

            def cost(leg):
                nonlocal memtile
                realization = classify_connection(leg, execution, self.ctx.device).realization
                if realization == 'memtile':
                    check_memtile_leg(leg, execution, self.ctx.device)
                    memtile += 1
                if leg.consumer is not None and (realization == 'memtile' or leg.producer.node is None):
                    sink = execution.get(leg.consumer.node.name)
                    binding = sink.ports.inputs[leg.consumer.tensor]
                    for port in leg.consumer.selected_ports(binding.count):
                        tiles = port_tiles(sink, 'inputs', binding.group, port, 0, 0)
                        below[sink.name][(binding.group, port)] = (BELOW, tuple(tiles))
                if leg.producer.node is not None and (realization == 'memtile' or leg.consumer is None):
                    source = execution.get(leg.producer.node.name)
                    binding = source.ports.outputs[leg.producer.tensor]
                    for port in leg.producer.selected_ports(binding.count):
                        (tile,) = port_tiles(source, 'outputs', binding.group, port, 0, 0)
                        below[source.name][(binding.group, port)] = (tile, (BELOW,))
                readers = self.readers.get(leg.producer.tensor, 0)
                shared = shareable(leg, realization, execution, readers)
                # Placement shares a direct leg's buffer unless its producer's ports are surely read by another
                # leg too: a whole tensor read by several layers, or a graph output.
                if (
                    leg.consumer is not None
                    and realization == 'direct'
                    and leg.producer.node is not None
                    and leg.producer.tensor not in self.graph_outputs
                    and (readers == 1 or leg.consumer.tensor != leg.producer.tensor)
                ):
                    sink = execution.get(leg.consumer.node.name)
                    ports = leg.consumer.selected_ports(sink.ports.inputs[leg.consumer.tensor].count)
                    may_share[sink.name].update((leg.consumer.group, port) for port in ports)
                return (leg.producer.tensor, *leg_cycles(leg, realization, shared, execution, self.ctx.device))

            try:
                arrivals = tuple(tuple(cost(leg) for leg in collector.input_connections(kernel)) for kernel in kernels)
                leaving = [cost(leg)[1:] for leg in collector.output_connections(kernels[-1], self.last_read)]
                departure = tuple(max(cc, default=0) for cc in zip(*leaving)) or (0, 0)
                reserved = sum(_reserved_tiles(kernel, may_share[kernel.name]) for kernel in kernels)
                for name, streams in below.items():
                    problem = route_overflow(streams, self.ctx.device.stream_switch_ports)
                    if problem:
                        raise ConfigRefused(f'{name}: its memory-tile and PLIO streams alone: {problem}.')
                self._legs_cache[key] = (memtile, arrivals, departure, reserved)
            except ConfigRefused as refusal:
                self.refusals[node.name][str(refusal)] += 1
                self._legs_cache[key] = None
        return self._legs_cache[key]

    # -- the dynamic program ------------------------------------------------------------------------

    def designs(self, bound: Optional[int] = None, fewest: bool = False) -> List[_Design]:
        """The complete designs whose estimated interval stays within `bound`, as the program keeps them; with
        `fewest`, only each state's designs no other beats on both tiles and area, which tells whether any design
        completes within the budget and the array."""
        frontier = {(): [_Design(0, 0, 0, 0, 0, (), (), ())]}
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
                    memtile, arrivals, departure, reserved = legs
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
                        if total > self.max_tiles or total + design.reserved + reserved > self.room:
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
                                design.reserved + reserved,
                                max(design.interval, slowest),
                                max(design.latency, finish + departure[0]),
                                design.chosen + ((node.name, inst, option),),
                                tuple(kept.items()),
                                tuple((t, ready[t]) for t in kept),
                            )
                        )
            frontier = {state: self._kept(found, fewest) for state, found in grown.items()}
            if sum(len(found) for found in frontier.values()) > MAX_PARTIALS:
                frontier = self._narrow(node, frontier)
        return [design for found in frontier.values() for design in found]

    def _narrow(self, node: OpNode, frontier: Dict[Any, List[_Design]]) -> Dict[Any, List[_Design]]:
        """MAX_PARTIALS of the designs: each state's designs no other beats on tiles and area, of which one can be
        completed within the budget and the array whenever any of its state's can, so narrowing loses no feasible
        design, then the best of the others."""
        if len(frontier) > MAX_PARTIALS:
            raise SearchLimit(
                f'choose_parallelism: over {MAX_PARTIALS} search states at {node.name}; give some layers a '
                'parallelism directive to narrow the search.'
            )
        self.narrowed.add(node.name)
        self.truncated = True
        kept = {state: self._kept(found, fewest=True) for state, found in frontier.items()}
        others = sorted(
            (
                (state, design)
                for state, found in frontier.items()
                for design in found
                if all(design is not k for k in kept[state])
            ),
            key=lambda item: self.rank(item[1]),
        )
        for state, design in others[: max(0, MAX_PARTIALS - sum(map(len, kept.values())))]:
            kept[state].append(design)
        return kept

    def _kept(self, designs: List[_Design], fewest: bool) -> List[_Design]:
        """The designs of one state the search carries on: those no other beats on tiles, area (tiles and reserved
        tiles), latency and memory-tile legs at once, latencies to within INTERVAL_TOLERANCE -- or with `fewest`, on
        tiles and area alone. Designs sharing a state have the same futures, so this keeps every trade of tiles and
        area spent here against what later layers need, and a direct design beside a cheaper one through memory
        tiles: one beaten on all of them only completes as a design beaten on all of them, and with `fewest` some
        design kept completes within the budget and the array whenever any does. Intervals need no such trade:
        `best_buildable` bounds them. One design per point of the front: ties placement refuses alike would spend
        its trials on one failure."""
        ordered = sorted(
            designs, key=lambda d: (d.tiles, d.tiles + d.reserved, _step(d.latency), d.memtile, *self.rank(d))
        )
        kept: List[Tuple[Tuple[int, ...], _Design]] = []
        for design in ordered:
            area = design.tiles + design.reserved
            point = (area,) if fewest else (area, _step(design.latency), design.memtile)
            if not any(all(a <= b for a, b in zip(other, point)) for other, _ in kept):
                kept.append((point, design))
        self.truncated |= not fewest and len(designs) > len(kept)
        return [design for _, design in kept]

    def rank(self, design: _Design) -> tuple:
        if self.mode == 'resource':
            return (design.tiles, design.memtile, design.latency, design.interval)
        # intervals and latencies, within INTERVAL_TOLERANCE of each other the estimate cannot tell apart: of those,
        # the fewest memory-tile legs, then the fewest tiles
        first, second = (
            (design.latency, design.interval) if self.mode == 'latency' else (design.interval, design.latency)
        )
        return (_step(first), _step(second), design.memtile, design.tiles, first, second)

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
            compared.update(json.dumps(self.choices(design), sort_keys=True) for design in complete)
            self.compared = len(compared)
            for design in sorted(complete, key=self.rank):
                key = json.dumps(self.choices(design), sort_keys=True)
                if key in tried:
                    continue
                tried.add(key)
                built = self._build(design)
                if not isinstance(built, str):
                    return design, built, len(tried)
                unbuilt[built] += 1
                if len(tried) == MAX_PLACEMENT_TRIALS:
                    raise SearchLimit(
                        self._failure(f'search limit: the {len(tried)} best designs cannot be built', unbuilt)
                    )
        if not tried:
            raise ConfigRefused(self._failure('no design resolves within the tile budget', unbuilt))
        if not self.truncated:
            raise ConfigRefused(
                self._failure(f'none of the {len(tried)} designs within the tile budget can be built', unbuilt)
            )
        raise SearchLimit(self._failure(f'search limit: none of the {len(tried)} designs kept can be built', unbuilt))

    def choices(self, design: _Design) -> Dict[str, Dict[str, Any]]:
        """Per directive the search chooses, what the design resolved each chosen layer under: its parallelism as
        resolved, its microtiling as chosen."""
        made: Dict[str, Dict[str, Any]] = {'parallelism': {}, 'microtiling': {}}
        for name, inst, option in design.chosen:
            if name in self.choosing:
                made['parallelism'][name] = asdict(inst.config.parallelism)
            if option and 'microtiling' in option:
                made['microtiling'][name] = option['microtiling']
        return made

    def _build(self, design: _Design):
        """The rest of the pipeline run on a copy of the context with this design: the built copy, or why the
        pipeline refuses it."""
        from ..pipeline import DEFAULT_PIPELINE

        trial = copy.deepcopy(self.ctx)
        trial.ir.optimizer = self.choices(design)
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
