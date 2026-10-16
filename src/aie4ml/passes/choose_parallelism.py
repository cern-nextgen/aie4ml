"""Choose every layer's parallelism, the row height of its microtile blocks and its fallback kernel, for the whole
graph at once (AIEConfig `Optimize`), before resolution.

Resolution and the transport classifier stay the only judges of legality; user directives are constraints and are
never rewritten -- the choice lives in `ctx.ir.optimizer`, which Resolve and placement read. Designs are ranked by
the mode's ranking (`ranking.RANKING`) of an estimate of their interval (the slowest kernel or leg) and latency (the
critical path), in cycles (`estimate`), their memory-tile legs, hand-overs a DMA copies, and tiles.

The search is exact. A layer's alternatives are an option and the layouts of the tensors it reads -- layouts whose
legs into it cost alike counted once. One constraint program (OR-Tools CP-SAT) picks an alternative per layer, each
reading a layout its producer's alternative publishes, ranked as above; it runs deterministically, so one model
always yields one design. Which hand-overs are one shared buffer is placement's to decide, where the geometry is
known, so the search ranks each design by its bound: every leg that some placement can share taken as shared. The
rest of the pipeline builds designs in the order of their bounds, placement ranking each one's placements by the same
ranking (`PhysicalIR.estimate`), until the next bound ranks no better than the best design built: that one is the best
there is, unless a solve ran out of time or the builds reached AIEConfig SearchBuilds, which
`ctx.ir.optimizer['proven']` records. A bound sees no conflict between hand-overs some placement shares each, which
placement alone can tell, so where placement shares fewer than the bound the builds bound the effort, not the proof.
A design placement refuses is excluded, with every design placement sees alike.

A hand-off is direct only where producer and consumer write and read the same blocks, so the block height is chosen
for the graph, not per layer: one search with every layer's own microtile, then one per height the layers offer
(`_block_heights`), each layer offering it taking it, which builds a design only where it ranks better than the best
so far.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ortools.sat.python import cp_model

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..ir.graph import ExecutionIR, ExecutionValue, OpNode
from ..op_impls.family_registry import get_family_resolver_registry
from .base import AIEPass, run_aie_passes
from .estimate import kernel_cycles, leg_cycles
from .legalize_layouts import convert_inputs
from .placement import PlacementInfeasibleError, instance_rect, pair_offsets, share_offsets
from .ranking import PLACED, RANKING, optimize_mode, rank, solve_ranked
from .resolve import logical_values, output_contracts, resolve_instance
from .shared_buffer import static_problem
from .transport.classify import classify_connection, direct_unit
from .transport.collect import TransportCollector
from .transport.memtile import check_memtile_leg
from .transport.routing import BELOW, port_tiles, route_overflow

log = logging.getLogger(__name__)

SEARCH_WORKERS = 8


class SearchLimit(RuntimeError):
    """The search could not build any of the designs within its limits; a design may still exist."""


@dataclass(frozen=True)
class _Design:
    """A design: what it costs, and per layer the instance chosen and the search's choice or None."""

    memtile: int
    tiles: int
    reserved: int  # tiles beside the kernels that their input buffers keep every other op off (`_reserved_tiles`)
    interval: int  # estimated cycles: the slowest kernel or leg
    latency: int  # estimated cycles: the last kernel or graph output to finish
    copies: int  # hand-overs a tile DMA copies (`_Alternative.copies`)
    chosen: Tuple[Tuple[str, Any, Any], ...] = ()

    @property
    def fallbacks(self) -> int:
        """How many of its layers resolve with a fallback kernel."""
        return sum(bool(_kernel(option)) for _, _, option in self.chosen)


class _LegFacts(NamedTuple):
    """One leg's part in a layer's cost and placement (`_Search._leg_facts`)."""

    memtile: int  # the memory-tile legs it counts
    streams: Tuple[Tuple[str, Tuple[str, int], Any], ...]  # its streams from or to below: (kernel, port, stream)
    ports: Optional[Tuple[Tuple[str, int], ...]]  # its consumer's ports some placement has read the producer's buffer
    own: Tuple[str, int, int]  # (source tensor, latency, interval) through a buffer of its own
    shared: Optional[Tuple[str, int, int]]  # the same through the producer's buffer, or None where none can
    edge: Optional[Tuple[bool, Tuple[Tuple[int, int], ...]]]  # a direct kernel leg's (shareable, port pairs)
    unshared: int  # its direct port pairs no placement shares, each alone: those a DMA copies at best


class _Placed(NamedTuple):
    """What placement sees of a layer's kernels and legs (`placement._build_graph`): designs alike in it place alike,
    whatever else sets them apart."""

    views: Tuple[Tuple[str, tuple], ...]  # (kernel, its footprint, pins on every anchor row and output groups)
    edges: Tuple[Tuple[str, str, Any, tuple], ...]  # each kernel leg: (producer, consumer, `_LegFacts.edge` or None
    # through a memory tile, (its tensors, groups and whether the consumer requires it shared))
    below: tuple  # their streams from or to below (`_LegFacts.streams`)


class _Arrival(NamedTuple):
    """How a tensor a layer reads from another layer arrives into an alternative's kernels, from a class of the
    producer's layouts whose legs into them are alike (`_Search._classes`), at best: each leg some placement shares
    (`_LegFacts.ports`) taken as shared."""

    layouts: frozenset  # the producer's layouts in the class
    placed: _Placed  # what placement sees of its legs
    facts: tuple  # its legs' (kernel position, `_LegFacts`): arrivals alike in them place and cost alike
    delays: Tuple[Tuple[str, int], ...]  # per output of the layer it reaches, the cycles from the tensor to it
    slowest: int  # its slowest leg's interval
    memtile: int  # its legs through a memory tile
    copies: int  # its port pairs a tile DMA copies at best (`_LegFacts.unshared`)

    @property
    def alike(self) -> tuple:
        return self.placed, self.facts


@dataclass(frozen=True, eq=False)
class _Alternative:
    """One way a layer resolves: its option and kernels, resolved against the contracts of the layouts it reads, and
    what they and their own legs -- from the graph's inputs, between its kernels and to the graph's outputs -- cost
    at best; and per tensor it reads from another layer, the ways it may arrive (`_Arrival`), of which a design takes
    one."""

    option: Any
    kernels: Tuple[Any, ...]
    published: Tuple[Tuple[str, str], ...]  # (output tensor, its layout)
    placed: _Placed  # what placement sees of its kernels and own legs
    tiles: int
    cycles: Tuple[int, ...]
    reserved: int  # at fewest (`_reserved_tiles`): each input port some placement shares excused
    memtile: int  # its own legs through a memory tile
    copies: int  # its own port pairs a tile DMA copies at best (`_LegFacts.unshared`)
    slowest: int  # the slowest of its kernels and own legs
    delays: Tuple[Tuple[str, int], ...]  # per output the graph's inputs reach, the cycles from them to it
    departure: int  # its slowest leg's latency to the graph's outputs
    arrivals: Tuple[Tuple[str, Tuple[_Arrival, ...]], ...]  # per tensor it reads from another layer

    @property
    def alike(self) -> tuple:
        """What placement sees of it and what it costs: alternatives alike in it place and cost alike."""
        costs = (self.tiles, self.cycles, self.reserved, self.memtile, self.copies, self.slowest, self.delays)
        return self.placed, (*costs, self.departure)

    def arriving(self, tensor: str) -> Tuple[_Arrival, ...]:
        return dict(self.arrivals)[tensor]


def _delays(kernels, cycles, legs, origin) -> Tuple[Tuple[str, int], ...]:
    """Per output of the last of `kernels` that `origin` reaches, the cycles from `origin` being written to it being
    written: along `legs` -- per kernel its (source, latency) legs, from `origin` or from a tensor the kernels write --
    and the kernels' `cycles`."""
    ready: Dict[str, int] = {}
    for kernel, into, cc in zip(kernels, legs, cycles):
        starts = [
            (0 if src == origin else ready[src]) + latency for src, latency in into if src == origin or src in ready
        ]
        if starts:
            ready.update(dict.fromkeys(kernel.outputs, max(starts) + cc))
    return tuple((tensor, ready[tensor]) for tensor in kernels[-1].outputs if tensor in ready)


def _at_best(kernel, leg, facts: '_LegFacts') -> Tuple[Optional[str], int, int]:
    """(source, latency, interval) of a leg into `kernel` at best -- through the producer's buffer where some
    placement shares it -- its source None from the graph's inputs; refused where its kernel requires it shared and
    no placement shares it."""
    source = None if leg.producer.node is None else leg.producer.tensor
    if facts.ports is not None:
        return (source, *facts.shared[1:])
    if kernel.input(leg.consumer.tensor).shared_memory:
        raise ConfigRefused(
            f"{kernel.name}: reads {leg.consumer.tensor!r} in its producer's buffer, which no placement can share."
        )
    return (source, *facts.own[1:])


def _placed(item) -> _Placed:
    return item.placed


def _alike(item) -> tuple:
    return item.alike


def _keys(alternatives: Dict[str, List[_Alternative]], picks: Dict[str, tuple], key) -> Dict[str, tuple]:
    """Per layer, the `key` of the alternative picked (`_Search._solve`) and, by tensor, of the arrivals picked."""
    keys = {}
    for name, (i, taken) in picks.items():
        alt = alternatives[name][i]
        keys[name] = (key(alt), {tensor: key(found[taken[tensor]]) for tensor, found in alt.arrivals})
    return keys


def _groups(alternatives: Dict[str, List[_Alternative]], keys: Dict[str, tuple], key) -> List[tuple]:
    """A cut (`_Search._solve`) of every design whose alternatives and arrivals have the `key`s of `keys` (`_keys`):
    per layer, its alternatives of the key, and per tensor they read, their arrivals of the key."""
    groups = []
    for name, (alt_key, arrival_keys) in keys.items():
        same = [j for j, alt in enumerate(alternatives[name]) if key(alt) == alt_key]
        groups.append((name, None, frozenset(same)))
        for tensor, wanted in arrival_keys.items():
            pairs = frozenset(
                (j, c)
                for j in same
                for c, arrival in enumerate(alternatives[name][j].arriving(tensor))
                if key(arrival) == wanted
            )
            groups.append((name, tensor, pairs))
    return groups


def _least(alternatives, cost) -> int:
    """The least `cost` any design takes: each layer's cheapest alternative, summed."""
    return sum(min(cost(alt) for alt in options) for options in alternatives.values())


def _kernel(option) -> Dict[str, Any]:
    """The fallback kernel a search option resolves its layer with, or {} for the kernel resolution prefers."""
    return {key: value for key, value in (option or {}).items() if key not in ('parallelism', 'microtiling')}


def _reserved_tiles(inst, shared_ports: set) -> int:
    """Tiles outside the kernel's footprint whose memory its input buffers pin, less those holding a buffer it reads
    in its producer's: placement keeps every other op's tiles and buffers off them (`placement`'s conflict rule), so
    they are taken as surely as the footprint. The fewer over the anchor rows' parities (AIE1 mirrors its banks on
    odd rows)."""
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


def _block_heights(ctx) -> List[int]:
    """The row heights the layers' microtile candidates offer. A hand-off is direct only where producer and consumer
    share a block, so besides the layers' own microtiles, the search tries each height shared by every layer that
    offers it, one search apiece, rather than every layer's microtile apart, which multiplies its alternatives."""
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
        mode = optimize_mode(ctx)
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
        found, refusal, trials, doubts = None, None, 0, []
        shared = _Shared(logical_values(ctx.ir.logical))
        for height in (None, *_block_heights(ctx)):
            search = _Search(ctx, mode, max_tiles, shared, height)
            if height is not None and search.microtiling == {}:
                continue  # no layer offers this height: the pass would repeat the first
            try:
                better = search.best_buildable(None if found is None else search.rank(found[1]))
            except (ConfigRefused, SearchLimit) as failure:
                refusal = refusal or failure
                if isinstance(failure, SearchLimit):  # a better design of this height may exist
                    doubts.append(str(failure).splitlines()[0])
                better = None
            trials += search.tried
            doubts += search.doubts
            if better is not None:
                found = (search, *better)
        if shared.unplaced_fill is not None:  # the fills the searches halved (`_Search._fill`)
            doubts.append(
                f'designs filling more than {shared.placed_fill} tiles were not all tried: one filling '
                f'{shared.unplaced_fill} found no placement within AIEConfig PlacerTime'
            )
        if found is None:
            raise refusal
        search, design, built = found
        print(
            f'[aie4ml] Design found: {design.tiles} AIE tiles, the best of {trials} built'
            f'{"" if not doubts else ", not proven the best: " + doubts[0]}.',
            flush=True,
        )
        ctx.ir.optimizer = {
            'mode': mode,
            'max_tiles': max_tiles,
            'tiles': design.tiles,
            'memtile_legs': design.memtile,
            'designs_tried': trials,
            'proven': not doubts,
            'doubts': doubts,
            **search.choices(design),
            # The trial ranked the design's placement already: the placer, which consumes these, only lays it out.
            'placement': {
                name: {key: placed[key] for key in ('col', 'row')}
                for name, placed in built.ir.physical.placements.items()
            },
            'estimate': built.ir.physical.estimate,
        }
        log.info('choose_parallelism: %s', ctx.ir.optimizer)
        return True


@dataclass
class _Shared:
    """What a layer, its legs and its refusals resolve to, and the designs placement refused, which no microtile
    height changes: the searches of one pass share it, so each resolves an option it meets again only once, and none
    builds a design placement sees alike to one it refused."""

    values: Dict[str, Any]
    resolved: Dict[Any, Any] = field(default_factory=dict)
    own: Dict[Any, Any] = field(default_factory=dict)  # `_own_legs`, by kernels
    leg: Dict[Any, Any] = field(default_factory=dict)  # `_leg_facts`, by a leg and its two instances
    routes: Dict[Any, Optional[str]] = field(default_factory=dict)
    geometry: Dict[Any, Any] = field(default_factory=dict)  # an instance's `_rect`, `_placed` view and `_port_tiles`
    contracts: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    signatures: Dict[Tuple[int, str], str] = field(default_factory=dict)
    refusals: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    # what placement sees of the designs it refused (`_keys` of `_Placed`): no design alike in all of it is built,
    # whichever microtile height its search is of
    unplaced: List[Dict[str, tuple]] = field(default_factory=list)
    # Near a full array placement cannot tell a design it places in no time from one with no placement: the fill
    # (tiles a design's kernels and reserved buffers take) of the fullest design placed and of the emptiest that found
    # no placement in its time, the fills between which the searches halve (`_Search._fill`).
    placed_fill: int = 0
    unplaced_fill: Optional[int] = None
    built: int = 0  # designs built, by the searches of every block height (AIEConfig SearchBuilds)


class _Search:
    def __init__(self, ctx, mode: str, max_tiles: int, shared: _Shared, height: Optional[int] = None):
        self.ctx = ctx
        self.mode = mode
        self.max_tiles = max_tiles
        # the solver's deterministic time per solve, in its own units (about a second of one worker's search each)
        self.search_time = float(ctx.aie_config.get('SearchTime', 300))
        # the designs the searches of one pass build at most: placement alone tells hand-overs some placement shares
        # each from those it shares together, so a bound may rank many designs better than they place
        self.builds = int(ctx.aie_config.get('SearchBuilds', 4))
        if self.builds < 1:
            raise ValueError(f'AIEConfig SearchBuilds={self.builds} must be positive.')
        device = ctx.device
        # the tiles kernels and their buffers may take: the placement region, and the column west and row south of
        # it where the device has them, which only buffers may use
        columns = int(device.columns - max(0, device.column_start - 1))
        self.room = columns * int(device.rows - max(0, device.row_start - 1))
        self.shared = shared
        self.values = shared.values
        # what each layer's legs read: its inputs, or the sources of a folded view it reads
        self.reads: Dict[str, Tuple[str, ...]] = {}
        # the logical order, producer before consumer
        self.layers: List[OpNode] = [node for node in ctx.ir.logical if not node.is_folded_view]
        for node in self.layers:
            self.reads[node.name] = tuple(
                dict.fromkeys(source for t in node.inputs if not t.is_parameter for source in self._sources(t.name))
            )
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
            fallbacks = family.fallback_kernels(node, ctx.device)
            if parallelisms or microtiling or fallbacks:
                self.choosing.add(node.name)
                extra = {'microtiling': microtiling} if microtiling else {}
                self.options[node.name] = [{**p, **extra, **k} for p in parallelisms or [{}] for k in ({}, *fallbacks)]
            else:
                self.options[node.name] = [None]
        self.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
        self.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
        self.read = {tensor for tensors in self.reads.values() for tensor in tensors}  # every tensor a layer reads
        self.produced = {t.name for node in self.layers for t in node.outputs}  # every tensor a layer writes
        # how many layers read each tensor: a tensor more than one reads is copied to each, never shared
        self.readers: Dict[str, int] = defaultdict(int)
        for tensors in self.reads.values():
            for tensor in tensors:
                self.readers[tensor] += 1
        # each read of a tensor: what it reads it as -- the tensor itself, or a view over it -- and the layer reading
        # it, once per input reading it
        self.read_as: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for node in self.layers:
            for t in node.inputs:
                for source in () if t.is_parameter else self._sources(t.name):
                    self.read_as[source].append((t.name, node.name))
        self.refusals = shared.refusals
        self.unplaced = shared.unplaced
        self.doubts: List[str] = []  # why the design found may not be the best there is
        self.tried = 0  # designs built (`best_buildable`)
        self._resolved = shared.resolved
        self._own_cache = shared.own
        self._leg_cache = shared.leg
        self._route_cache = shared.routes
        self._geometry = shared.geometry
        self._contracts = shared.contracts
        self._signatures = shared.signatures
        # an option's content: the searches share what it resolves to
        self._option_keys = {
            id(option): json.dumps(option, sort_keys=True) for options in self.options.values() for option in options
        }

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
        key = (node.name, self._option_keys[id(option)], arrival)
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
            cycles = tuple(kernel_cycles(kernel, self.ctx.device) for kernel in kernels)
            resolved = (tuple(kernels), tiles, cycles)
        except ConfigRefused as refusal:
            self.refusals[node.name][str(refusal)] += 1
            resolved = None
        self._resolved[key] = resolved
        return resolved

    @staticmethod
    def _area(footprint) -> int:
        return footprint.width * footprint.height

    def _own_legs(self, node: OpNode, kernels, live: Dict[str, Any]):
        """The legs of `node`'s kernels that no other layer's choice bears on -- into each from the graph's inputs and
        from the kernels before it, and out of its last to the graph's outputs -- each with its facts (`_leg_facts`),
        or why the transport classifier refuses one. `live` publishes each tensor it reads from another layer, whose
        legs its arrivals hold."""
        key = id(kernels[-1])
        if key not in self._own_cache:
            execution = self._execution(kernels, {id(inst): inst for inst in live.values()}.values())
            collector = TransportCollector(execution)
            try:
                arriving = tuple(
                    tuple(
                        (leg, self._leg_facts(leg, execution))
                        for item in kernel.inputs
                        if not set(self._sources(item.tensor)) <= set(live)
                        for leg in collector.item_connections(kernel, item)
                        if leg.producer.tensor not in live
                    )
                    for kernel in kernels
                )
                leaving = tuple(
                    (leg, self._leg_facts(leg, execution))
                    for leg in collector.output_connections(kernels[-1], self.read)
                )
                refused = [facts for legs in (*arriving, leaving) for _, facts in legs if isinstance(facts, str)]
            except ConfigRefused as refusal:
                refused = [str(refusal)]
            if refused:
                self.refusals[node.name][refused[0]] += 1
            self._own_cache[key] = refused[0] if refused else (arriving, leaving)
        return self._own_cache[key]

    def _sources(self, tensor: str) -> Tuple[str, ...]:
        """The tensors reading `tensor` reads: the sources of the folded view it is, or itself."""
        view = self.values[tensor].view
        return tuple(view.sources) if view else (tensor,)

    def _execution(self, kernels, producers) -> ExecutionIR:
        """The execution graph of a layer's kernels and the producers they read, for the transport classifier."""
        instances = [*producers, *kernels]
        return ExecutionIR(
            instances={inst.name: inst for inst in instances},
            tensor_contracts={t: c for inst in instances for t, c in self._output_contracts(inst).items()},
            values=self.values,
            graph_inputs=self.graph_inputs,
            graph_outputs=self.graph_outputs,
        )

    def _leg_facts(self, leg, execution):
        """One leg's part in a layer's cost and placement (`_LegFacts`), which depends on its two instances alone and
        is cached by them -- or why it is refused."""
        ends = [None if end is None or end.node is None else end for end in (leg.producer, leg.consumer)]
        insts = [None if end is None else execution.get(end.node.name) for end in ends]
        key = (
            *(id(inst) for inst in insts),
            leg.logical_tensor,
            *((end.tensor, end.group, end.ports, end.offset_base) for end in (leg.producer, leg.consumer) if end),
        )
        if key in self._leg_cache:
            return self._leg_cache[key]
        source, sink = insts
        streams = []
        try:
            # an ordered merge only where the leg is its tensor's only reader, not through a view over it either
            # (`classify.gathers`); a value only the execution graph has is a converter's, which the op that asked for
            # it alone reads
            tensor = leg.producer.tensor
            merge = (
                bool(self.ctx.device.packet_ordered_merge)
                and sink is not None
                and self.readers.get(tensor, 1) == 1
                and all(value == tensor for value, _ in self.read_as.get(tensor, ()))
                and tensor not in self.graph_outputs
            )
            realization = classify_connection(leg, execution, self.ctx.device, merge=merge).realization
            if realization == 'memtile':
                check_memtile_leg(leg, execution, self.ctx.device)
            if sink is not None and (realization == 'memtile' or source is None):
                binding = sink.ports.inputs[leg.consumer.tensor]
                for port in leg.consumer.selected_ports(binding.count):
                    tiles = self._port_tiles(sink, 'inputs', binding.group, port)
                    streams.append((sink.name, (binding.group, port), (BELOW, tuple(tiles))))
            if source is not None and (realization == 'memtile' or leg.consumer is None):
                binding = source.ports.outputs[leg.producer.tensor]
                for port in leg.producer.selected_ports(binding.count):
                    (tile,) = self._port_tiles(source, 'outputs', binding.group, port)
                    streams.append((source.name, (binding.group, port), (tile, (BELOW,))))
            # placement's edge (`placement._transport_edges`): a direct leg's port pairs, and whether each may be
            # one buffer (`shared_buffer.static_problem`)
            edge = None
            src_group, dst_group = leg.producer.group, None if leg.consumer is None else leg.consumer.group
            if realization == 'direct' and source is not None and sink is not None:
                unit = direct_unit(execution, leg, merge=merge)
                pairs = tuple(zip(unit.producer_ports, unit.consumer_ports))
                problems = (
                    static_problem(unit, source, src_group, w, sink, dst_group, r, self._readers(leg, w, execution))
                    for w, r in pairs
                )
                edge = (not any(problems), pairs)
            # the consumer's ports it may read in the producer's buffer: where some placement lines them all up; and
            # its pairs no placement lines up even alone, which placement decides pair by pair
            ports, unshared = None, 0 if edge is None else len(edge[1])
            pairs = () if edge is None else edge[1]
            if edge is not None and edge[0]:
                read = tuple(r for _, r in pairs)
                device = self.ctx.device
                rows, mirrored = int(device.rows - device.row_start), device.alternating_horizontal
                src_rect, dst_rect = self._rect(source), self._rect(sink)
                if share_offsets(src_rect, src_group, dst_rect, dst_group, pairs, rows, mirrored):
                    ports = tuple((dst_group, port) for port in read)
                unshared = sum(
                    not any(pair_offsets(src_rect, src_group, w, dst_rect, dst_group, r, rows, mirrored).values())
                    for w, r in pairs
                )
            own, shared = (
                (leg.producer.tensor, *leg_cycles(leg, realization, sharing, execution, self.ctx.device))
                for sharing in (False, True)
            )
            facts = _LegFacts(
                int(realization == 'memtile'),
                tuple(streams),
                ports,
                own,
                None if ports is None else shared,
                edge,
                unshared,
            )
        except ConfigRefused as refusal:
            facts = str(refusal)
        self._leg_cache[key] = facts
        return facts

    def _readers(self, leg, port: int, execution) -> List[str]:
        """Who reads port `port` of the leg's producer, as `TransportCollector.port_readers` lists them over the whole
        graph, which the search's execution graph of one layer does not hold: each layer once per input reading it,
        whole or through a view over that port, and the graph boundary. A value only the execution graph has is a
        converter's, which the op that asked for it alone reads."""
        tensor = leg.producer.tensor
        collector = TransportCollector(execution)
        reads = self.read_as.get(tensor) or [(tensor, leg.consumer.node.name)]
        readers = [layer for value, layer in reads if port in collector.ports_read(value, tensor)]
        return readers + (['the graph boundary'] if tensor in self.graph_outputs else [])

    def _rect(self, inst):
        """`instance_rect` of an instance, cached by it."""
        key = (id(inst), 'rect')
        if key not in self._geometry:
            self._geometry[key] = instance_rect(inst, int(self.ctx.device.row_start))
        return self._geometry[key]

    def _view(self, kernels) -> tuple:
        """What placement sees of `kernels` (`_Placed.views`)."""
        rows = range(int(self.ctx.device.rows - self.ctx.device.row_start))
        views = []
        for inst in kernels:
            key = (id(inst), 'view')
            if key not in self._geometry:
                rect = self._rect(inst)
                self._geometry[key] = (
                    inst.name,
                    (
                        rect.w,
                        rect.h,
                        repr((rect.input_face, rect.output_face, rect.extras)),
                        tuple(rect.locations_at(row) for row in rows),
                        tuple(tuple(rect.stream_locations(row)) if rect.stream_locations else () for row in rows),
                        frozenset(binding.group for binding in inst.ports.outputs.values()),
                    ),
                )
            views.append(self._geometry[key])
        return tuple(views)

    @staticmethod
    def _edges(kernels, legs) -> tuple:
        """What placement sees of the kernel legs `legs`, (kernel position, leg, facts), into `kernels`
        (`_Placed.edges`)."""
        return tuple(
            (
                leg.producer.node.name,
                kernels[position].name,
                facts.edge,
                (
                    leg.producer.tensor,
                    leg.producer.group,
                    leg.consumer.tensor,
                    leg.consumer.group,
                    kernels[position].input(leg.consumer.tensor).shared_memory,
                ),
            )
            for position, leg, facts in legs
            if leg.producer.node is not None
        )

    def _overflows(self, node: OpNode, streams: Dict[str, Dict[Any, Any]]) -> bool:
        """Whether a kernel's streams from or to below (`_LegFacts.streams`, by kernel) alone overflow its switches,
        counted among `node`'s refusals."""
        for name, ports in streams.items():
            problem = self._route_overflow(ports)
            if problem:
                self.refusals[node.name][f'{name}: its memory-tile and PLIO streams alone: {problem}.'] += 1
                return True
        return False

    def _port_tiles(self, inst, direction: str, group: str, port: int):
        """`port_tiles` of an instance placed at (0, 0), cached by it."""
        key = (id(inst), direction, group, port)
        if key not in self._geometry:
            self._geometry[key] = port_tiles(inst, direction, group, port, 0, 0)
        return self._geometry[key]

    def _route_overflow(self, streams):
        """`route_overflow` of one kernel's streams from or to below, cached by them."""
        key = frozenset(streams.items())
        if key not in self._route_cache:
            self._route_cache[key] = route_overflow(streams, self.ctx.device.stream_switch_ports)
        return self._route_cache[key]

    # -- every layer's alternatives ------------------------------------------------------------------

    def _alternatives(self) -> Dict[str, List[_Alternative]]:
        """Per layer, every alternative a design may take: an option resolved against the contracts of the layouts
        it reads, with per tensor it reads from another layer an arrival per class of the tensor's layouts whose legs
        into it are alike (`_classes`). Classes merge no two arrivals that cost or build differently, so a design of
        them is any design there is."""
        layouts: Dict[str, Dict[str, Any]] = {}  # tensor -> its layouts (signatures) -> an instance publishing it
        found: Dict[str, List[_Alternative]] = {}
        for node in self.layers:
            reads = [t for t in self.reads[node.name] if t in self.produced]  # in a layout a layer chose
            resolves = {t.name for t in node.inputs if t.name in layouts}  # resolution reads their contracts
            groups: Dict[str, Dict[str, Dict[str, Any]]] = {t: defaultdict(dict) for t in reads}
            for t in reads:
                for sig, inst in layouts[t].items():
                    contract = self._output_contracts(inst).get(t) if t in resolves else None
                    groups[t][repr(contract)][sig] = inst
            alternatives = found[node.name] = []
            for picked in itertools.product(*(groups[t].values() for t in reads)):
                arriving = dict(zip(reads, picked))
                for option in self.options[node.name]:
                    alt = self._alternative(node, option, arriving)
                    if alt is None:
                        continue
                    for t, sig in alt.published:
                        layouts.setdefault(t, {}).setdefault(sig, alt.kernels[-1])
                    alternatives.append(alt)
            if not alternatives:
                raise ConfigRefused(
                    self._failure(f'{node.name} resolves in no design: every option of it is refused', {})
                )
        return found

    def _alternative(self, node: OpNode, option, arriving: Dict[str, Dict[str, Any]]) -> Optional[_Alternative]:
        """`node` under `option`, each tensor it reads from another layer arriving in one of the layouts of
        `arriving` (tensor -> layout -> an instance publishing it), all alike in the contracts resolution reads; None
        where resolution, the transport classifier or every arrival of a tensor refuses it."""
        live = {t: next(iter(members.values())) for t, members in arriving.items()}
        resolved = self._resolve(node, option, live)
        if resolved is None:
            return None
        kernels, tiles, cycles = resolved
        own = self._own_legs(node, kernels, live)
        if isinstance(own, str):
            return None
        into, leaving = own
        legs = [(position, leg, facts) for position, kernel_legs in enumerate(into) for leg, facts in kernel_legs]
        streams = defaultdict(dict)  # kernel -> its own streams from or to below, at (0, 0)
        for _, facts in (*itertools.chain.from_iterable(into), *leaving):
            for name, port, stream in facts.streams:
                streams[name][port] = stream
        if self._overflows(node, streams):
            return None
        costs = [[] for _ in kernels]  # per kernel, its own legs' (source, latency, interval) at best
        try:
            for position, leg, facts in legs:
                costs[position].append(_at_best(kernels[position], leg, facts))
        except ConfigRefused as refusal:
            self.refusals[node.name][str(refusal)] += 1
            return None
        arrivals = []
        for tensor, members in arriving.items():
            found = [
                arrival
                for layouts, tensor_legs in self._classes(node, kernels, live, tensor, members)
                if (arrival := self._arrival(node, kernels, cycles, costs, streams, tensor, layouts, tensor_legs))
            ]
            if not found:
                return None
            arrivals.append((tensor, tuple(found)))
        may_share = defaultdict(set)  # kernel -> its input ports some placement reads in their producer's buffer
        for arrival in itertools.chain.from_iterable(found for _, found in arrivals):
            for position, facts in arrival.facts:
                may_share[kernels[position].name].update(facts.ports or ())
        for position, _, facts in legs:
            may_share[kernels[position].name].update(facts.ports or ())
        departure = [facts.own[1:] for _, facts in leaving]
        inst = kernels[-1]
        return _Alternative(
            option=option,
            kernels=kernels,
            published=tuple((t, self._signature(inst, t)) for t in inst.outputs),
            placed=_Placed(self._view(kernels), self._edges(kernels, legs), self._below(streams)),
            tiles=tiles,
            cycles=cycles,
            reserved=sum(_reserved_tiles(kernel, may_share[kernel.name]) for kernel in kernels),
            memtile=sum(facts.memtile for _, _, facts in legs) + sum(facts.memtile for _, facts in leaving),
            copies=sum(facts.unshared for _, _, facts in legs),
            slowest=max((*cycles, *(cc for kernel in costs for *_, cc in kernel), *(cc for _, cc in departure))),
            delays=_delays(kernels, cycles, [[cost[:2] for cost in kernel] for kernel in costs], None),
            departure=max((cc for cc, _ in departure), default=0),
            arrivals=tuple(arrivals),
        )

    def _arrival(self, node, kernels, cycles, own, streams, tensor, layouts, legs) -> Optional[_Arrival]:
        """`tensor` arriving into `kernels` along `legs`, (kernel position, leg, facts), beside their own legs' costs
        `own` and streams from or to below `streams` (`_alternative`); None where a leg its kernel requires shared no
        placement shares (`_at_best`), or a kernel's streams from or to below with these overflow its switches."""
        try:
            costs = [(position, _at_best(kernels[position], leg, facts)) for position, leg, facts in legs]
        except ConfigRefused as refusal:
            self.refusals[node.name][str(refusal)] += 1
            return None
        merged = defaultdict(dict)
        for _, _, facts in legs:
            for name, port, stream in facts.streams:
                merged[name][port] = stream
        if self._overflows(node, {name: {**streams[name], **ports} for name, ports in merged.items()}):
            return None
        into = [[cost[:2] for cost in kernel] for kernel in own]
        for position, (_, latency, _) in costs:
            into[position].append((tensor, latency))
        return _Arrival(
            layouts=frozenset(layouts),
            placed=_Placed((), self._edges(kernels, legs), self._below(merged)),
            facts=tuple((position, facts) for position, _, facts in legs),
            delays=_delays(kernels, cycles, into, tensor),
            slowest=max((cc for _, (*_, cc) in costs), default=0),
            memtile=sum(facts.memtile for _, _, facts in legs),
            copies=sum(facts.unshared for _, _, facts in legs),
        )

    @staticmethod
    def _below(streams: Dict[str, Dict[Any, Any]]) -> tuple:
        """Streams from or to below, by kernel, as `_Placed.below` holds them."""
        return tuple((name, port, stream) for name, ports in streams.items() for port, stream in ports.items())

    def _classes(self, node: OpNode, kernels, live, tensor: str, layouts: Dict[str, Any]) -> List[tuple]:
        """`layouts` of `tensor` in classes whose legs into `kernels` have the same facts, each with those legs,
        (kernel position, leg, facts); a layout whose legs are refused is in none. Only the inputs reading `tensor`
        are classified, so no other tensor's layout in `live` bears on its classes."""
        classes: Dict[str, Tuple[Dict[str, Any], list]] = {}
        for sig, inst in layouts.items():
            producers = {id(i): i for i in {**live, tensor: inst}.values()}
            execution = self._execution(kernels, producers.values())
            collector = TransportCollector(execution)
            legs, facts = [], []
            try:
                for position, kernel in enumerate(kernels):
                    for item in kernel.inputs:
                        if tensor not in self._sources(item.tensor):
                            continue
                        for leg in collector.item_connections(kernel, item):
                            if leg.producer.tensor != tensor:
                                continue
                            found = self._leg_facts(leg, execution)
                            if isinstance(found, str):
                                raise ConfigRefused(found)
                            ends = (leg.producer, leg.consumer)
                            legs.append((position, leg, found))
                            facts.append(
                                (position, *((e.group, e.ports, e.offset_base) for e in ends), item.tensor, found)
                            )
            except ConfigRefused as refusal:
                self.refusals[node.name][str(refusal)] += 1
                continue
            members, _ = classes.setdefault(repr(facts), ({}, legs))
            members[sig] = inst
        return list(classes.values())

    # -- the constraint program ---------------------------------------------------------------------

    def _solve(self, alternatives: Dict[str, List[_Alternative]], cuts) -> Optional[Dict[str, tuple]]:
        """The best design left by `rank`, or None where every design is cut: per layer an alternative and, per tensor
        it reads from another layer, an arrival of a layout the producer's alternative publishes, as (alternative
        index, {tensor: arrival index}). A cut is groups (layer, tensor or None, indices) of which a design takes
        no more than all but one: alternatives where the tensor is None, else (alternative, arrival) pairs."""
        model = cp_model.CpModel()
        x: Dict[str, list] = {}  # by layer, a literal per alternative
        y: Dict[str, list] = {}  # by layer and alternative, per tensor a literal per arrival: one where it is taken
        for name, alts in alternatives.items():
            x[name] = [model.NewBoolVar('') for _ in alts]
            model.AddExactlyOne(x[name])
            y[name] = []
            for literal, alt in zip(x[name], alts):
                taken = {tensor: [model.NewBoolVar('') for _ in found] for tensor, found in alt.arrivals}
                for literals in taken.values():
                    model.Add(sum(literals) == literal)
                y[name].append(taken)

        def arrivals(name):
            """Each arrival a layer may take: (tensor, arrival, literal)."""
            for alt, taken in zip(alternatives[name], y[name]):
                for tensor, found in alt.arrivals:
                    yield from ((tensor, arrival, literal) for arrival, literal in zip(found, taken[tensor]))

        published = defaultdict(lambda: defaultdict(list))  # tensor -> layout -> the alternatives publishing it
        for name, alts in alternatives.items():
            for literal, alt in zip(x[name], alts):
                for tensor, sig in alt.published:
                    published[tensor][sig].append(literal)
        for name in alternatives:
            reading = defaultdict(list)  # (tensor, layouts) -> its arrivals from one of them
            for tensor, arrival, literal in arrivals(name):
                reading[tensor, arrival.layouts].append(literal)
            for (tensor, layouts), literals in reading.items():
                model.Add(sum(literals) <= sum(p for sig in layouts for p in published[tensor][sig]))

        def most(alt, arrival_cost) -> int:
            return sum(max(arrival_cost(arrival) for arrival in found) for _, found in alt.arrivals)

        # no design waits longer than every layer's slowest alternative and arrivals in turn
        horizon = 1 + sum(
            max(
                alt.departure
                + alt.slowest
                + max((cc for _, cc in alt.delays), default=0)
                + most(alt, lambda arrival: arrival.slowest + max((cc for _, cc in arrival.delays), default=0))
                for alt in alts
            )
            for alts in alternatives.values()
        )
        latency, interval = model.NewIntVar(0, horizon, 'latency'), model.NewIntVar(0, horizon, 'interval')
        ready: Dict[str, Any] = {}
        for node in self.layers:
            name, alts = node.name, alternatives[node.name]
            outputs = alts[0].kernels[-1].outputs
            ready.update({tensor: model.NewIntVar(0, horizon, '') for tensor in outputs})
            # from each source -- the graph's inputs (None), each tensor read from another layer -- to each output
            waits = defaultdict(list)
            for alt, literal in zip(alts, x[name]):
                waits[None].append((dict(alt.delays), literal))
            for tensor, arrival, literal in arrivals(name):
                waits[tensor].append((dict(arrival.delays), literal))
            for source, terms in waits.items():
                for tensor in outputs:
                    after = [delays.get(tensor) for delays, _ in terms]
                    if all(cc is None for cc in after):
                        continue
                    if any(cc is None for cc in after):
                        raise RuntimeError(f'{name}: {tensor!r} waits for {source!r} in some alternatives only.')
                    begin = 0 if source is None else ready[source]
                    model.Add(ready[tensor] >= begin + sum(cc * literal for cc, (_, literal) in zip(after, terms)))
            departure = sum(alt.departure * literal for alt, literal in zip(alts, x[name]))
            for tensor in outputs:
                model.Add(latency >= ready[tensor] + departure)
            model.Add(interval >= sum(alt.slowest * literal for alt, literal in zip(alts, x[name])))
            for tensor in {tensor for tensor, _, _ in arrivals(name)}:
                model.Add(interval >= sum(a.slowest * literal for t, a, literal in arrivals(name) if t == tensor))

        def total(alt_cost, arrival_cost=lambda arrival: 0) -> Any:
            return sum(
                alt_cost(alt) * literal for name, alts in alternatives.items() for alt, literal in zip(alts, x[name])
            ) + sum(arrival_cost(arrival) * literal for name in alternatives for _, arrival, literal in arrivals(name))

        def bound(alt_cost, arrival_cost) -> int:
            return sum(max(alt_cost(alt) + most(alt, arrival_cost) for alt in alts) for alts in alternatives.values())

        def memtile(item) -> int:
            return item.memtile

        def copies(item) -> int:
            return item.copies

        # each criterion of `RANKING` and a bound on it
        criteria = {
            'latency': (latency, horizon),
            'interval': (interval, horizon),
            'memtile': (total(memtile, memtile), bound(memtile, memtile)),
            'copies': (total(copies, copies), bound(copies, copies)),
            'tiles': (total(lambda alt: alt.tiles), self.max_tiles),
            'fallbacks': (total(lambda alt: int(bool(_kernel(alt.option)))), len(alternatives)),
        }
        model.Add(criteria['tiles'][0] <= self.max_tiles)
        model.Add(criteria['tiles'][0] + total(lambda alt: alt.reserved) <= self._fill(alternatives))
        for groups in cuts:
            literals = [
                x[name][index] if tensor is None else y[name][index[0]][tensor][index[1]]
                for name, tensor, indices in groups
                for index in indices
            ]
            model.Add(sum(literals) <= len(groups) - 1)

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = SEARCH_WORKERS
        solver.parameters.interleave_search = True  # deterministic: a model always yields the same design
        solver.parameters.max_deterministic_time = self.search_time
        decisions = [
            *(literal for literals in x.values() for literal in literals),
            *(literal for name in alternatives for _, _, literal in arrivals(name)),
        ]
        found, proven = solve_ranked(model, solver, criteria, RANKING[self.mode], decisions)
        if found is None and proven:
            return None
        if found is None:
            raise SearchLimit(
                f'choose_parallelism: the solver found no design within AIEConfig SearchTime={self.search_time:g}.'
            )
        if not proven:
            self.doubts.append(f'a solve was not proven optimal within AIEConfig SearchTime={self.search_time:g}')
        picks = {}
        for name, literals in x.items():
            i = next(i for i, literal in enumerate(literals) if found.decisions[literal.Index()])
            taken = {
                tensor: next(c for c, literal in enumerate(arrival) if found.decisions[literal.Index()])
                for tensor, arrival in y[name][i].items()
            }
            picks[name] = (i, taken)
        return picks

    def _fill(self, alternatives: Dict[str, List[_Alternative]]) -> int:
        """The most tiles a design may fill: the room, or once a design found no placement in its time, halfway from
        the fullest design placed -- or the least any design fills -- to the emptiest that found none, so each design
        built halves the fills whose placement is unknown."""
        shared = self.shared
        if shared.unplaced_fill is None:
            return self.room
        known = max(shared.placed_fill, _least(alternatives, lambda alt: alt.tiles + alt.reserved))
        if known >= shared.unplaced_fill:
            return min(self.room, shared.unplaced_fill - 1)
        return min(self.room, known + (shared.unplaced_fill - known) // 2)

    def _design(self, alternatives: Dict[str, List[_Alternative]], picks: Dict[str, tuple]) -> _Design:
        """The design the picks make (`_solve`), its latency the latest any kernel or graph output finishes."""
        ready: Dict[str, int] = {}
        layout: Dict[str, str] = {}
        memtile = tiles = reserved = interval = latency = copies = 0
        chosen = []
        for node in self.layers:
            i, taken = picks[node.name]
            alt = alternatives[node.name][i]
            done = dict.fromkeys(alt.kernels[-1].outputs, 0)
            for tensor, cc in alt.delays:
                done[tensor] = max(done[tensor], cc)
            for tensor, found in alt.arrivals:
                arrival = found[taken[tensor]]
                if layout[tensor] not in arrival.layouts:
                    raise RuntimeError(f'{node.name} reads {tensor!r} in a layout its producer does not publish.')
                for out, cc in arrival.delays:
                    done[out] = max(done[out], ready[tensor] + cc)
                memtile, copies, interval = (
                    memtile + arrival.memtile,
                    copies + arrival.copies,
                    max(interval, arrival.slowest),
                )
            ready.update(done)
            layout.update(alt.published)
            latency = max(latency, *(cc + alt.departure for cc in done.values()))
            memtile, tiles, reserved = memtile + alt.memtile, tiles + alt.tiles, reserved + alt.reserved
            interval, copies = max(interval, alt.slowest), copies + alt.copies
            chosen.append((node.name, alt.kernels[-1], alt.option))
        return _Design(memtile, tiles, reserved, interval, latency, copies, tuple(chosen))

    # -- the best design the pipeline builds --------------------------------------------------------

    def rank(self, design: _Design) -> tuple:
        """The design's rank by the mode's criteria (`RANKING`): the lower, the better."""
        order = RANKING[self.mode]
        return rank({name: getattr(design, name) for name, _ in order}, order)

    def best_buildable(self, beat: Optional[tuple] = None) -> Optional[Tuple[_Design, Any]]:
        """The best-ranked design the rest of the pipeline builds, as placed (`_as_placed`), and the context it was
        built in; None where no design ranks better than `beat`. Designs are built in the order of their bounds
        until the next ranks no better than the best built, which no design left can then beat. A design built is
        cut with every design alike (`_Alternative.alike`); a design placement proves it cannot place with every design
        placement sees alike (`_Placed`), in this search and the others (`_Shared.unplaced`); one it finds no
        placement for in its time halves the fills tried (`_fill`), which leaves the design found not proven the best;
        any other refusal, it alone. `tried` counts the designs built."""
        alternatives = self._alternatives()
        cuts, unbuilt, best = [], defaultdict(int), None
        while True:
            unplaced = [_groups(alternatives, keys, _placed) for keys in self.unplaced]
            picks = self._solve(alternatives, unplaced + cuts)
            if picks is None:
                break
            bound = self._design(alternatives, picks)
            target = beat if best is None else self.rank(best[0])
            if target is not None and self.rank(bound) >= target:
                break  # no design left ranks better
            if self.shared.built == self.builds:
                failure = self._failure(f'search limit: {self.builds} designs built (AIEConfig SearchBuilds)', unbuilt)
                if best is None:
                    raise SearchLimit(failure)
                self.doubts.append(failure.splitlines()[0])
                break
            built = self._build(bound)
            self.tried += 1
            self.shared.built += 1
            if isinstance(built, PlacementInfeasibleError):
                unbuilt[str(built)] += 1
                if built.proven:
                    self.unplaced.append(_keys(alternatives, picks, _placed))
                    continue
                full = bound.tiles + bound.reserved  # the fills tried halve (`_fill`)
                unplaced = self.shared.unplaced_fill
                self.shared.unplaced_fill = full if unplaced is None else min(unplaced, full)
                continue
            if isinstance(built, ConfigRefused):
                cuts.append(_groups(alternatives, _keys(alternatives, picks, id), id))
                unbuilt[str(built)] += 1
                continue
            # every design alike in each layer places as this one: of them, the search took the best ranked first
            cuts.append(_groups(alternatives, _keys(alternatives, picks, _alike), _alike))
            self.shared.placed_fill = max(self.shared.placed_fill, bound.tiles + bound.reserved)
            design = self._as_placed(bound, built)
            if target is None or self.rank(design) < target:
                best = (design, built)
        if best is not None:
            return best
        if beat is not None and picks is not None:
            return None
        if not self.tried:
            kernels = _least(alternatives, lambda alt: alt.tiles)
            filled = _least(alternatives, lambda alt: alt.tiles + alt.reserved)
            raise ConfigRefused(
                self._failure(
                    f'no design resolves within the tile budget: every design takes at least {kernels} tiles for its '
                    f"kernels, and fills at least {filled} of the array's {self.room} with those its input buffers "
                    'reserve',
                    unbuilt,
                )
            )
        raise ConfigRefused(
            self._failure(f'none of the {self.tried} designs within the tile budget can be built', unbuilt)
        )

    def _as_placed(self, bound: _Design, built) -> _Design:
        """The design as its placement estimates it (`PhysicalIR.estimate`), which the search bounds: a placement
        that beats its bound means the two estimate it apart."""
        estimate = built.ir.physical.estimate
        placed = replace(bound, **{name: int(estimate[name]) for name in PLACED})
        for name in PLACED:
            if getattr(placed, name) < getattr(bound, name):
                raise RuntimeError(
                    f"choose_parallelism: placement estimates {name} {getattr(placed, name)} below the search's bound "
                    f'{getattr(bound, name)}: the two estimate the design apart.'
                )
        if not estimate['proven']:
            self.doubts.append('a placement was not proven the best within AIEConfig PlacerTime')
        return placed

    def choices(self, design: _Design) -> Dict[str, Dict[str, Any]]:
        """Per directive the search chooses, what the design resolved each chosen layer under: its parallelism as
        resolved, its microtiling and fallback kernel (`fallback_kernels`) as chosen."""
        made: Dict[str, Dict[str, Any]] = {'parallelism': {}, 'microtiling': {}, 'kernel': {}}
        for name, inst, option in design.chosen:
            if name in self.choosing:
                made['parallelism'][name] = asdict(inst.config.parallelism)
            if option and 'microtiling' in option:
                made['microtiling'][name] = option['microtiling']
            if _kernel(option):
                made['kernel'][name] = _kernel(option)
        return made

    def _build(self, design: _Design):
        """The rest of the pipeline run on a copy of the context with this design: the built copy, or the refusal."""
        from ..pipeline import DEFAULT_PIPELINE

        trial = copy.deepcopy(self.ctx)
        # the design's bound on what placement decides, which a placement reaching it is proven the best by
        trial.ir.optimizer = {**self.choices(design), 'bound': {name: getattr(design, name) for name in PLACED}}
        rest = DEFAULT_PIPELINE[DEFAULT_PIPELINE.index(ChooseParallelism) + 1 :]
        try:
            run_aie_passes(trial, [cls() for cls in rest])
        except ConfigRefused as refusal:
            return refusal
        return trial

    def _failure(self, what: str, unbuilt: Dict[str, int]) -> str:
        lines = [f'choose_parallelism ({self.mode}, MaxTiles={self.max_tiles}): {what}.']
        if what.startswith('search limit'):
            lines.append(
                '  A design may still exist: raise AIEConfig SearchBuilds, or give some layers a parallelism directive '
                'to narrow the search.'
            )
        for name, messages in self.refusals.items():
            for message, count in sorted(messages.items(), key=lambda item: -item[1])[:3]:
                lines.append(f'  {name}: {count} option(s) refused: {message}')
        lines += [f'  {count} design(s) not built: {message}' for message, count in unbuilt.items()]
        return '\n'.join(lines)
