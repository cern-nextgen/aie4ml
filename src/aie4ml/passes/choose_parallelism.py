"""Choose every layer's parallelism for the whole graph at once (AIEConfig `Optimize`), before resolution.

Resolution and the transport classifier stay the only judges of legality; user directives are constraints and are
never rewritten -- the choice lives in `ctx.ir.optimizer`, which Resolve and placement read. Work is
multiply-accumulates per tile, a proxy for time: a cost model replaces `_work` and `_legs`.

The search is bounded, not exhaustive: it keeps KEEP designs per state and tile count and builds at most
MAX_PLACEMENT_TRIALS, so it reports a search limit unless it discarded nothing.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..errors import ConfigRefused
from ..ir import get_backend_context
from ..ir.graph import ExecutionIR, ExecutionValue, OpNode
from ..op_impls import get_family_resolver_registry
from .base import AIEPass, run_aie_passes
from .legalize_layouts import convert_inputs
from .resolve import logical_values, output_contracts, resolve_instance
from .transport.classify import classify_connection
from .transport.collect import TransportCollector

log = logging.getLogger(__name__)

MODES = ('performance', 'resource')
KEEP = 4  # designs kept per state and tile count, so the next ones are at hand when placement refuses one
MAX_PARTIALS = 50_000
MAX_PLACEMENT_TRIALS = 16


@dataclass(frozen=True)
class _Design:
    """The layers resolved so far: what they cost, the instance chosen for each, and which of their outputs later
    layers still read."""

    memtile: int
    tiles: int
    work: int
    chosen: Tuple[Tuple[str, Any], ...]  # (layer, instance)
    live: Tuple[Tuple[str, Any], ...]  # (tensor, producing instance)


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
        print(f'[aie4ml] Design found: {design.tiles} AIE tiles, the best of {search.compared} compared.', flush=True)
        ctx.ir.optimizer = {
            'mode': mode,
            'max_tiles': max_tiles,
            'tiles': design.tiles,
            'memtile_legs': design.memtile,
            'work_per_tile': design.work if mode == 'performance' else None,  # proxy: multiply-accumulates
            'designs_tried': trials,
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
        self.layers: List[OpNode] = [node for node in ctx.ir.logical if not node.is_folded_view]
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
        self.values = logical_values(ctx.ir.logical)
        self.graph_inputs = tuple(ctx.ir.logical.input_tensor_names)
        self.graph_outputs = tuple(ctx.ir.logical.output_tensor_names)
        # what each layer's legs read -- its inputs, or the sources of a folded view it reads -- and after which
        # layer each tensor is read no more: a tensor stays in the state until then
        self.reads: Dict[str, Tuple[str, ...]] = {}
        self.last_read: Dict[str, int] = {}
        for position, node in enumerate(self.layers):
            views = [self.values[t.name].view for t in node.inputs if not t.is_parameter]
            inputs = [t.name for t in node.inputs if not t.is_parameter]
            self.reads[node.name] = tuple(
                source for name, view in zip(inputs, views) for source in (view.sources if view else (name,))
            )
            self.last_read.update({tensor: position for tensor in self.reads[node.name]})
        self.refusals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.truncated = False  # whether the program has discarded a design since last reset
        self._resolved: Dict[Any, Any] = {}
        self._memtile_legs: Dict[Any, Optional[int]] = {}
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
                inst.variant.describe_output_staging(inst.node, inst.config, tensor, port, None)
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
        """(kernels, tiles, work) for `node` under `option` -- the layout converters it needs, then its own
        instance -- or None when resolution refuses it; per input arrival."""
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
            resolved = (tuple(kernels), tiles, self._work(inst) if node.name in self.choosing else 0)
        except ConfigRefused as refusal:
            self.refusals[node.name][str(refusal)] += 1
            resolved = None
        self._resolved[key] = resolved
        return resolved

    @staticmethod
    def _area(footprint) -> int:
        return footprint.width * footprint.height

    def _work(self, inst) -> int:
        return inst.variant.work(inst.node, inst.config) if self.mode == 'performance' else 0

    def _legs(self, node: OpNode, kernels, live: Dict[str, Any]) -> Optional[int]:
        """How many legs into and out of `node`'s kernels a memory tile carries, or None when the transport
        classifier refuses one."""
        producers = {id(live[t]): live[t] for t in self.reads[node.name] if t in live}
        key = (id(kernels[-1]), tuple(sorted(producers)))
        if key not in self._memtile_legs:
            instances = [*producers.values(), *kernels]
            execution = ExecutionIR(
                instances={inst.name: inst for inst in instances},
                tensor_contracts={t: c for inst in instances for t, c in self._output_contracts(inst).items()},
                values=self.values,
                graph_inputs=self.graph_inputs,
                graph_outputs=self.graph_outputs,
            )
            collector = TransportCollector(execution)
            try:
                legs = [leg for kernel in kernels for leg in collector.input_connections(kernel)]
                legs += collector.output_connections(kernels[-1], self.last_read)
                decisions = [classify_connection(leg, execution, self.ctx.device.has_memtile) for leg in legs]
                self._memtile_legs[key] = sum(decision.realization == 'memtile' for decision in decisions)
            except ConfigRefused as refusal:
                self.refusals[node.name][str(refusal)] += 1
                self._memtile_legs[key] = None
        return self._memtile_legs[key]

    # -- the dynamic program ------------------------------------------------------------------------

    def designs(self, work_bound: Optional[int]) -> List[_Design]:
        """The complete designs whose busiest chosen tile stays within `work_bound`, as the program keeps them."""
        frontier = {((), 0): [_Design(0, 0, 0, (), ())]}
        for position, node in enumerate(self.layers):
            grown: Dict[Any, List[_Design]] = defaultdict(list)
            for designs in frontier.values():
                live = dict(designs[0].live)  # every design of a state sees the same arrivals
                for option in self.options[node.name]:
                    resolved = self._resolve(node, option, live)
                    if resolved is None or (work_bound is not None and resolved[2] > work_bound):
                        continue
                    kernels, tiles, work = resolved
                    memtile = self._legs(node, kernels, live)
                    if memtile is None:
                        continue
                    inst = kernels[-1]
                    kept = {t: i for t, i in live.items() if self.last_read[t] > position}
                    kept.update({t: inst for t in inst.outputs if self.last_read.get(t, -1) > position})
                    state = tuple(sorted((t, self._signature(i, t)) for t, i in kept.items()))
                    for design in designs:
                        total = design.tiles + tiles
                        if total <= self.max_tiles:
                            grown[(state, total)].append(
                                _Design(
                                    design.memtile + memtile,
                                    total,
                                    max(design.work, work),
                                    design.chosen + ((node.name, inst),),
                                    tuple(kept.items()),
                                )
                            )
            # Designs sharing a state and tile count have the same futures: keeping KEEP of them loses alternatives
            # for placement, never feasibility, so an empty result still proves no design resolves.
            self.truncated |= any(len(found) > KEEP for found in grown.values())
            frontier = {key: sorted(found, key=lambda d: (d.memtile, d.work))[:KEEP] for key, found in grown.items()}
            if sum(len(found) for found in frontier.values()) > MAX_PARTIALS:
                raise RuntimeError(
                    f'choose_parallelism: over {MAX_PARTIALS} partial designs at {node.name}; give some layers a '
                    'parallelism directive to narrow the search.'
                )
        return [design for found in frontier.values() for design in found]

    def _rank(self, design: _Design) -> tuple:
        if self.mode == 'resource':
            return (design.tiles, design.memtile)
        return (design.work, design.memtile, design.tiles)

    def best_buildable(self) -> Tuple[_Design, Any, int]:
        """The best-ranked design the rest of the pipeline builds, the context it was built in, and how many
        designs were tried."""
        bounds: List[Optional[int]] = [None]
        complete = self.designs(None)  # resolves every option, so each one's work is known
        self.compared = len({json.dumps(self.parallelism(design), sort_keys=True) for design in complete})
        if self.mode == 'performance':
            bounds = sorted({resolved[2] for resolved in self._resolved.values() if resolved is not None})
            low, high = 0, len(bounds)  # designs exist from some bound on
            while low < high:
                middle = (low + high) // 2
                low, high = (low, middle) if self.designs(bounds[middle]) else (middle + 1, high)
            bounds = bounds[low:]
        tried, unbuilt = set(), defaultdict(int)
        self.truncated = False
        for bound in bounds:
            for design in sorted(self.designs(bound), key=self._rank):
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
        raise RuntimeError(
            self._failure(
                f'search limit: none of the {len(tried)} designs kept ({KEEP} per search state) can be built', unbuilt
            )
        )

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
