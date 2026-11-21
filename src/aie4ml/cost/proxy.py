"""A compile-free estimate of a kernel's cycles per call: per code group and cascade role, a linear fit over every
product of a subset of the kernel's shape features, on compiled kernels' static issue cycles, plus the group's
stall share from simulated single kernels and its wrapper cycles. Each estimate carries the empirical range of
held-out calibration kernels -- evidence, not a bound -- and anything outside the calibrated region, range, groups,
target or sources is refused with its reason.

A kernel's cost reads only its own features and role; its chains' length and number are the graph's. For chains,
`cascade_timeline` spreads the cycles over blocks of partial results, words read before each block and written
after. The artifact also holds the target's fitted cascade link latency and handoff setups (aie4ml.cost.design).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..device_catalog import lookup_device
from .chain import Timeline, timeline
from .listing import Loop, LoopPlan

SCHEMA_VERSION = 6
FOLDS = 5
QUANTILES = (0.05, 0.95)


@dataclass(frozen=True)
class KernelCost:
    """Cycles per call (own stalls included, wrapper apart) with their empirical range and, in a chain, (blocks,
    words per block); all None, with the `refusal`, when there is no estimate."""

    cycles: Optional[float]
    low: Optional[float]
    high: Optional[float]
    wrapper: Optional[int]
    cascade: Optional[Tuple[int, int]]
    evidence: str
    refusal: Optional[str] = None


@dataclass(frozen=True)
class KernelProxy:
    part: str
    generation: str
    compiler: str  # the AIE compiler release the calibration kernels were compiled with
    sources: Dict[str, str]  # variant id -> its kernel templates' fingerprint, as calibrated
    regions: Dict[str, Dict[str, int]]  # variant id -> the least value of each feature it was calibrated from
    blocks: Dict[str, int]  # variant id -> cascade words of one block of partial results
    stalls: Dict[str, List[float]]  # 'variant group' -> lowest, median and highest stall share simulated
    models: Dict[str, Dict]  # 'variant role group' -> coefficients, cascade words, wrapper, kernels, features' ranges
    residuals: Dict[str, List[float]]  # variant id -> relative error quantiles of left-out kernels
    link_latency: int  # cycles from a cascade word's write to its earliest unstalled read (aie4ml.cost.chain.Link)
    transport: Dict[str, float]  # the setup of each kind of handoff: boundary_in, boundary_out, lock, dma

    def cost(self, variant: str, sources: str, role: str, group: str, features: Mapping[str, int]) -> KernelCost:
        """The cost of a kernel of `variant` whose kernel templates have the fingerprint `sources`."""
        model = self.models.get(_model(variant, role, group))
        if variant not in self.sources:
            refusal = f'{variant} has no cost model for {self.generation} and compiler {self.compiler}.'
        elif sources != self.sources[variant]:
            refusal = f'{variant} changed since its cost model was calibrated.'
        elif model is None:
            refusal = f'compile required: no cost model for {role} kernels of {variant} {group}.'
        else:
            below = [
                f'{n}={features[n]} < {least}' for n, least in self.regions[variant].items() if features[n] < least
            ]
            outside = [
                f'{n}={features[n]} outside {low}..{high}'
                for n, (low, high) in model['ranges'].items()
                if not low <= features[n] <= high
            ]
            if below:
                refusal = f'compile required: below the calibrated region ({", ".join(below)}).'
            elif outside:
                refusal = f'compile required: beyond the calibrated range ({", ".join(outside)}).'
            else:
                refusal = None
        if refusal is not None:
            return KernelCost(None, None, None, None, None, '', refusal)
        terms = _terms(features)
        issue = float(np.dot(model['coefficients'], terms))
        least, stall, most = self.stalls[_group(variant, group)]
        low, high = self.residuals[variant]
        cascade = None
        if role != 'single':
            words = round(float(np.dot(model['words'], terms)))
            cascade = (words // self.blocks[variant], self.blocks[variant])
        evidence = f"cost model of {model['kernels']} {role} kernels, {self.generation}, compiler {self.compiler}"
        return KernelCost(
            issue * (1 + stall),
            issue * (1 + low) * (1 + least),
            issue * (1 + high) * (1 + most),
            model['wrapper'],
            cascade,
            evidence,
        )

    @classmethod
    def calibrate(
        cls,
        records: Sequence[Mapping],
        regions: Mapping[str, Mapping[str, int]],
        link_latency: int,
        transport: Mapping[str, float],
        exclude: Collection[str] = (),
    ) -> 'KernelProxy':
        """Fit from calibration records (one per compiled kernel), each variant's region and the target's fitted link
        latency and handoff setups; no model for the `exclude` keys (`_model`)."""
        records = [r for r in records if r['group'] is not None and r['static'] is not None]
        targets = {(r['part'], r['generation'], r['compiler']) for r in records}
        if len(targets) != 1:
            raise ValueError(f'calibration records span {len(targets)} targets, not one: {sorted(targets)}')
        sources: Dict[str, str] = {}
        compiled: Dict[tuple, Mapping] = {}
        words: Dict[tuple, int] = {}
        for r in records:
            if sources.setdefault(r['variant'], r['templates']) != r['templates']:
                raise ValueError(f"{r['variant']} was calibrated from two versions of its sources.")
            key = (r['variant'], r['role'], r['coordinate'])
            if _kernel(compiled.setdefault(key, r)) != _kernel(r):
                raise ValueError(f"{' '.join(key)} compiled to two kernels: {compiled[key]['design']}, {r['design']}.")
            if words.setdefault(key, _cascade_words(r)) != _cascade_words(r):
                raise ValueError(f"{' '.join(key)} moves other cascade words in {r['design']} than elsewhere.")
        stalls = defaultdict(list)
        for r in records:
            if r['cycles'] is not None and r['role'] == 'single':
                stalls[_group(r['variant'], r['group'])].append(r['cycles'] / r['static'] - 1)
        blocks = {
            variant: math.gcd(*[n for (v, _, _), n in words.items() if v == variant and n]) or 1 for variant in sources
        }
        regions = {variant: dict(regions[variant]) for variant in sources}
        inside = [
            {**r, 'words': words[(r['variant'], r['role'], r['coordinate'])]}
            for r in compiled.values()
            if all(r['features'][name] >= least for name, least in regions[r['variant']].items())
        ]
        models = {key: model for key, model in _fit(inside).items() if model is not None and key not in exclude}
        for key in models:
            variant, _, group = key.split(' ', 2)
            if _group(variant, group) not in stalls:
                raise ValueError(f'{key}: no kernel of its group was simulated alone to measure its stalls.')
        residuals = {}
        for variant in sources:
            errors = []
            for fold in range(FOLDS):
                train = _fit([r for r in inside if r['variant'] == variant and _fold(r) != fold])
                for r in inside:
                    held = train.get(_model(r['variant'], r['role'], r['group']))
                    if r['variant'] == variant and _fold(r) == fold and held is not None and _within(held, r):
                        errors.append(float(np.dot(held['coefficients'], _terms(r['features']))) / r['static'] - 1)
            if not errors:
                raise ValueError(f'{variant}: too few kernels in its cost region to fit a cost model.')
            residuals[variant] = [float(np.quantile(errors, q)) for q in QUANTILES]
        stalls = {key: [min(v), float(np.median(v)), max(v)] for key, v in stalls.items()}
        ((part, generation, compiler),) = targets
        part = lookup_device(part)['Part']  # the catalog's name for it, as a platform resolves to it
        return cls(
            part,
            generation,
            compiler,
            sources,
            regions,
            blocks,
            stalls,
            models,
            residuals,
            int(link_latency),
            dict(transport),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({'schema': SCHEMA_VERSION, **asdict(self)}, sort_keys=True) + '\n')

    @classmethod
    def load(cls, path: Path) -> 'KernelProxy':
        data = json.loads(path.read_text())
        if data.pop('schema') != SCHEMA_VERSION:
            raise ValueError(f'{path} holds a cost model of another schema version.')
        return cls(**data)


def cascade_timeline(cycles: float, blocks: int, words: int, reads: bool, writes: bool) -> Timeline:
    """One call of a kernel of a cascade chain whose `cycles` go to `blocks` equal blocks of work: a reader takes
    each block's `words` before the block's work, a writer hands them on after it, each word taking a cycle."""
    work = (cycles - blocks * words * (reads + writes)) / blocks
    events: Timeline = []
    before = 0.0
    for _ in range(blocks):
        if reads:
            events += [(before + 1 if i == 0 else 1.0, 'read') for i in range(words)]
            before = 0.0
        before += work
        if writes:
            events += [(before + 1 if i == 0 else 1.0, 'write') for i in range(words)]
            before = 0.0
    return events + [(before, 'end')]


def _fit(records: Sequence[Mapping]) -> Dict[str, Optional[Dict]]:
    """Per variant, role and group: least-squares coefficients of the feature terms for issue cycles and cascade
    words, or None unless the kernels determine every term with some left over. Cascade words must fit exactly."""
    groups = defaultdict(list)
    for r in records:
        groups[_model(r['variant'], r['role'], r['group'])].append(r)
    models = {}
    for key, members in groups.items():
        wrappers = {r['wrapper'] for r in members}
        if len(wrappers) != 1:
            raise ValueError(f'{key}: wrapper cycles vary within one group and role: {sorted(wrappers)}.')
        matrix = np.array([_terms(r['features']) for r in members], dtype=float)
        # A feature constant over the group adds no term the others do not span, and the range keeps every query at
        # that constant, where the least-squares fit is unique: only the varying features' terms must be determined.
        varying = sum(len({r['features'][n] for r in members}) > 1 for n in members[0]['features'])
        if len(members) <= 2**varying or np.linalg.matrix_rank(matrix) < 2**varying:
            models[key] = None
            continue
        coefficients = np.linalg.lstsq(matrix, np.array([r['static'] for r in members], float), rcond=None)[0]
        moved = np.array([r['words'] for r in members], float)
        words = np.linalg.lstsq(matrix, moved, rcond=None)[0]
        if np.abs(matrix @ words - moved).max() > 1e-6:
            raise ValueError(f'{key}: the cascade words a call moves are no product of its features.')
        ranges = {
            n: [min(r['features'][n] for r in members), max(r['features'][n] for r in members)]
            for n in members[0]['features']
        }
        models[key] = {
            'coefficients': coefficients.tolist(),
            'words': words.tolist(),
            'wrapper': wrappers.pop(),
            'kernels': len(members),
            'ranges': ranges,
        }
    return models


def _within(model: Mapping, record: Mapping) -> bool:
    """Whether a kernel lies in a model's calibrated range, where the model is asked at all."""
    return all(low <= record['features'][n] <= high for n, (low, high) in model['ranges'].items())


def _terms(features: Mapping[str, int]) -> List[float]:
    """Every product of a subset of the features, the empty one (1) first."""
    names = sorted(features)
    return [
        float(math.prod(features[n] for n in subset))
        for size in range(len(names) + 1)
        for subset in itertools.combinations(names, size)
    ]


def _cascade_words(record: Mapping) -> int:
    """The cascade words a compiled kernel moves per call, the same read as written for a middle one."""
    if record['plan'] is None or record['role'] == 'single':
        return 0
    events = [kind for _, kind in timeline(LoopPlan(Loop.from_record(record['plan']), conditional=False), 0.0)]
    counts = {kind: events.count(kind) for kind in ('read', 'write') if kind in events}
    if len(set(counts.values())) > 1:
        raise ValueError(
            f"{record['design']} {record['role']}: reads {counts['read']} words, writes {counts['write']}."
        )
    return next(iter(counts.values()), 0)


def _kernel(record: Mapping) -> tuple:
    """What a calibration kernel contributes: its group, features, static work and wrapper."""
    return (record['group'], record['features'], record['static'], record['wrapper'])


def _fold(record: Mapping) -> int:
    return int(hashlib.sha256(record['coordinate'].encode()).hexdigest(), 16) % FOLDS


def _model(variant: str, role: str, group: str) -> str:
    return f'{variant} {role} {group}'


def _group(variant: str, group: str) -> str:
    return f'{variant} {group}'
