"""aie4ml-calibrate: a variant's cost model on a device part, from kernels it generates, compiles and measures.

    aie4ml-calibrate --variant dense.b.r.v1 --target xcvc1902-vsva2197-2MP-e-S --jobs 44 \\
        --region tile_inner_lhs=48,tile_inner_rhs=32

Run where Vitis is installed. It lowers every discrete choice of the op type's calibration space (aie4ml.cost.variants)
to find its code groups; per group and chain length it samples shape points far apart among those lowering accepts,
inside the region, holding some out; compiles each kernel once ever (the evidence cache); simulates a few for stalls;
adds points where held-out kernels miss the `--gate`; then fits the target's cascade link and handoff setups through
`estimate` on simulated chains and multi-layer models. It writes the generation's artifact into the package
(`ARTIFACTS`).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import signal
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..device_catalog import resolve_device
from ..op_impls.registry import get_op_impl_registry
from ..passes.utils import sanitize_identifier
from .estimate import ARTIFACTS, estimate
from .listing import ScheduleUnavailable, loop_plan, wrapper_bundles
from .probe import compiled_kernels, run_profile
from .proxy import KernelProxy, _terms, _within
from .proxy import _model as _model_key
from .specialization import installed_compiler, kernel_specialization, kernel_templates
from .variants import DESCRIPTORS

FIT, VALIDATE, GROW = 16, 4, 8  # points per group and chain length: first fit and validation, and more per round
ITERATIONS = 6  # graph iterations simulated: enough for a steady interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--target', required=True, help='a device part or platform of aie_devices.json')
    parser.add_argument('--region', default='', help='FEATURE=LEAST[,...]: where the compiled schedules are regular')
    parser.add_argument('--only', default='', help='CHOICE=VALUE[,...]: calibrate only models making these choices')
    parser.add_argument('--jobs', type=int, default=4, help='designs built at once')
    parser.add_argument('--threads', type=int, default=2, help='compile threads per design')
    parser.add_argument('--gate', type=float, default=0.10, help='largest relative error on validation kernels')
    parser.add_argument(
        '--rounds', type=int, default=6, help='rounds adding points; groups still over the gate are left out'
    )
    parser.add_argument('--workdir', type=Path, default=Path.home() / '.cache' / 'aie4ml' / 'calibration')
    args = parser.parse_args()
    os.setpgrp()  # the builds' compilers and simulators too, so stopping this stops them
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, _stop)
    region = {k: int(v) for k, v in _pairs(args.region)}
    only = {k: json.loads(v) for k, v in _pairs(args.only)}
    calibrate(args.variant, args.target, region, only, args.workdir, args.jobs, args.threads, args.gate, args.rounds)


def _stop(number, _frame) -> None:
    signal.signal(number, signal.SIG_DFL)
    os.killpg(0, number)


def _pairs(text: str) -> List[Tuple[str, str]]:
    return [tuple(item.split('=')) for item in text.split(',') if item]


def calibrate(
    variant_id: str,
    target: str,
    region: Dict[str, int],
    only: Dict[str, Any],
    workdir: Path,
    jobs: int,
    threads: int,
    gate: float,
    rounds: int,
) -> Path:
    device, _ = resolve_device(target, {})
    compiler = _compiler(device.generation)
    variant = _variant(variant_id)
    space = DESCRIPTORS[variant.op_type].space
    work = workdir / device.part / compiler
    evidence = Evidence(work / 'evidence.json')
    plan = Plan(space, variant, device.part, compiler, region, work / 'designs', jobs)

    groups = plan.groups(only)
    print(f'{variant_id}: {len(groups)} code groups', flush=True)
    for length in space.lengths:
        for group in groups:
            plan.extend(group, length, FIT, VALIDATE, least=True)
    for round_ in range(rounds):
        plan.drop(_build(plan, plan.designs(), evidence, threads, simulate=False))
        plan.drop(_build(plan, plan.stall_designs(), evidence, threads, simulate=True))
        records = plan.records(evidence, compiler)
        failing = _gate(records, plan, region, gate)
        print(
            f'round {round_ + 1}: {len(records)} kernels; groups over the {gate:.0%} gate: {len(failing)}', flush=True
        )
        if not failing or round_ == rounds - 1:
            break
        for group, length in failing:
            plan.extend(group, length, GROW, VALIDATE // 2, least=False)
    excluded = sorted(
        {
            _model_key(k['variant'], k['role'], k['group'])
            for key in failing
            for d in plan.designs_of(key)
            for k in d['kernels']
        }
    )
    for key in excluded:  # never estimated: compile required
        print(f'  over the gate after {rounds} rounds, left out: {key}', flush=True)

    fitted = work / 'fitted'
    fitted.mkdir(parents=True, exist_ok=True)
    (fitted / f'{variant_id}.json').write_text(json.dumps({'region': region, 'records': records, 'excluded': excluded}))
    link_latency, transport = _fit_transport(plan, evidence, threads, fitted)
    (work / 'transport.json').write_text(json.dumps({'link_latency': link_latency, 'transport': transport}))
    proxy = _artifact(fitted, link_latency, transport)
    path = ARTIFACTS / f'{proxy.generation.lower()}.json'
    proxy.save(path)
    print(f'{path}: {sorted(proxy.sources)}, link latency {link_latency}, transport {transport}', flush=True)
    return path


class Evidence:
    """Compiled and simulated kernels by specialization key: what compiling and simulating showed, nothing derived
    from lowering. Kept across runs; saved after every design."""

    def __init__(self, path: Path):
        self.path = path
        self.kernels: Dict[str, Dict] = json.loads(path.read_text()) if path.exists() else {}

    def has(self, keys: List[str], simulated: bool) -> bool:
        return all(
            key in self.kernels
            and (not simulated or self.kernels[key]['cycles'] is not None or 'refusal' in self.kernels[key])
            for key in keys
        )

    def refused(self, design: Dict) -> bool:
        return any('refusal' in self.kernels.get(k['key'], {}) for k in design['kernels'])

    def add(self, found: Dict[str, Dict]) -> None:
        for key, kernel in found.items():
            if self.kernels.get(key, {}).get('cycles') is not None and kernel['cycles'] is None:
                continue  # never forget a measurement for a compile-only rebuild
            self.kernels[key] = kernel
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_suffix('.partial')
        part.write_text(json.dumps(self.kernels))
        os.replace(part, self.path)


class Plan:
    """Per code group and chain length, the designs calibration builds: points sampled far apart among the
    legal lattice points, each kept as what lowering showed of it and rebuilt when compiled."""

    def __init__(self, space, variant, part: str, compiler: str, region: Dict[str, int], out: Path, jobs: int):
        self.space, self.variant, self.part, self.compiler = space, variant, part, compiler
        self.region, self.out, self.jobs = region, out, jobs
        self.lattice = [dict(zip(space.shape, values)) for values in itertools.product(*space.shape.values())]
        self.spans = {a: (math.log2(min(v)), math.log2(max(v))) for a, v in space.shape.items()}
        self.choices: Dict[str, Dict[str, Any]] = {}  # group -> the choices that reach it
        self.picked: Dict[Tuple[str, int], Dict[str, List[Dict]]] = {}  # (group, length) -> use -> designs
        self.surveyed: Dict[Tuple[str, int], List[Dict]] = {}  # (group, length) -> its legal designs

    def groups(self, only: Dict[str, Any]) -> List[str]:
        for values in itertools.product(*self.space.choices.values()):
            choice = dict(zip(self.space.choices, values))
            if any(choice[name] != value for name, value in only.items()):
                continue
            for point in sorted(self.lattice, key=lambda p: tuple(p.values()))[:8]:
                design = _survey(self._job(choice, point, 1))
                if design is not None:
                    self.choices.setdefault(design['group'], choice)
                    break
        return list(self.choices)

    def extend(self, group: str, length: int, fit: int, validate: int, least: bool) -> None:
        """Pick `fit` more fitting points and `validate` more validation points inside the region for a group and
        chain length, each farthest from those it has; and, first, the lattice's least legal point."""
        legal = self._survey(group, length)
        picked = self.picked.setdefault((group, length), {'fit': [], 'validate': [], 'least': []})
        if least and legal:
            picked['least'].append(min(legal, key=lambda d: (sum(d['point'].values()), tuple(d['point'].values()))))
        inside = [d for d in legal if all(d['features'][n] >= least for n, least in self.region.items())]
        for kind, count in (('fit', fit), ('validate', validate)):
            for _ in range(count):
                taken = [d for designs in picked.values() for d in designs]
                names = {d['name'] for d in taken}
                candidates = [d for d in inside if d['name'] not in names]
                if not candidates:
                    return
                picked[kind].append(self._farthest(taken, candidates))

    def model(self, design: Dict):
        """A design's ONNX model, directives and simulation inputs."""
        if 'model' in design:
            return design['model'], design['directives'], design['feeds']
        return self.space.build(design['name'], {**design['choice'], **design['point']}, design['length'], self.variant)

    def drop(self, names: set) -> None:
        """Forget designs that failed to build."""
        for picked in self.picked.values():
            for use, designs in picked.items():
                picked[use] = [d for d in designs if d['name'] not in names]

    def designs_of(self, key: Tuple[str, int]) -> List[Dict]:
        return [d for designs in self.picked.get(key, {}).values() for d in designs]

    def designs(self) -> List[Dict]:
        return [d for picked in self.picked.values() for designs in picked.values() for d in designs]

    def stall_designs(self) -> List[Dict]:
        """Per group, the single kernels of least and most work it fits on."""
        out = []
        for (group, length), picked in self.picked.items():
            if length == 1 and picked['fit']:
                work = sorted(picked['fit'], key=lambda d: math.prod(d['features'].values()))
                out += [dict(work[0], simulate=True), dict(work[-1], simulate=True)]
        return out

    def records(self, evidence: Evidence, compiler: str) -> List[Dict]:
        """Every picked kernel's record: what lowering derives, with what compiling and simulating showed."""
        records = []
        for picked in self.picked.values():
            for kind, designs in picked.items():
                for design in designs:
                    for kernel in design['kernels']:
                        found = evidence.kernels.get(kernel['key'])
                        if found is not None:
                            records.append(
                                {**kernel, **found, 'design': design['name'], 'use': kind, 'compiler': compiler}
                            )
        return records

    def _survey(self, group: str, length: int) -> List[Dict]:
        if (group, length) not in self.surveyed:
            jobs = [self._job(self.choices[group], point, length) for point in self.lattice]
            with ProcessPoolExecutor(max_workers=self.jobs) as pool:
                designs = list(pool.map(_survey, jobs, chunksize=16))
            self.surveyed[(group, length)] = [d for d in designs if d is not None and d['group'] == group]
        return self.surveyed[(group, length)]

    def _job(self, choice: Dict[str, Any], point: Dict[str, int], length: int) -> Tuple:
        return (self.variant.variant_id, self.part, self.compiler, choice, point, length, self.out)

    def _farthest(self, taken: List[Dict], candidates: List[Dict]) -> Dict:
        """The candidate farthest from the taken points in the log-scaled lattice; the least one when none is taken;
        ties to the least."""
        order = sorted(candidates, key=lambda d: tuple(d['point'].values()))
        if not taken:
            return order[0]
        at = np.array([self._position(d['point']) for d in order])
        to = np.array([self._position(d['point']) for d in taken])
        distance = np.linalg.norm(at[:, None, :] - to[None, :, :], axis=2).min(axis=1)
        return order[int(np.argmax(distance))]

    def _position(self, point: Dict[str, int]) -> List[float]:
        return [(math.log2(point[a]) - lo) / ((hi - lo) or 1) for a, (lo, hi) in self.spans.items()]


def _survey(job: Tuple) -> Optional[Dict]:
    """One choice, shape point and chain length of a variant, lowered: what its kernels are, or None if lowering
    refuses it or it selects another variant."""
    variant_id, part, compiler, choice, point, length, out = job
    warnings.filterwarnings('ignore')
    variant = _variant(variant_id)
    name = f'{variant.op_type}_{_digest([choice, point, length])}'
    model, directives, feeds = DESCRIPTORS[variant.op_type].space.build(name, {**choice, **point}, length, variant)
    try:
        m = _lowered(model, directives, part, name, out, feeds)
    except (NotImplementedError, ValueError, RuntimeError):
        return None
    kernels = []
    for inst in m.context.ir.execution:
        spec = kernel_specialization(inst)
        if spec.variant_id != variant_id:
            return None
        templates = _templates(spec.kernel, spec.parameters)
        kernels += [
            {
                'instance': sanitize_identifier(inst.name),
                'variant': spec.variant_id,
                'role': role,
                'key': spec.key(role, part, compiler),
                'group': spec.group,
                'coordinate': spec.coordinate,
                'features': spec.features,
                'templates': templates,
            }
            for role in sorted(set(spec.roles))
        ]
    features = kernels[0]['features']
    return {'name': name, 'choice': choice, 'point': point, 'length': length, 'group': kernels[0]['group'],
            'features': features, 'kernels': kernels}  # fmt: skip


@lru_cache(maxsize=None)
def _variant(variant_id: str):
    variant = next(
        (v for op in DESCRIPTORS for v in get_op_impl_registry().candidates(op) if v.variant_id == variant_id), None
    )
    if variant is None:
        raise ValueError(f'{variant_id}: no registered variant of an op type with a cost descriptor.')
    return variant


@lru_cache(maxsize=None)
def _templates(kernel: str, parameters: str) -> str:
    return kernel_templates(kernel, parameters)


def _gate(records: List[Dict], plan: Plan, region: Dict[str, int], gate: float) -> List[Tuple[str, int]]:
    """Groups and chain lengths whose validation kernels a fit on the others misses by more than `gate`, or cannot
    estimate at all."""
    fit = [r for r in records if r['use'] != 'validate']
    proxy = KernelProxy.calibrate(fit, {plan.variant.variant_id: region}, 0, dict.fromkeys(TRANSPORT, 0.0))
    failing = set()
    for (group, length), picked in plan.picked.items():
        for design in picked['validate']:
            for kernel in design['kernels']:
                model = proxy.models.get(_model_key(kernel['variant'], kernel['role'], kernel['group']))
                static = next((r['static'] for r in records if r['key'] == kernel['key']), None)
                if static is None or model is None or not _within(model, kernel):
                    failing.add((group, length))
                elif abs(float(np.dot(model['coefficients'], _terms(kernel['features']))) / static - 1) > gate:
                    failing.add((group, length))
    return sorted(failing)


TRANSPORT = ('boundary_in', 'boundary_out', 'lock', 'dma')


def _fit_transport(plan: Plan, evidence: Evidence, threads: int, fitted: Path):
    """Fit the cascade link latency on simulated chains, the boundary setup on simulated single kernels and the
    lock and DMA setups on the handoff models, each through `estimate`."""
    chains = [
        dict(d, simulate=True) for (_, length), picked in plan.picked.items() if length > 1 for d in picked['fit'][:1]
    ]
    singles = plan.stall_designs()
    handoffs, models = [], {}
    fitted_so_far = _artifact(fitted, 0, dict.fromkeys(TRANSPORT, 0.0))
    for choice in plan.choices.values():  # the first calibrated choice whose handoff models the cost model covers
        for name, (model, directives, feeds) in plan.space.handoffs(plan.variant, choice).items():
            try:
                models[name] = _lowered(model, directives, plan.part, name, plan.out, feeds)
            except (NotImplementedError, ValueError, RuntimeError) as error:
                print(f'  {name}: dropped, lowering refuses it: {error}', flush=True)
                continue
            if estimate(models[name], fitted_so_far).refusals:
                continue
            handoffs.append({'name': name, 'model': model, 'directives': directives, 'feeds': feeds, 'kernels': []})
        if len(handoffs) >= 2:
            break
        handoffs = []
    if len(handoffs) < 2:
        raise RuntimeError(f'{plan.variant.op_type}: {len(handoffs)} usable handoff models, too few to fit transport.')
    failed = _build(plan, chains + singles + handoffs, evidence, threads, simulate=True, projects=True)
    chains, singles, handoffs = ([d for d in ds if d['name'] not in failed] for ds in (chains, singles, handoffs))
    measured = {d['name']: _measured(plan.out / d['name']) for d in chains + singles + handoffs}
    unmeasured = sorted(name for name, found in measured.items() if found is None)
    if unmeasured:
        raise RuntimeError(f'simulated but no latency or interval to read, even rebuilt: {unmeasured}')
    for d in chains + singles:
        models[d['name']] = _lowered(*plan.model(d)[:2], plan.part, d['name'], plan.out, plan.model(d)[2])

    def error(names, quantity, link_latency, transport) -> float:
        proxy = _artifact(fitted, link_latency, transport)
        errors = []
        for name in names:
            e = estimate(models[name], proxy)
            if e.refusals:
                raise RuntimeError(f'{name}: the calibrated cost model refuses it: {e.refusals}')
            errors.append(abs(getattr(e, quantity)[1] / measured[name][quantity] - 1))
        return float(np.median(errors))

    zero = dict.fromkeys(TRANSPORT, 0.0)
    link_latency = min(range(-4, 5), key=lambda latency: error([d['name'] for d in chains], 'interval', latency, zero))
    residuals = []
    for d in singles:
        e = estimate(models[d['name']], _artifact(fitted, link_latency, zero))
        residuals.append(measured[d['name']]['latency'] - e.latency[1])
    boundary = float(np.median(residuals))
    names = [d['name'] for d in handoffs]
    lock, dma = min(
        itertools.product(range(0, 201, 10), range(0, 401, 10)),
        key=lambda s: error(
            names, 'latency', link_latency, {**zero, 'boundary_in': boundary, 'lock': s[0], 'dma': s[1]}
        ),
    )
    return link_latency, {**zero, 'boundary_in': boundary, 'lock': float(lock), 'dma': float(dma)}


def _artifact(fitted: Path, link_latency: int, transport: Dict[str, float]) -> KernelProxy:
    """The cost model of every variant calibrated for the part, each on the records its last run fitted."""
    runs = {path.stem: json.loads(path.read_text()) for path in sorted(fitted.glob('*.json'))}
    records = [r for run in runs.values() for r in run['records']]
    regions = {v: run['region'] for v, run in runs.items()}
    excluded = {key for run in runs.values() for key in run['excluded']}
    return KernelProxy.calibrate(records, regions, link_latency, transport, excluded)


_REFUSED = {'part': None, 'generation': None, 'plan': None, 'static': None, 'wrapper': None, 'cycles': None}


def _build(
    plan: Plan, designs: List[Dict], evidence: Evidence, threads: int, simulate: bool, projects: bool = False
) -> set:
    """Build each design whose kernels the evidence lacks -- compile-only, or simulated -- and add what it showed;
    with `projects`, also each simulated design whose measurement cannot be read. Returns the designs that failed,
    ever."""
    todo = {}
    out = evidence.path.parent / 'designs'
    for design in designs:
        simulated = simulate or design.get('simulate', False)
        known = design['kernels'] and evidence.has([k['key'] for k in design['kernels']], simulated)
        gone = projects and not evidence.refused(design) and _measured(out / design['name']) is None
        if known and not gone:
            continue
        todo[design['name']] = (design, simulated)
    failed = set()
    with ProcessPoolExecutor(max_workers=plan.jobs) as pool:
        futures = {
            pool.submit(_measure, (d['name'], *plan.model(d), s, threads, plan.part), out): d for d, s in todo.values()
        }
        for done, future in enumerate(as_completed(futures), start=1):
            design = futures[future]
            try:
                found = future.result()
            except Exception as error:  # a design Vitis fails to build is evidence too: never built again
                found = {k['key']: {**_REFUSED, 'refusal': str(error)[-500:]} for k in design['kernels']}
                failed.add(design['name'])
            evidence.add(found)
            outcome = 'failed' if design['name'] in failed else f'{len(found)} kernels'
            print(f"  [{done}/{len(futures)}] {design['name']}: {outcome}", flush=True)
    return failed | {d['name'] for d in designs if d['kernels'] and evidence.refused(d)}


def _measure(job, out: Path) -> Dict[str, Dict]:
    """Build one design -- compiled, and simulated if asked -- and read each kernel's schedule and cycles."""
    name, model, directives, feeds, simulate, threads, part = job
    warnings.filterwarnings('ignore')
    shutil.rmtree(out / name, ignore_errors=True)
    m = _lowered(model, directives, part, name, out, feeds)
    m.build('all' if simulate else 'compile', jobs=threads, log_to_stdout=False)
    if simulate:
        m.predict(feeds, simulator='aie', quantize_in=False, dequantize_out=False, aie_profile=True)
    device = m.context.device
    compiler = _compiler(device.generation)
    project = out / name
    report = json.loads((project / 'Work' / 'reports' / 'compiler_report.json').read_text())
    instances = {sanitize_identifier(inst.name): inst for inst in m.context.ir.execution}
    found = {}
    for kernel in compiled_kernels(report):
        spec = kernel_specialization(instances[kernel.graph])
        listing = project / 'Work' / 'aie' / kernel.tile / 'Release' / f'{kernel.tile}.lst'
        try:
            plan = loop_plan(listing, device.generation)
            static, wrapper = (None if plan.conditional else plan.cycles()), wrapper_bundles(listing, device.generation)
        except ScheduleUnavailable:
            plan, static, wrapper = None, None, None
        cycles = None
        if simulate and static is not None:
            profile = run_profile(project, kernel.tile)
            if profile['instructions'] != static * profile['calls']:
                raise RuntimeError(f'{name} {kernel.tile}: profiled instructions do not match its loop plan.')
            cycles = profile['cycles'] / profile['calls']
        found[spec.key(kernel.role, device.part, compiler)] = {
            'part': device.part,
            'generation': device.generation,
            'plan': None if plan is None else plan.root.record(),
            'static': static,
            'wrapper': wrapper,
            'cycles': cycles,
        }
    if not simulate:  # its kernels are evidence now; only simulated projects are read again
        shutil.rmtree(project)
    return found


def _measured(project: Path) -> Optional[Dict[str, float]]:
    """A simulated project's latency and interval, or None where it holds no simulation that yields both (a run
    stopped or a log cut short)."""
    from ..report import report

    if not (project / 'aiesimulator_output').exists():
        return None
    latency = report(project)['latency']
    cycles, interval = latency.get('latency_cc'), (latency.get('global') or {}).get('avg_cc')
    return None if cycles is None or interval is None else {'latency': float(cycles), 'interval': float(interval)}


def _lowered(model, directives, part: str, name: str, out: Path, feeds):
    from ..frontends.onnx import from_onnx

    rows = next(iter(feeds.values())).shape[0]
    config = {'Part': part, 'AIEConfig': {'BatchSize': rows, 'Iterations': ITERATIONS}, 'LayerDirectives': directives}
    return from_onnx(model, config, output_dir=out / name, project_name=name).run_pipeline()


@lru_cache(maxsize=None)
def _compiler(generation: str) -> str:
    compiler = installed_compiler(generation)
    if compiler is None:
        raise RuntimeError('calibration compiles kernels: source the Vitis settings first.')
    return compiler


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


if __name__ == '__main__':
    main()
