from __future__ import annotations

import dataclasses

from aie4ml.cost.calibrate import Plan
from aie4ml.cost.variants import DESCRIPTORS
from aie4ml.op_impls.registry import get_op_impl_registry


def test_calibration_picks_legal_points_inside_the_region_and_holds_validation_points_out(tmp_path):
    space = dataclasses.replace(
        DESCRIPTORS['dense'].space,
        choices={'act_bits': (8,), 'weight_bits': (8, 16), 'out_bits': (8,), 'bias': (True,), 'relu': (True,)},
        shape={'rows': (4, 8), 'k': (16, 48, 96, 192), 'n': (16, 32, 64)},
    )
    variant = next(v for v in get_op_impl_registry().candidates('dense') if v.variant_id == 'dense.b.r.v1')
    region = {'tile_inner_lhs': 48, 'tile_inner_rhs': 32}
    plan = Plan(space, variant, 'xcvc1902-vsva2197-2mp-e-s', 'C', region, tmp_path, jobs=2)
    (group,) = plan.groups({})  # int8 x int16 weights: refused by lowering
    plan.extend(group, 1, fit=6, validate=2, least=True)
    picked = plan.picked[(group, 1)]
    assert [d['features'] for d in picked['least']] == [{'full_outer': 4, 'tile_inner_lhs': 16, 'tile_inner_rhs': 16}]
    fit, validate = ([tuple(d['features'].values()) for d in picked[kind]] for kind in ('fit', 'validate'))
    assert len(fit) == 6 and len(validate) == 2 and not set(fit) & set(validate)
    assert all(k >= 48 and n >= 32 for _, k, n in fit + validate)
    assert {(4, 48, 32), (8, 192, 32)} <= set(fit)  # the region's corners come first


def _broken_build(job, out):
    raise RuntimeError('Make target "compile" failed')


def test_a_design_vitis_fails_to_build_is_refused_once_and_calibration_goes_on(tmp_path, monkeypatch):
    from aie4ml.cost import calibrate

    space = dataclasses.replace(
        DESCRIPTORS['dense'].space,
        choices={'act_bits': (8,), 'weight_bits': (8,), 'out_bits': (8,), 'bias': (True,), 'relu': (True,)},
        shape={'rows': (4,), 'k': (48, 96), 'n': (32,)},
    )
    variant = next(v for v in get_op_impl_registry().candidates('dense') if v.variant_id == 'dense.b.r.v1')
    plan = Plan(space, variant, 'xcvc1902-vsva2197-2mp-e-s', 'C', {}, tmp_path / 'designs', jobs=1)
    (group,) = plan.groups({})
    plan.extend(group, 1, fit=2, validate=0, least=False)
    evidence = calibrate.Evidence(tmp_path / 'evidence.json')
    monkeypatch.setattr(calibrate, '_measure', _broken_build)
    failed = calibrate._build(plan, plan.designs(), evidence, threads=1, simulate=False)
    assert failed == {d['name'] for d in plan.designs()} and len(failed) == 2
    assert all('compile' in kernel['refusal'] for kernel in evidence.kernels.values())
    assert (
        calibrate._build(plan, plan.designs(), calibrate.Evidence(evidence.path), threads=1, simulate=False) == failed
    )
    plan.drop(failed)
    assert plan.designs() == []
