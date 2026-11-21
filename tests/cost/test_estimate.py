from __future__ import annotations

import pytest

pytest.importorskip('onnx')

from aie4ml import from_onnx  # noqa: E402
from aie4ml.cost.estimate import estimate  # noqa: E402
from aie4ml.cost.proxy import KernelProxy  # noqa: E402
from aie4ml.cost.specialization import kernel_specialization, kernel_templates  # noqa: E402
from aie4ml.cost.variants import DESCRIPTORS  # noqa: E402
from aie4ml.op_impls.registry import get_op_impl_registry  # noqa: E402

PART = 'xcvc1902-vsva2197-2mp-e-s'
CHOICE = {'act_bits': 8, 'weight_bits': 8, 'out_bits': 8, 'bias': True, 'relu': True}


def _issue(features):
    return 100 + 2 * features['full_outer'] * features['tile_inner_lhs'] * features['tile_inner_rhs'] // 64


def _dense(tmp_path, k, n, part=PART):
    variant = next(v for v in get_op_impl_registry().candidates('dense') if v.variant_id == 'dense.b.r.v1')
    model, directives, _ = DESCRIPTORS['dense'].space.build('d', {**CHOICE, 'rows': 8, 'k': k, 'n': n}, 1, variant)
    config = {'Part': part, 'AIEConfig': {'BatchSize': 8, 'Iterations': 1}, 'LayerDirectives': directives}
    return from_onnx(model, config, output_dir=tmp_path / f'd{k}_{n}', project_name='d').run_pipeline()


def _artifact(spec) -> KernelProxy:
    """A cost model fitted on kernels of the lowered kernel's group whose issue cycles are `_issue` exactly."""
    records = []
    for rows in (4, 8):
        for k in (48, 96, 192):
            for n in (32, 64, 128):
                features = {'full_outer': rows, 'tile_inner_lhs': k, 'tile_inner_rhs': n}
                records.append(
                    {
                        'design': f'{rows}_{k}_{n}',
                        'part': PART,
                        'generation': 'AIE',
                        'compiler': 'C',
                        'variant': spec.variant_id,
                        'templates': kernel_templates(spec.kernel, spec.parameters),
                        'role': 'single',
                        'group': spec.group,
                        'coordinate': f'{rows}_{k}_{n}',
                        'features': features,
                        'plan': None,
                        'static': _issue(features),
                        'wrapper': 70,
                        'cycles': _issue(features) * 1.02 if (rows, k, n) == (8, 96, 64) else None,
                    }
                )
    transport = {'boundary_in': 50.0, 'boundary_out': 0.0, 'lock': 0.0, 'dma': 0.0}
    return KernelProxy.calibrate(records, {spec.variant_id: {}}, -2, transport)


def test_estimate_composes_the_calibrated_kernel_cost_and_the_exact_tile_use(tmp_path, monkeypatch):
    monkeypatch.delenv('XILINX_VITIS', raising=False)  # no installed compiler to compare the artifact's with
    m = _dense(tmp_path, 64, 32)
    (inst,) = m.context.ir.execution
    spec = kernel_specialization(inst)
    proxy = _artifact(spec)
    e = estimate(m, proxy)
    assert not e.refusals and e.tiles == 1
    kernel = _issue(spec.features) * 1.02 + 70  # its stall share, then its wrapper
    assert e.interval[1] == pytest.approx(kernel)
    assert e.interval[0] <= e.interval[1] <= e.interval[2] and e.latency[1] > e.interval[1]
    beyond = estimate(_dense(tmp_path, 256, 32), proxy)  # K per tile 256: past the calibrated range
    assert beyond.interval is None and 'beyond the calibrated range' in beyond.refusals[0]
    with pytest.raises(TypeError, match='AIEModel'):
        estimate(m.context, proxy)


def test_a_cost_model_serves_every_part_of_its_generation(tmp_path, monkeypatch):
    """Its cycles are the generation's cores: another AIE part takes them, an AIE-ML part is refused."""
    monkeypatch.delenv('XILINX_VITIS', raising=False)
    (inst,) = _dense(tmp_path, 64, 32).context.ir.execution
    proxy = _artifact(kernel_specialization(inst))
    other = estimate(_dense(tmp_path / 'vp2802', 64, 32, part='xcvp2802-vsva5601-2MHP-e-S'), proxy)
    assert not other.refusals and f'calibrated on {PART}' in other.evidence
    newer = estimate(_dense(tmp_path / 've2802', 64, 32, part='xcve2802-vsvh1760-2mp-e-s'), proxy)
    assert newer.interval is None and 'calibrated for AIE, not AIE-ML' in newer.refusals[0]
