from __future__ import annotations

import pytest
from aie4ml.cost.proxy import KernelProxy, cascade_timeline

REGIONS = {'dense.b.r.v1': {'J': 2}}
LINK, TRANSPORT = -2, {'boundary_in': 57.0, 'boundary_out': 0.0, 'lock': 0.0, 'dma': 0.0}
GROUP = '{"flags": 1}'
SOURCES = 'abc'


def _plan(blocks):
    """A first kernel handing on 8 words after each of its blocks."""
    return {
        'hardware': False,
        'trips': 1,
        'items': [5, {'hardware': False, 'trips': blocks, 'items': [3] + ['write'] * 8}],
    }


def _record(Z, J, Kt, role='single', design='d'):
    return {
        'design': design,
        'part': 'xcvc1902-vsva2197-2MP-e-S',
        'generation': 'AIE',
        'compiler': 'C1',
        'variant': 'dense.b.r.v1',
        'templates': 'abc',
        'role': role,
        'group': GROUP,
        'coordinate': f'{{"Z": {Z}, "J": {J}, "Kt": {Kt}}}',
        'features': {'Z': Z, 'J': J, 'Kt': Kt},
        'plan': _plan(Z * J) if role == 'first' else None,
        'static': 10 + 5 * Z + 7 * Z * J + 3 * Z * J * Kt,
        'wrapper': 74,
        'cycles': None,
    }


def _simulated(Z, J, Kt, stall):
    record = _record(Z, J, Kt, design='simulated')
    return {**record, 'cycles': record['static'] * (1 + stall)}


def _calibration(roles=('single',)):
    grid = [_record(Z, J, Kt, role) for role in roles for Z in (1, 2) for J in range(1, 7) for Kt in range(4, 13, 2)]
    return grid + [_simulated(1, 2, 4, 0.0), _simulated(2, 4, 8, 0.1), _simulated(2, 6, 12, 0.2)]


def _proxy(records):
    return KernelProxy.calibrate(records, REGIONS, LINK, TRANSPORT)


def test_the_proxy_recovers_an_unseen_kernel_with_its_groups_stall_and_range(tmp_path):
    _proxy(_calibration()).save(tmp_path / 'proxy.json')
    proxy = KernelProxy.load(tmp_path / 'proxy.json')
    cost = proxy.cost('dense.b.r.v1', SOURCES, 'single', GROUP, {'Z': 2, 'J': 5, 'Kt': 9})
    issue = 10 + 5 * 2 + 7 * 10 + 3 * 10 * 9
    assert cost.refusal is None and cost.wrapper == 74 and cost.cascade is None
    assert cost.cycles == pytest.approx(issue * 1.1)
    assert cost.low == pytest.approx(issue) and cost.high == pytest.approx(issue * 1.2)


def test_the_cascade_words_a_chain_kernel_moves_are_learned_from_its_compiled_kernels():
    proxy = _proxy(_calibration(roles=('single', 'first')))
    cost = proxy.cost('dense.b.r.v1', SOURCES, 'first', GROUP, {'Z': 2, 'J': 5, 'Kt': 9})
    assert proxy.blocks == {'dense.b.r.v1': 8} and cost.cascade == (10, 8)


def test_the_proxy_refuses_what_it_was_not_calibrated_for():
    proxy = _proxy(_calibration())
    for features, reason in [
        ({'Z': 2, 'J': 1, 'Kt': 8}, 'below the calibrated region (J=1 < 2)'),
        ({'Z': 2, 'J': 4, 'Kt': 14}, 'beyond the calibrated range (Kt=14 outside 4..12)'),
    ]:
        cost = proxy.cost('dense.b.r.v1', SOURCES, 'single', GROUP, features)
        assert cost.cycles is None and reason in cost.refusal
    inside = {'Z': 2, 'J': 4, 'Kt': 8}
    assert 'no cost model for last' in proxy.cost('dense.b.r.v1', SOURCES, 'last', GROUP, inside).refusal
    assert 'changed since' in proxy.cost('dense.b.r.v1', 'abd', 'single', GROUP, inside).refusal
    with pytest.raises(ValueError, match='simulated alone'):
        _proxy([r for r in _calibration() if r['cycles'] is None])


def test_a_feature_constant_over_a_group_needs_no_kernels_of_its_own():
    """Every term with Z repeats one without, so only J and Kt must be determined; the range keeps queries at Z=1."""
    proxy = _proxy([r for r in _calibration() if r['features']['Z'] == 1])
    cost = proxy.cost('dense.b.r.v1', SOURCES, 'single', GROUP, {'Z': 1, 'J': 5, 'Kt': 9})
    assert cost.cycles == pytest.approx(10 + 5 + 7 * 5 + 3 * 5 * 9)
    assert 'Z=2 outside 1..1' in proxy.cost('dense.b.r.v1', SOURCES, 'single', GROUP, {'Z': 2, 'J': 4, 'Kt': 8}).refusal


def test_kernels_whose_features_cannot_determine_every_term_get_no_model():
    few = [r for r in _calibration() if r['features']['Z'] == 1 and r['features']['J'] < 4 and r['features']['Kt'] < 8]
    with pytest.raises(ValueError, match='too few kernels'):  # 4 kernels for the 4 terms of J and Kt
        _proxy(few)


def test_a_kernel_costs_the_same_in_any_chain_and_calibration_fails_if_it_did_not():
    records = _calibration(roles=('single', 'first'))
    first = next(r for r in records if r['role'] == 'first')
    again = {**first, 'design': 'wider layer'}  # the same kernel, compiled in a layer of more chains
    assert _proxy([*records, again]).models == _proxy(records).models
    with pytest.raises(ValueError, match='compiled to two kernels: d, wider layer'):
        _proxy([*records, {**again, 'static': again['static'] + 1}])
    with pytest.raises(ValueError, match='moves other cascade words in wider layer'):
        _proxy([*records, {**again, 'plan': _plan(first['features']['Z'] * first['features']['J'] + 1)}])


def test_a_middle_kernel_reads_each_block_before_its_work_and_writes_it_after():
    events = cascade_timeline(100.0, blocks=2, words=2, reads=True, writes=True)
    assert [kind for _, kind in events] == ['read', 'read', 'write', 'write'] * 2 + ['end']
    assert sum(active for active, _ in events) == 100.0
    assert events[0][0] == 1.0 and events[2][0] == (100 - 8) / 2 + 1
