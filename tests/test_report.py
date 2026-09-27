from __future__ import annotations

import json

import pytest
from aie4ml.report import _aie_clock_ghz, _analyze_aie_out_interval


def _pipeline(elements: int) -> dict:
    return {
        'physical': {
            'plan': {
                'io_ports': [
                    {
                        'direction': 'output',
                        'port': 0,
                        'tensor': 'y',
                        'staging': {'io_tiling_dimension': [elements]},
                    }
                ]
            }
        }
    }


@pytest.mark.parametrize(
    'frames',
    [
        [(10, 8), (20, 8), (30, 8)],
        [(8, 4), (10, 4), (18, 4), (20, 4), (28, 4), (30, 4)],
    ],
    ids=['one-frame-per-inference', 'two-frames-per-inference'],
)
def test_report_groups_tlast_frames_into_logical_inferences(tmp_path, frames):
    data_dir = tmp_path / 'aiesimulator_output' / 'data'
    data_dir.mkdir(parents=True)
    lines = []
    for timestamp, elements in frames:
        lines.extend([f'T {timestamp} ns', 'TLAST', ' '.join(['1'] * elements)])
    (data_dir / 'y_p0.txt').write_text('\n'.join(lines) + '\n')

    latency = _analyze_aie_out_interval(tmp_path, _pipeline(8))

    assert latency['global'] == {'min_ns': 10.0, 'max_ns': 10.0, 'avg_ns': 10.0, 'samples': 2}
    assert latency['per_port']['y_p0.txt'] == latency['global']


def test_report_reads_the_compiled_aie_clock(tmp_path):
    """Latencies convert to cycles at the clock the design was compiled for, not an assumed one."""
    assert _aie_clock_ghz(tmp_path) is None
    config = tmp_path / 'Work' / 'ps' / 'c_rts' / 'aie_control_config.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'aie_metadata': {'DeviceData': {'AIEFrequency': 1000}}}))
    assert _aie_clock_ghz(tmp_path) == 1.0


def test_report_latency_runs_from_the_first_input(tmp_path):
    """Latency = the host's input-start to output-start cycles plus the first sample's own output
    duration, at the compiled clock: simulation start (configuration, weights) does not count."""
    from aie4ml.report import measured_latency_cc

    config = tmp_path / 'Work' / 'ps' / 'c_rts' / 'aie_control_config.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'aie_metadata': {'DeviceData': {'AIEFrequency': 1250}}}))
    data_dir = tmp_path / 'aiesimulator_output' / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / 'y_p0.txt').write_text('T 5000 ns\n1 1\nT 5080 ns\nTLAST\n1 1\nT 6000 ns\nTLAST\n1 1\n')
    assert measured_latency_cc(tmp_path) is None  # no host measurement yet
    (tmp_path / 'log').write_text('AIE4ML_LATENCY_START_CC 900\n...\nAIE4ML_LATENCY_START_CC 1000\n')
    assert measured_latency_cc(tmp_path) == 1000 + 100  # the last run's count, plus 80 ns at 1.25 GHz


def test_a_profile_is_charged_to_the_core_its_tile_names(tmp_path):
    """AIE-MLv2 names the file of core (7, 0) profile_funct_7_2: the tile inside, less the design's first
    core row, is the core its op was placed on."""
    from aie4ml.report import _kernel_cycles

    (tmp_path / 'aie_pipeline.json').write_text(
        json.dumps(
            {
                'physical': {'placements': {'c1_aie': {'col': 7, 'row': 0}}},
                'execution': [{'node': 'c1_aie', 'config': {'parallelism': {'cas_num': 1, 'cas_length': 1}}}],
            }
        )
    )
    profiles = tmp_path / 'aiesimulator_output'
    profiles.mkdir()
    (profiles / 'profile_funct_7_2.txt').write_text(
        'Function profiling report information for ::tl.aie_logical.aie_xtlm.math_engine'
        '.array.tile_7_3.cm.proc.iss\n'
        '  6  7374  76.88%  1229  1229  1229  7374  76.88%  1229  1229  1229  704  1245  0'
        ' run _ZN13conv2d_singleI5L1Cfg\n'
    )
    (kernel,) = _kernel_cycles(tmp_path, {'aie_tile_row_start': 3})
    assert (kernel['tile'], kernel['op'], kernel['cycles_per_call']) == ('7_0', 'c1_aie', 1229)
