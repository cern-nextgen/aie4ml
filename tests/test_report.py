from __future__ import annotations

import json

import pytest
from aie4ml.report import _aie_clock_ghz, _analyze_aie_out_interval
from aie4ml.report_layout import format_svg_layout, format_terminal_layout, layout_from_pipeline


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


def _layout_pipeline() -> dict:
    def execution(name, variant, cas_num, cas_length):
        return {
            'node': name,
            'op_type': 'test',
            'variant_id': variant,
            'config': {'parallelism': {'contract': 'inner', 'cas_num': cas_num, 'cas_length': cas_length}},
        }

    return {
        'device': 'xcve2802-vsvh1760-2mp-e-s',
        'execution': [
            execution('conv_aie', 'conv2d.buffer.v1', 2, 2),
            execution('dense_aie', 'dense.buffer.v1', 1, 2),
            execution('head_aie', 'dense.buffer.v1', 1, 1),
        ],
        'physical': {
            'placements': {
                'conv_aie': {'col': 7, 'row': 0, 'width': 2, 'height': 2},
                'dense_aie': {'col': 12, 'row': 1, 'width': 2, 'height': 1},
                'head_aie': {'col': 20, 'row': 0, 'width': 1, 'height': 1},
            },
            'plan': {
                'direct_edges': [
                    {'source': 'ifm[0]', 'target': 'conv_aie.in[0]', 'tensor': 'x'},
                    {
                        'source': 'conv_aie.out[0]',
                        'target': 'dense_aie.in[0]',
                        'tensor': 'flat',
                        'realization': 'shared_memory',
                    },
                    {
                        'source': 'conv_aie.out[1]',
                        'target': 'dense_aie.in[1]',
                        'tensor': 'flat',
                        'realization': 'dma',
                    },
                    {'source': 'head_aie.out[0]', 'target': 'ofm[0]', 'tensor': 'y'},
                ],
                'buffers': [
                    {
                        'name': 'buffer_hidden',
                        'writers': [
                            {
                                'source_type': 'op_impl',
                                'source_endpoint': {'op_impl': 'dense_aie'},
                            }
                        ],
                        'readers': [
                            {
                                'target_type': 'op_impl',
                                'target_endpoint': {'op_impl': 'head_aie'},
                            }
                        ],
                    }
                ],
            },
        },
    }


def test_layout_reports_splits_placement_and_graph_level_connections():
    layout = layout_from_pipeline(_layout_pipeline())
    rendered = format_terminal_layout(layout, terminal_width=44)

    assert 'array columns 0-37, rows 0-7  |  7 of 304 tiles occupied (2.3%)' in rendered
    assert 'columns 0-9' in rendered and 'columns 30-37' in rendered
    assert '┌' in rendered and '┘' in rendered and '·' in rendered
    assert 'conv_aie  conv2d.buffer.v1  inner 2×2  4 tiles at (7,0) 2×2' in rendered
    assert 'conv_aie -> 02 dense_aie: shared ×1, DMA ×1' in rendered
    assert 'dense_aie -> 03 head_aie: memtile ×1' in rendered
    assert 'profile' not in rendered and 'make all' not in rendered


def test_layout_svg_contains_the_same_layers_and_connections():
    rendered = format_svg_layout(layout_from_pipeline(_layout_pipeline()))

    assert rendered.startswith('<svg')
    assert 'conv_aie' in rendered and 'dense_aie' in rendered and 'head_aie' in rendered
    assert 'shared ×1, DMA ×1' in rendered or 'DMA ×1, shared ×1' in rendered
    assert 'memtile ×1' in rendered


def test_report_layout_modes_need_no_build_or_profile_artifacts(tmp_path, capsys):
    from aie4ml.report import layout, main

    (tmp_path / 'aie_pipeline.json').write_text(json.dumps(_layout_pipeline()))
    rendered = layout(tmp_path)
    assert 'AIE layout' in repr(rendered)
    assert rendered['device']['part'] == 'xcve2802-vsvh1760-2MP-e-S'  # AMD's spelling, as the catalog gives it
    assert rendered._repr_html_().startswith('<div style="max-width:100%; overflow-x:auto"><svg')

    assert main([str(tmp_path), '--layout']) == 0
    output = capsys.readouterr().out
    assert 'AIE layout' in output and 'Not collected' not in output and 'make profile' not in output

    svg = tmp_path / 'placement.svg'
    assert main([str(tmp_path), '--layout-svg', str(svg)]) == 0
    assert capsys.readouterr().out.strip() == str(svg)
    assert svg.read_text().startswith('<svg')


def test_layout_refuses_overlapping_placements():
    pipeline = _layout_pipeline()
    pipeline['physical']['placements']['dense_aie']['col'] = 8
    with pytest.raises(ValueError, match=r'overlap at tile \(8, 1\)'):
        layout_from_pipeline(pipeline)


def test_layout_explicitly_crops_an_older_pipeline_without_device_geometry():
    pipeline = _layout_pipeline()
    del pipeline['device']
    rendered = format_terminal_layout(layout_from_pipeline(pipeline))
    assert 'occupied region columns 7-20, rows 0-1; device geometry was not emitted' in rendered


def test_layout_paginates_the_full_fifty_column_aie1_array():
    pipeline = _layout_pipeline()
    pipeline['device'] = 'xcvc1902-vsva2197-2mp-e-s'
    rendered = format_terminal_layout(layout_from_pipeline(pipeline), terminal_width=120)

    assert 'array columns 0-49, rows 0-7' in rendered
    assert 'columns 0-24' in rendered and 'columns 25-49' in rendered


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


def test_a_kernel_whose_run_was_inlined_is_charged_its_costliest_member(tmp_path):
    """With `run` inlined, the profile names only the helper doing the work, beside the startup functions."""
    from aie4ml.report import _kernel_cycles

    (tmp_path / 'aie_pipeline.json').write_text(
        json.dumps(
            {
                'physical': {'placements': {'c2_aie': {'col': 4, 'row': 1}}},
                'execution': [{'node': 'c2_aie', 'config': {'parallelism': {'cas_num': 1, 'cas_length': 1}}}],
            }
        )
    )
    profiles = tmp_path / 'aiesimulator_output'
    profiles.mkdir()
    (profiles / 'profile_funct_4_3.txt').write_text(
        'Function profiling report information for ::tl.aie_logical.aie_xtlm.math_engine'
        '.array.tile_4_4.cm.proc.iss\n'
        '  1  3450  31.25%  3450  3450  3450  10991  99.57%  10991  10991  10991  192  717  192 main _main\n'
        '  6  7541  68.31%  1253  1256  1265  7541  68.31%  1253  1256  1265  720  3413  64'
        ' compute _ZN16conv2d_halo_baseI5L3CfgLi0EE7computeEPKh\n'
    )
    (kernel,) = _kernel_cycles(tmp_path, {'aie_tile_row_start': 3})
    assert (kernel['op'], kernel['kernel'], kernel['cycles_per_call']) == ('c2_aie', 'conv2d_halo_base', 1257)


def test_a_kernel_without_a_profile_is_named_not_dropped(tmp_path):
    """aiesim can leave a core unprofiled; the report says so rather than show the design without that stage."""
    (tmp_path / 'aie_pipeline.json').write_text(
        json.dumps(
            {
                'physical': {'placements': {'c1_aie': {'col': 7, 'row': 0}, 'fc_aie': {'col': 8, 'row': 0}}},
                'execution': [
                    {'node': name, 'config': {'parallelism': {'cas_num': 1, 'cas_length': 1}}}
                    for name in ('c1_aie', 'fc_aie')
                ],
            }
        )
    )
    profiles = tmp_path / 'aiesimulator_output'
    profiles.mkdir()
    (profiles / 'profile_funct_7_0.txt').write_text(
        'Function profiling report information for ::tl.aie_logical.aie_xtlm.math_engine'
        '.array.tile_7_1.cm.proc.iss\n'
        '  6  7374  76.88%  1229  1229  1229  7374  76.88%  1229  1229  1229  704  1245  0'
        ' run _ZN13conv2d_singleI5L1Cfg\n'
    )
    compiler_report = tmp_path / 'Work' / 'reports' / 'compiler_report.json'
    compiler_report.parent.mkdir(parents=True)
    compiler_report.write_text(json.dumps({'aie_driver_config': {'aie_tile_row_start': 1}}))
    from aie4ml.report import report

    missing = report(tmp_path)['missing']
    assert any(item.startswith('per-kernel cycles of fc_aie:') for item in missing)
