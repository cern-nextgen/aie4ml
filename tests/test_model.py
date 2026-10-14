"""AIEModel's project lifecycle: a model builds and simulates the project it planned itself."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from aie4ml import design_directives
from aie4ml import model as aie_model
from aie4ml.frontends.onnx import from_onnx
from frontends.test_onnx_aie1 import _dense_model
from helpers import PART
from ops.test_conv2d import AIE1_PART, _strided_chain_model


def test_a_model_builds_its_own_project_over_another_in_the_same_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(aie_model.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=''))
    config = {'Part': PART, 'AIEConfig': {'Iterations': 1}}

    first = from_onnx(_dense_model(out_features=8), config, output_dir=tmp_path, project_name='proj')
    first.build()
    second = from_onnx(_dense_model(in_features=32, out_features=16), config, output_dir=tmp_path, project_name='proj')
    with pytest.raises(RuntimeError, match='Run write'):
        second.write_inputs(np.zeros((8, 32), np.int8), quantize_in=False)
    second.build()

    execution = json.loads((tmp_path / 'aie_pipeline.json').read_text())['execution']
    assert [inst['port_views']['y']['logical'] for inst in execution] == [[8, 16]]
    (port,) = second.write_inputs(np.zeros((8, 32), np.int8), quantize_in=False).inputs['x_q']
    assert port.staging['io_boundary_dimension'] == [32, 8]


def test_a_configuration_changed_after_building_is_built_again(tmp_path, monkeypatch):
    monkeypatch.setattr(aie_model.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=''))
    config = {'Part': PART, 'AIEConfig': {'Iterations': 1, 'Optimize': 'resource'}}
    model = from_onnx(_dense_model(in_features=64, out_features=64), config, output_dir=tmp_path, project_name='proj')
    model.build()

    model.context.aie_config['Optimize'] = 'throughput'
    with pytest.raises(RuntimeError, match='as configured now'):
        model.write_inputs(np.zeros((8, 64), np.int8), quantize_in=False)
    model.build()
    assert json.loads((tmp_path / 'aie_pipeline.json').read_text())['optimizer']['mode'] == 'throughput'

    model.context.reset_ir()
    assert model.context.emitted is None


def test_a_recorded_design_is_built_again_without_a_search(tmp_path):
    """The design a project records (aie_pipeline.json), pinned as directives, is built again place for place --
    the retiler the compiler adds too -- and a directive the record contradicts is refused."""
    config = {'Part': AIE1_PART, 'AIEConfig': {'Iterations': 1}}
    from_onnx(_strided_chain_model(), config, output_dir=tmp_path / 'first', project_name='proj').write()
    recorded = tmp_path / 'first' / 'aie_pipeline.json'
    pinned = {**config, 'LayerDirectives': design_directives(recorded)}
    again = from_onnx(_strided_chain_model(), pinned, output_dir=tmp_path / 'again', project_name='proj')
    again.write()

    replayed = tmp_path / 'again' / 'aie_pipeline.json'
    first, second = (json.loads(path.read_text())['physical'] for path in (recorded, replayed))
    assert 'second_aie_retile' in first['placements'] and first['placements'] == second['placements']
    assert first['plan']['direct_edges'] == second['plan']['direct_edges']
    assert again.context.ir.optimizer['designs_tried'] == 1
    with pytest.raises(ValueError, match='contradicts the recorded design'):
        design_directives(recorded, {'first': {'placement': {'col': 99, 'row': 0}}})


@pytest.mark.parametrize('frac', [3, 0, -2])
def test_fixed_point_scaling_holds_for_a_step_coarser_than_one(frac):
    """A negative frac (bit-exact types of coarse constants) scales by 2**frac like any other, never by 1."""
    from aie4ml.aie_types import AIEDataType, RoundingMode, SaturationMode
    from aie4ml.op_impls.families.matmul.common import quantize_to_int
    from aie4ml.simulation import _quantize_to_int

    values = np.array([-8.0, 4.0, 12.0])
    codes = np.ldexp(values, frac).astype(np.int64)
    assert np.array_equal(quantize_to_int(values, frac, 8, True, RoundingMode.RND, SaturationMode.SAT), codes)
    assert np.array_equal(_quantize_to_int(values, AIEDataType('int8', frac=frac)), codes)
