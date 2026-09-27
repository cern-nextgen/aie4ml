# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os

from hls4ml.backends.backend import Backend, extract_optimizers_from_path
from hls4ml.backends.fpga.fpga_backend import FPGABackend as _FPGABackendHelper
from hls4ml.model.attributes import Attribute, ConfigurableAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import Activation, Dense
from hls4ml.model.optimizer import layer_optimizer, model_optimizer
from hls4ml.model.optimizer.optimizer import ModelOptimizerPass
from hls4ml.writer import get_writer

from ...device_catalog import PART_HELP, known_boards, lookup_device
from ...ir import get_backend_context
from ...model import AIEModel
from ...system_plan import normalize_pl_config

log = logging.getLogger(__name__)


def _make_hls4ml_pass(aie_pass_instance):
    """Wrap a plain AIEPass instance as a ModelOptimizerPass for hls4ml registration."""

    class _Wrapper(ModelOptimizerPass):
        def __init__(self):
            self._aie = aie_pass_instance
            self.name = aie_pass_instance.name

        def transform(self, model):
            return self._aie.transform(model)

    _Wrapper.__name__ = type(aie_pass_instance).__name__
    return _Wrapper


class AIEBackend(Backend):
    def __init__(self):
        super().__init__('AIE')
        self.writer = get_writer(self.name)
        self.attribute_map = {}
        self._register_aie_layer_attributes()
        self._register_flows()

    def _init_file_optimizers(self):
        from aie4ml.pipeline import DEFAULT_PIPELINE

        result = {}
        for cls in DEFAULT_PIPELINE:
            wrapper_cls = _make_hls4ml_pass(cls())
            wrapper_inst = wrapper_cls()
            result[wrapper_inst.name] = wrapper_inst

        # Scan this directory for hls4ml-native passes (lower.py)
        result.update(extract_optimizers_from_path(os.path.dirname(__file__), 'aie4ml.frontends.hls4ml', self))

        return result

    def _register_aie_layer_attributes(self):
        dense_attrs = self.attribute_map.get(Dense, [])
        custom_dense_attrs = [
            ConfigurableAttribute('cas_num', default=-1),
            ConfigurableAttribute('cas_length', default=-1),
            ConfigurableAttribute('microtiling', value_type=dict, default={}),
            Attribute('placement', value_type=dict, default={}, configurable=True),
        ]
        for attr in custom_dense_attrs:
            if attr not in dense_attrs:
                dense_attrs.append(attr)
        self.attribute_map[Dense] = dense_attrs

        activation_attrs = self.attribute_map.get(Activation, [])
        custom_act_attrs = [
            Attribute('placement', value_type=dict, default={}, configurable=True),
        ]
        for attr in custom_act_attrs:
            if attr not in activation_attrs:
                activation_attrs.append(attr)
        self.attribute_map[Activation] = activation_attrs

    def _register_flows(self):
        from aie4ml.pipeline import HLS4ML_FLOW_SPEC

        initializers = self._get_layer_initializers()
        flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)
        flow = register_flow('lower', ['aie:lower_to_aie_ir'], requires=[flow], backend=self.name)

        for flow_name, pass_cls in HLS4ML_FLOW_SPEC:
            pass_name = pass_cls().name
            flow = register_flow(flow_name, [f'aie:{pass_name}'], requires=[flow], backend=self.name)

        flow = register_flow('apply_templates', self._get_layer_templates, requires=[flow], backend=self.name)
        self._default_flow = register_flow('project', None, requires=[flow], backend=self.name)
        self._writer_flow = register_flow(
            'write', ['make_stamp', 'aie:write_aie'], requires=[self._default_flow], backend=self.name
        )

    def create_layer_class(self, layer_class):
        new_attributes = []
        for cls, attributes in self.attribute_map.items():
            if issubclass(layer_class, cls):
                new_attributes.extend(attributes)

        layer_cls_fqn = layer_class.__module__ + '.' + layer_class.__qualname__

        return type(
            self.name + layer_class.__name__,
            (layer_class,),
            {'_expected_attributes': new_attributes, '_wrapped': layer_cls_fqn},
        )

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def _aie_model(self, model):
        return AIEModel.from_context(get_backend_context(model), source_model=model)

    @layer_optimizer(Dense)
    def init_dense_defaults(self, layer):
        if layer.get_attr('microtiling', None) is None:
            layer.set_attr('microtiling', {})
        if layer.get_attr('placement', None) is None:
            layer.set_attr('placement', {})

    @layer_optimizer(Activation)
    def init_activation_defaults(self, layer):
        pass

    def compile(self, model):
        self._aie_model(model).compile()
        return None

    def predict(
        self,
        model,
        X,
        simulator='x86',
        *,
        quantize_in=True,
        dequantize_out=True,
    ):
        return self._aie_model(model).predict(
            X,
            simulator=simulator,
            quantize_in=quantize_in,
            dequantize_out=dequantize_out,
        )

    def write(self, model):
        model.apply_flow(self.get_writer_flow())

    @classmethod
    def convert_precision_string(cls, precision):
        return _FPGABackendHelper.convert_precision_string(precision)

    def _get_device_info(self, part):
        if part is None:
            raise ValueError(f'No AIE part specified. {PART_HELP.format(boards=known_boards())}')
        entry = lookup_device(part)
        if not entry:
            raise ValueError(f'Unknown part "{part}". {PART_HELP.format(boards=known_boards())}')
        return entry

    def create_initial_config(
        self,
        part='xilinx_vek280_base_202610_1',
        plio_width_bits=None,
        pl_clock_freq_mhz=None,
        batch_size=8,
        iterations=8,
        column_start=None,
        row_start=None,
        namespace=None,
        write_tar=False,
        compute_dtype=None,
        target='aie',
        pl_memory='uram',
        enable_pl_timing=False,
        pl_data_mover_mode='benchmark',
        **_,
    ):
        device_info = copy.deepcopy(self._get_device_info(part))

        def _require(key):
            if key not in device_info:
                raise KeyError(f'Device catalog entry "{part}" missing required key "{key}".')
            return device_info[key]

        plio_width = plio_width_bits if plio_width_bits is not None else _require('PLIOWidthBits')
        pl_freq = pl_clock_freq_mhz if pl_clock_freq_mhz is not None else _require('PLClockFreqMHz')
        col_start_val = column_start if column_start is not None else _require('ColumnStart')
        row_start_val = row_start if row_start is not None else _require('RowStart')
        if 'MaxMemTileInPorts' not in device_info or 'MaxMemTileOutPorts' not in device_info:
            raise KeyError(f'Device catalog entry "{part}" missing MaxMemTile port limits.')

        config = {
            'Part': part,
            'AIEConfig': {
                'Device': _require('DeviceName'),
                'Generation': _require('Generation'),
                'Columns': _require('Columns'),
                'Rows': _require('Rows'),
                'ColumnStart': col_start_val,
                'RowStart': row_start_val,
                'PLIOWidthBits': plio_width,
                'PLClockFreqMHz': pl_freq,
                'BatchSize': batch_size,
                'Iterations': iterations,
                'Target': target,
                'PLMemory': pl_memory,
                'EnablePLTiming': bool(enable_pl_timing),
                'PLDataMoverMode': pl_data_mover_mode,
                'Memory': device_info.get('Memory'),
                'MaxMemTileInPorts': int(device_info['MaxMemTileInPorts']),
                'MaxMemTileOutPorts': int(device_info['MaxMemTileOutPorts']),
                **({'ComputeDtype': compute_dtype} if compute_dtype else {}),
            },
            'HLSConfig': {},
            'WriterConfig': {
                'Namespace': namespace,
                'WriteTar': write_tar,
            },
        }

        normalize_pl_config(config['AIEConfig'])
        return config

    def build(self, model, make_target='all', env=None, log_to_stdout=True):
        return self._aie_model(model).build(make_target=make_target, env=env, log_to_stdout=log_to_stdout)

    @model_optimizer()
    def write_aie(self, model):
        self.writer.write_aie(model)
        return True
