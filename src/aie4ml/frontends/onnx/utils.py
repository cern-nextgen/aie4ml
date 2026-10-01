from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ml_dtypes
import numpy as np

from ...aie_types import FloatFormat, FloatIntent, QuantIntent, RoundingMode, SaturationMode
from ...device_catalog import resolve_device
from ...ir import BackendPolicies
from ...ir.context import AIEBackendContext, ProjectConfig
from ...system_plan import normalize_pl_config


def require_onnx():
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise ImportError('ONNX frontend requires the "onnx" Python package to be installed.') from exc
    return onnx, helper, numpy_helper


def node_name(node, index: int) -> str:
    name = (node.name or '').strip()
    return name if name else f'{node.op_type}_{index}'


def shape_from_value_info(value_info) -> Tuple[int, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        raise ValueError(f'{value_info.name}: tensor shape is missing.')
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField('dim_value'):
            raise ValueError(f'{value_info.name}: dynamic shapes are not supported.')
        dims.append(int(dim.dim_value))
    return tuple(dims)


def resolve_project_name(model_path, model_proto, project_name: Optional[str]) -> str:
    if project_name:
        return project_name
    if model_path is not None:
        return Path(model_path).stem
    graph_name = str(getattr(model_proto.graph, 'name', '') or '').strip()
    if graph_name:
        return graph_name
    raise ValueError('project_name must be provided when lowering an in-memory ONNX model without graph.name.')


def create_context(config: Dict[str, Any], output_dir, project_name: str, stamp, custom_sources) -> AIEBackendContext:
    aie_cfg = dict(config.get('AIEConfig', {}) or {})
    part_name = aie_cfg.get('Part') or config.get('Part') or aie_cfg.get('Device')
    if not part_name:
        raise KeyError('ONNX frontend requires Part or AIEConfig.Part in the config dict.')

    device, merged = resolve_device(part_name, aie_cfg)
    policies = BackendPolicies(
        fusion=dict(config.get('AIEFusionPolicy', {}) or {}),
        decomposition=dict(config.get('AIEDecompositionPolicy', {}) or {}),
        pack=dict(config.get('AIEPackPolicy', {}) or {}),
        cache=dict(config.get('AIECachePolicy', {}) or {}),
        tensors_have_batch=True,
    )
    project_config = ProjectConfig(
        output_dir=Path(output_dir),
        project_name=project_name,
        stamp=stamp,
        custom_sources=dict(custom_sources or {}),
    )
    resolved_aie_config = dict(merged)
    resolved_aie_config['Part'] = str(part_name)

    normalize_pl_config(resolved_aie_config)

    ctx = AIEBackendContext(
        device=device,
        policies=policies,
        project_config=project_config,
        aie_config=resolved_aie_config,
    )
    return ctx


# ONNX bfloat16 and float8 e4m3fn, decoded here: onnx < 1.18 returns their raw bits, and reads a bfloat16 tensor's
# `int32_data` even when it holds `raw_data`, leaving the array uninitialized.
_NARROW_FLOATS = {16: (np.uint16, ml_dtypes.bfloat16), 17: (np.uint8, ml_dtypes.float8_e4m3fn)}


def initializer_map(graph, numpy_helper) -> Dict[str, np.ndarray]:
    def values(init) -> np.ndarray:
        if init.data_type not in _NARROW_FLOATS:
            return np.asarray(numpy_helper.to_array(init))
        bits, dtype = _NARROW_FLOATS[init.data_type]
        stored = np.frombuffer(init.raw_data, np.dtype(bits).newbyteorder('<')) if init.raw_data else init.int32_data
        return np.asarray(stored, dtype=bits).view(dtype).reshape(tuple(init.dims))

    return {init.name: values(init) for init in graph.initializer}


def input_maps(graph, initializer_names) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, int]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    elem_types: Dict[str, int] = {}
    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        shapes[value_info.name] = shape_from_value_info(value_info)
        elem_types[value_info.name] = value_info.type.tensor_type.elem_type
    return shapes, elem_types


def scalar_tensor(initializers: Dict[str, np.ndarray], name: str, node_name: str) -> np.ndarray:
    if name not in initializers:
        raise ValueError(f'{node_name}: quantization parameter {name} must be an initializer.')
    arr = np.asarray(initializers[name])
    if arr.size != 1:
        raise ValueError(f'{node_name}: per-axis quantization is not supported for {name}.')
    return arr.reshape(())


def intent_from_qparams(
    initializers: Dict[str, np.ndarray],
    scale_name: str,
    zero_name: str,
    qdtype,
    node_name: str,
) -> QuantIntent:
    scale = float(scalar_tensor(initializers, scale_name, node_name))
    zero = scalar_tensor(initializers, zero_name, node_name)
    if int(zero) != 0:
        raise ValueError(f'{node_name}: zero_point must be 0 for symmetric quantization.')
    dtype = np.dtype(qdtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(f'{node_name}: only integer quantization is supported; got {dtype}.')
    if scale <= 0.0:
        raise ValueError(f'{node_name}: quantization scale must be positive.')
    log2_scale = np.log2(scale)
    rounded = round(float(log2_scale))
    if not np.isclose(log2_scale, rounded, atol=1e-7):
        raise ValueError(f'{node_name}: scale {scale} is not a power of two.')
    return QuantIntent(
        width=int(dtype.itemsize * 8),
        frac=int(-rounded),
        signed=bool(np.issubdtype(dtype, np.signedinteger)),
        rounding=RoundingMode.RND_CONV,
        saturation=SaturationMode.SAT,
    )


def dequantize_data(
    data: np.ndarray,
    initializers: Dict[str, np.ndarray],
    scale_name: str,
    zero_name: str,
    node_name: str,
) -> np.ndarray:
    scale = float(scalar_tensor(initializers, scale_name, node_name))
    zero = int(scalar_tensor(initializers, zero_name, node_name))
    return (np.asarray(data, dtype=np.float64) - float(zero)) * scale


def intent_from_initializer(data: np.ndarray, node_name: str):
    dtype = np.asarray(data).dtype
    if dtype == np.dtype(np.float32):
        return FloatIntent(width=32, format=FloatFormat.FP32)
    if dtype == np.dtype(ml_dtypes.bfloat16):
        return FloatIntent(width=16, format=FloatFormat.BF16)
    if dtype == np.dtype(ml_dtypes.float8_e4m3fn):
        return FloatIntent(width=8, format=FloatFormat.FP8_E4M3)
    raise ValueError(
        f'{node_name}: direct initializer inputs must be float32/bfloat16/fp8_e4m3, '
        f'or quantized via DequantizeLinear; got {dtype}.'
    )


def attr(node, name: str, default=None):
    _, helper, _ = require_onnx()
    for value in node.attribute:
        if value.name != name:
            continue
        return helper.get_attribute_value(value)
    return default
