# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Serialization helpers for exporting the backend IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .aie_types import QuantIntent
from .ir.graph import ExecutionInstance, OpNode
from .op_impls.common_types import to_plain


def dump_pipeline_ir(ctx, destination: Path) -> None:
    """Serialize logical, execution, and physical IR into a single JSON file."""

    data = {
        'device': str(ctx.device.part),
        'logical': [serialize_logical_node(node) for node in ctx.ir.logical],
        'execution': [serialize_op_impl_instance(inst) for inst in ctx.ir.execution],
        'execution_values': [
            {'name': v.name, 'producer': v.producer, 'view': v.view.node if v.view else None}
            for v in ctx.ir.execution.values.values()
        ],
        'physical': serialize_physical_ir(ctx.ir.physical),
        'optimizer': ctx.ir.optimizer,
    }

    destination.write_text(json.dumps(data, indent=2))


def serialize_logical_node(node: OpNode) -> Dict[str, Any]:
    metadata = _serialize_metadata(node.metadata)

    return {
        'name': node.name,
        'op_type': node.op_type,
        'inputs': [t.name for t in node.inputs],
        'outputs': [t.name for t in node.outputs],
        'traits': {name: trait.data for name, trait in node.traits.items()},
        'metadata': metadata,
    }


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(metadata)
    quant_meta = metadata.get('quant')
    if isinstance(quant_meta, dict):
        metadata['quant'] = _serialize_quant_metadata(quant_meta)

    return metadata


def _serialize_quant_metadata(quant_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in quant_meta.items():
        if key.endswith('_precision'):
            out[key] = serialize_precision(value)
        else:
            out[key] = value
    return out


def serialize_op_impl_instance(inst: ExecutionInstance) -> Dict[str, Any]:
    return {
        'node': inst.name,
        'op_type': inst.op_type,
        'variant_id': inst.variant.variant_id,
        'inputs': [to_plain(item) for item in inst.inputs],
        'outputs': list(inst.outputs),
        'ports': to_plain(inst.ports),
        'io_route': to_plain(inst.io_route),
        'port_views': to_plain(inst.port_views),
        'graph_header': inst.graph_header,
        'graph_name': inst.graph_name,
        'param_template': inst.param_template,
        'config': to_plain(inst.config),
        # "input_staging": inst.variant.describe_input_staging(inst),
        # "output_staging": inst.variant.describe_output_staging(inst),
        # TODO: serialize implementation artifacts (weights, LUTs, etc.)
    }


def serialize_physical_ir(physical_ir) -> Dict[str, Any]:
    return physical_ir.to_dict()


def serialize_precision(precision):
    if precision is None:
        return None
    from .aie_types import FloatIntent

    if isinstance(precision, FloatIntent):
        return {'width': int(precision.width), 'format': precision.format.value}
    if not isinstance(precision, QuantIntent):
        raise TypeError(f'Unsupported precision type {type(precision)} when serializing backend IR.')
    return {
        'width': int(precision.width),
        'frac': int(precision.frac),
        'signed': bool(precision.signed),
        'rounding': precision.rounding.name,
        'saturation': precision.saturation.name,
    }


__all__ = ['dump_pipeline_ir']
