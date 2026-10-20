# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Serialization helpers for exporting the backend IR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .ir.graph import ExecutionInstance, OpNode
from .op_impls.common_types import to_plain
from .op_impls.family_registry import get_family_resolver_registry


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
    family = get_family_resolver_registry().find(node.op_type)
    return {
        'name': node.name,
        'op_type': node.op_type,
        'inputs': [t.name for t in node.inputs],
        'outputs': [t.name for t in node.outputs],
        'traits': {name: trait.data for name, trait in node.traits.items()},
        'metadata': dict(node.metadata),
        'macs': family.macs(node) if family is not None else 0,
    }


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
        'graph_header': inst.variant.graph_header,
        'graph_name': inst.variant.graph_name,
        'param_template': inst.variant.param_template,
        'config': to_plain(inst.config),
    }


def serialize_physical_ir(physical_ir) -> Dict[str, Any]:
    return physical_ir.to_dict()


__all__ = ['dump_pipeline_ir']
