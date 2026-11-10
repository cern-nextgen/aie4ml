# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""ONNX -> aie4ml logical IR importer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from ...aie_types import QuantIntent
from ...ir import LogicalIR, TensorVar, set_input_roles
from ...ir.context import AIEBackendContext, ProjectConfig
from ...model import AIEModel
from ..common import create_context
from . import handlers  # noqa: F401 - registers all handlers
from .constants import fold_constants
from .context import OnnxImportContext
from .heads import unbatch_heads
from .projections import split_packed_projections
from .registry import dispatch
from .shapes import build_shape_table
from .units import drop_unit_axes
from .utils import (
    copied,
    initializer_map,
    input_maps,
    require_onnx,
    resolve_project_name,
)
from .utils import (
    node_name as onnx_node_name,
)


def lower_onnx_model(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEBackendContext:
    onnx, _helper, numpy_helper = require_onnx()

    if isinstance(model_or_path, (str, Path)):
        model_path = Path(model_or_path)
        model_proto = onnx.load(str(model_path))
    else:
        model_path = None
        model_proto = model_or_path
    graph_proto = model_proto.graph

    resolved_project_name = resolve_project_name(model_path, model_proto, project_name)
    aie_config = config.get('AIEConfig', {}) or {}
    backend = create_context(
        aie_config.get('Part') or config.get('Part'),
        aie_config,
        ProjectConfig(Path(output_dir), resolved_project_name, stamp, dict(custom_sources or {})),
    )
    graph: LogicalIR = backend.ir.logical

    initializers = initializer_map(graph_proto, numpy_helper)

    # Normalized as exporters emit it: nodes named (directives use the names), constants folded, unit axes dropped,
    # packed projections split, batched heads rewritten head by head, then their projections split per head. The host
    # keeps each graph input's and output's shape.
    def shapes(model):
        return build_shape_table(model, initializers, input_maps(model.graph, set(initializers))[0])

    model_proto = fold_constants(_named(model_proto), initializers, shapes)
    declared = shapes(model_proto)
    for rewrite in (drop_unit_axes, split_packed_projections, unbatch_heads, split_packed_projections):
        model_proto = rewrite(model_proto, initializers, shapes)
    graph_proto = model_proto.graph
    shape_of = shapes(model_proto)
    io = [value.name for value in (*graph_proto.input, *graph_proto.output) if value.name in declared]
    graph.host_shapes = {name: declared[name] for name in io if shape_of.get(name) != declared[name]}

    input_shapes, input_dtypes = input_maps(graph_proto, set(initializers))
    for name, shape in input_shapes.items():
        if not shape:
            raise ValueError(f'{name}: scalar inputs are not supported.')
        graph.add_tensor(TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=None))
        graph.mark_graph_input(name)
    ctx = OnnxImportContext(
        backend,
        initializers=initializers,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        shape_of=shape_of,
        layer_directives=dict(config.get('LayerDirectives', {}) or {}),
    )

    for node in graph_proto.node:
        dispatch(ctx, node, node.name, ctx.take_directives(node.name))

    _finalize(ctx, graph_proto)
    return backend


def _named(model_proto):
    """The model with every unnamed node given the name it is directed by (`node_name`)."""
    model = copied(model_proto)
    for index, node in enumerate(model.graph.node):
        node.name = onnx_node_name(node, index)
    return model


def _exact_result(node) -> None:
    """A product or sum of two activations no quantizer follows (attention's scores and their bias, as exporters leave
    them) is exact in 32 bits: at the operands' fraction bits summed, or the finer operand's."""
    out = node.outputs[0]
    lhs, rhs = node.inputs[:2]
    if out.precision is None and isinstance(lhs.precision, QuantIntent) and isinstance(rhs.precision, QuantIntent):
        fracs = (int(lhs.precision.frac), int(rhs.precision.frac))
        frac = sum(fracs) if node.op_type == 'matmul' else max(fracs)
        out.precision = replace(lhs.precision, width=32, frac=frac, signed=True)


def _finalize(ctx: OnnxImportContext, graph_proto) -> None:
    graph = ctx.graph

    unused = sorted(set(ctx.layer_directives) - ctx.used_directives)
    if unused:
        raise ValueError('Unused LayerDirectives entries: ' + ', '.join(unused))

    for output in graph_proto.output:
        if output.name not in ctx.value_tensors:
            raise ValueError(f'Graph output {output.name}: expected a lowered semantic tensor.')
        ctx.canonical_source(output.name, f'graph output {output.name}')
        tensor = ctx.value_tensors[output.name]
        if output.name not in graph.tensors:
            graph.tensors.pop(tensor.name, None)
            tensor.name = output.name
            graph.tensors[output.name] = tensor
        graph.mark_graph_output(output.name)

    for node in graph.nodes:
        role_names = node.metadata.pop('input_roles', None)
        if role_names:
            set_input_roles(node, node.inputs, role_names)
        if node.op_type in ('matmul', 'add'):
            _exact_result(node)

    for target, source in ctx.precision_mirrors:
        tensor = graph.tensors.get(target)
        origin = graph.tensors.get(source)
        if tensor is not None and tensor.precision is None and origin is not None:
            tensor.precision = origin.precision

    for tensor in graph.tensors.values():
        if tensor.is_parameter:
            continue
        if tensor.precision is None:
            raise ValueError(f'{tensor.name}: missing quantization intent after ONNX QDQ lowering.')


def from_onnx(
    model_or_path,
    config: Dict[str, Any],
    *,
    output_dir,
    project_name: Optional[str] = None,
    stamp: Optional[str] = None,
    custom_sources: Optional[Dict[str, str]] = None,
) -> AIEModel:
    ctx = lower_onnx_model(
        model_or_path,
        config,
        output_dir=output_dir,
        project_name=project_name,
        stamp=stamp,
        custom_sources=custom_sources,
    )
    return AIEModel.from_context(ctx, source_model=model_or_path)
