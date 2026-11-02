# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Attention computed for every head at once, as exporters emit it, rewritten head by head.

An exporter splits the features into heads with a Reshape (..., D) -> (..., H, D / H), moves the head axis about with
Transposes, multiplies, scales, quantizes and normalizes all heads in one op each, and merges them back with a Reshape
(..., H, D / H) -> (..., D). Rewritten, each head is a Slice of the features through its own copy of every op, and the
heads a Concat: what the per-head lowerings (a head's attention, a Dense) read. A region with an op this does not
rewrite is left as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import attr, require_onnx, topological

COPIED = frozenset({'QuantizeLinear', 'DequantizeLinear', 'Div', 'Mul', 'Add', 'Sub'})
"""Copied per head where every operand but the batched one is a scalar constant."""


@dataclass(frozen=True)
class Batched:
    """A tensor of every head: its head axis, and its heads' tensors (its shape without that axis)."""

    axis: int
    heads: Tuple[str, ...]


def unbatch_heads(model_proto, initializers: Dict[str, np.ndarray], shape_of: Dict[str, Tuple[int, ...]]):
    """The model with each region of batched heads rewritten head by head (module docstring)."""
    onnx, helper, numpy_helper = require_onnx()
    model = onnx.ModelProto()
    model.CopyFrom(model_proto)
    while True:
        rewrite = _first_region(model.graph, initializers, shape_of, helper, numpy_helper)
        if rewrite is None:
            return model
        removed, added, constants = rewrite
        nodes = [node for node in model.graph.node if node.name not in removed] + added
        del model.graph.node[:]
        model.graph.node.extend(topological(nodes))
        for name, value in constants.items():
            initializers[name] = value
            model.graph.initializer.append(numpy_helper.from_array(value, name))
        _infer(model, shape_of)


def _infer(model, shape_of) -> None:
    onnx, _, _ = require_onnx()
    inferred = onnx.shape_inference.infer_shapes(model)
    for value_info in list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = value_info.type.tensor_type.shape.dim
        if dims and all(dim.HasField('dim_value') for dim in dims):
            shape_of[value_info.name] = tuple(int(dim.dim_value) for dim in dims)


def _split(node, initializers, shape_of) -> Optional[Tuple[int, int]]:
    """(heads, head size) where `node` is a Reshape splitting its input's last axis into heads; else None."""
    if node.op_type != 'Reshape' or node.input[0] not in shape_of or node.output[0] not in shape_of:
        return None
    src, out = shape_of[node.input[0]], shape_of[node.output[0]]
    if len(out) < 3 or out[-2] < 2 or out[-2] * out[-1] != src[-1] or tuple(src[:-1]) != tuple(out[:-2]):
        return None
    return int(out[-2]), int(out[-1])


def _first_region(graph, initializers, shape_of, helper, numpy_helper):
    """The region of the first head split, with every other split of as many heads of the same size (Q's, K's and V's
    meet in the products), rewritten: (names of the nodes it replaces, the nodes replacing them, new constants); None
    where no region rewrites."""
    splits = [(node, _split(node, initializers, shape_of)) for node in graph.node]
    splits = [(node, heads) for node, heads in splits if heads is not None]
    if not splits:
        return None
    heads = splits[0][1]
    entries = {node.name for node, other in splits if other == heads}
    return _Region(entries, heads, graph, initializers, shape_of, helper).rewrite()


class _Region:
    def __init__(self, entries, heads, graph, initializers, shape_of, helper):
        self.entries, (self.count, self.size) = entries, heads
        self.graph, self.initializers, self.shape_of, self.helper = graph, initializers, shape_of, helper
        self.nodes: List = []
        self.constants: Dict[str, np.ndarray] = {}
        self.batched: Dict[str, Batched] = {}
        self.removed = set()

    def constant(self, name: str, value) -> str:
        self.constants[name] = np.asarray(value)
        return name

    def emit(self, op_type: str, inputs, outputs, name: str, **attrs) -> None:
        self.nodes.append(self.helper.make_node(op_type, list(inputs), list(outputs), name=name, **attrs))

    def rewrite(self):
        """Every node in graph order, so a node's batched operands are rewritten before it."""
        exits = 0
        for node in self.graph.node:
            if node.name in self.entries:
                self.split(node)
            elif any(name in self.batched for name in node.input):
                result = self.node(node)
                if result is None:
                    return None
                exits += result == 'exit'
            else:
                continue
            self.removed.add(node.name)
        outputs = {value.name for value in self.graph.output}
        if not exits or any(name in outputs for name in self.batched):
            return None
        return self.removed, self.nodes, self.constants

    def split(self, entry) -> None:
        """A head split: each head a Slice of the features."""
        out_shape = self.shape_of[entry.output[0]]
        names = []
        for h in range(self.count):
            base = f'{entry.name}_h{h}'
            sliced = f'{base}_slice'
            self.emit(
                'Slice',
                [
                    entry.input[0],
                    self.constant(f'{base}_starts', [h * self.size]),
                    self.constant(f'{base}_ends', [(h + 1) * self.size]),
                    self.constant(f'{base}_axes', [-1]),
                ],
                [sliced],
                base,
            )
            self.shape_of[sliced] = tuple(self.shape_of[entry.input[0]][:-1]) + (self.size,)
            names.append(sliced)
        self.batched[entry.output[0]] = Batched(len(out_shape) - 2, tuple(names))

    def node(self, node):
        """Rewrite one node reading batched tensors: 'exit' for a merge, None where it does not rewrite."""
        ins = [self.batched.get(name) for name in node.input]
        op = node.op_type
        if op == 'Reshape':
            return self.merge(node, ins[0])
        if op == 'Transpose':
            return self.transpose(node, ins[0])
        if op in COPIED:
            return self.copy(node, ins)
        if op == 'MatMul':
            return self.matmul(node, ins)
        if op == 'Softmax':
            return self.softmax(node, ins[0])
        return None

    def merge(self, node, b: Optional[Batched]):
        """A Reshape (..., H, D / H) -> (..., D): the heads concatenated along their last axis."""
        out_shape = self.shape_of.get(node.output[0])
        in_shape = self.shape_of.get(node.input[0])
        if b is None or out_shape is None or b.axis != len(in_shape) - 2 or tuple(in_shape[:-2]) != out_shape[:-1]:
            return None
        if out_shape[-1] != self.count * self.size:
            return None
        self.emit('Concat', b.heads, [node.output[0]], f'{node.name}_concat', axis=-1)
        return 'exit'

    def transpose(self, node, b: Optional[Batched]):
        if b is None:
            return None
        perm = [int(p) for p in attr(node, 'perm')]
        axis = perm.index(b.axis)
        rest = [p - (p > b.axis) for p in perm if p != b.axis]
        heads = []
        for h, src in enumerate(b.heads):
            if rest == sorted(rest):
                heads.append(src)
                continue
            out = f'{node.name}_h{h}_out'
            self.emit('Transpose', [src], [out], f'{node.name}_h{h}', perm=rest)
            heads.append(out)
        self.batched[node.output[0]] = Batched(axis, tuple(heads))
        return [node.output[0]]

    def copy(self, node, ins: List[Optional[Batched]]):
        batched = [b for b in ins if b is not None]
        if len(batched) != 1:
            return None
        b = batched[0]
        for name, other in zip(node.input, ins):
            scalar = name in self.initializers and np.asarray(self.initializers[name]).size == 1
            if other is None and name and not scalar:
                return None
        heads = []
        for h, src in enumerate(b.heads):
            out = f'{node.name}_h{h}_out'
            inputs = [src if other is not None else name for name, other in zip(node.input, ins)]
            self.emit(node.op_type, inputs, [out], f'{node.name}_h{h}', **_attributes(node))
            heads.append(out)
        self.batched[node.output[0]] = Batched(b.axis, tuple(heads))
        return [node.output[0]]

    def matmul(self, node, ins: List[Optional[Batched]]):
        lhs, rhs = ins
        if lhs is None or rhs is None or lhs.axis != rhs.axis:
            return None
        rank = len(self.shape_of[node.input[0]])
        if lhs.axis >= rank - 2 or len(self.shape_of[node.input[1]]) != rank:
            return None
        heads = []
        for h, (a, c) in enumerate(zip(lhs.heads, rhs.heads)):
            out = f'{node.name}_h{h}_out'
            self.emit('MatMul', [a, c], [out], f'{node.name}_h{h}')
            heads.append(out)
        self.batched[node.output[0]] = Batched(lhs.axis, tuple(heads))
        return [node.output[0]]

    def softmax(self, node, b: Optional[Batched]):
        rank = len(self.shape_of[node.input[0]])
        axis = int(attr(node, 'axis', -1)) % rank
        if b is None or axis == b.axis:
            return None
        heads = []
        for h, src in enumerate(b.heads):
            out = f'{node.name}_h{h}_out'
            self.emit('Softmax', [src], [out], f'{node.name}_h{h}', axis=axis - (axis > b.axis))
            heads.append(out)
        self.batched[node.output[0]] = Batched(b.axis, tuple(heads))
        return [node.output[0]]


def _attributes(node) -> dict:
    _, helper, _ = require_onnx()
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}
