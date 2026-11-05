# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Attention of every head at once, as exporters emit it (a Reshape splitting the features into heads, each op on all
heads, a Reshape merging them), rewritten head by head: each head a Slice of the features through its own copy of each
op, the heads merged by a Concat. A region with an op this does not rewrite is kept."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import GraphEdit, attr, attributes, copied

COPIED = frozenset({'QuantizeLinear', 'DequantizeLinear', 'Div', 'Mul', 'Add', 'Sub'})
"""Copied per head where every operand but the batched one is a scalar constant, or for an Add the heads' bias."""


@dataclass(frozen=True)
class Batched:
    """A tensor of every head: its head axis, and its heads' tensors (its shape without that axis)."""

    axis: int
    heads: Tuple[str, ...]


def unbatch_heads(
    model_proto, initializers: Dict[str, np.ndarray], shapes: Callable[[object], Dict[str, Tuple[int, ...]]]
):
    """The model with each region of batched heads rewritten head by head; `shapes(model)` infers a model's shapes."""
    model = copied(model_proto)
    while True:
        region = _first_region(model.graph, initializers, shapes(model))
        if region is None:
            return model
        kept = [node for node in model.graph.node if node.name not in region.removed]
        region.apply(model, initializers, kept + region.nodes)


def _split(node, initializers, shape_of) -> Optional[Tuple[int, int]]:
    """(heads, head size) where `node` is a Reshape splitting its input's last axis into heads; else None."""
    if node.op_type != 'Reshape' or node.input[0] not in shape_of or node.output[0] not in shape_of:
        return None
    src, out = shape_of[node.input[0]], shape_of[node.output[0]]
    if len(out) < 3 or out[-2] < 2 or out[-2] * out[-1] != src[-1] or tuple(src[:-1]) != tuple(out[:-2]):
        return None
    return int(out[-2]), int(out[-1])


def _first_region(graph, initializers, shape_of) -> Optional['_Region']:
    """The region of the first head split and every split like it (Q's, K's and V's), rewritten; None if it is not."""
    splits = [(node, _split(node, initializers, shape_of)) for node in graph.node]
    splits = [(node, heads) for node, heads in splits if heads is not None]
    if not splits:
        return None
    heads = splits[0][1]
    region = _Region({node.name for node, other in splits if other == heads}, heads, graph, initializers, shape_of)
    return region if region.rewrite() else None


class _Region(GraphEdit):
    def __init__(self, entries, heads, graph, initializers, shape_of):
        super().__init__()
        self.entries, (self.count, self.size) = entries, heads
        self.graph, self.initializers, self.shape_of = graph, initializers, shape_of
        self.batched: Dict[str, Batched] = {}
        self.removed = set()

    def rewrite(self) -> bool:
        """Every node in graph order, so a node's batched operands are rewritten before it."""
        exits = 0
        for node in self.graph.node:
            if node.name in self.entries:
                self.split(node)
            elif any(name in self.batched for name in node.input):
                result = self.node(node)
                if result is None:
                    return False
                exits += result == 'exit'
            else:
                continue
            self.removed.add(node.name)
        outputs = {value.name for value in self.graph.output}
        return bool(exits) and not any(name in outputs for name in self.batched)

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
        rank = len(self.shape_of[node.input[ins.index(b)]])
        per_head = {}  # by operand: each head's part of a bias
        for i, (name, other) in enumerate(zip(node.input, ins)):
            scalar = name in self.initializers and np.asarray(self.initializers[name]).size == 1
            if other is None and name and not scalar:
                parts = self.bias(name, b, rank) if node.op_type == 'Add' else None
                if parts is None:
                    return None
                per_head[i] = parts
        heads = []
        for h, src in enumerate(b.heads):
            out = f'{node.name}_h{h}_out'
            inputs = [
                src if other is not None else per_head[i][h] if i in per_head else name
                for i, (name, other) in enumerate(zip(node.input, ins))
            ]
            self.emit(node.op_type, inputs, [out], f'{node.name}_h{h}', **attributes(node))
            heads.append(out)
        self.batched[node.output[0]] = Batched(b.axis, tuple(heads))
        return [node.output[0]]

    def bias(self, name: str, b: Batched, rank: int) -> Optional[List[str]]:
        """Each head's part of a tensor added to `b`: itself without a head axis (a 2D mask), the head's rows of a
        (heads, queries, keys) one; else None."""
        shape = self.shape_of.get(name)
        if shape is None:
            return None
        if b.axis < rank - len(shape):
            return [name] * self.count
        if b.axis - (rank - len(shape)) != 0 or len(shape) != 3 or shape[0] != self.count:
            return None
        rows = f'{name}_rows'  # the heads' rows one after another
        parts = [f'{name}_h{h}' for h in range(self.count)]
        if rows in self.shape_of:  # an earlier attention's, which every layer adding this bias shares
            return parts
        self.emit('Reshape', [name, self.constant(f'{rows}_shape', [shape[0] * shape[1], shape[2]])], [rows], rows)
        self.shape_of[rows] = (shape[0] * shape[1], shape[2])
        for h, base in enumerate(parts):
            bounds = [(n, v) for n, v in (('starts', h * shape[1]), ('ends', (h + 1) * shape[1]), ('axes', 0))]
            self.emit('Slice', [rows, *(self.constant(f'{base}_{n}', [v]) for n, v in bounds)], [base], base)
            self.shape_of[base] = tuple(shape[1:])
        return parts

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
