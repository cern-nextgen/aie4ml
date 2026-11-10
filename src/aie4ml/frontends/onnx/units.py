# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Axes of extent 1 dropped from activations, so (T, 1, D) and (1, T, D) both read as (T, D). A tensor an op needs at
full rank (a Conv's) keeps its shape, and an op this cannot rewrite gets Reshapes around it."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import GraphEdit, attr, attributes, copied

ELEMENTWISE = frozenset({'QuantizeLinear', 'DequantizeLinear', 'Relu', 'Add', 'Sub', 'Mul', 'Div'})
RESHAPES = frozenset({'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'})
AXIS_ATTRIBUTE = {'Softmax': -1, 'LayerNormalization': -1, 'Concat': None, 'Split': 0}
REWRITTEN = ELEMENTWISE | RESHAPES | set(AXIS_ATTRIBUTE) | {'Transpose', 'Slice', 'MatMul', 'Gemm'}


def squeezed(shape) -> Tuple[int, ...]:
    """`shape` without its axes of extent 1 but the last, at least rank 2."""
    return tuple(int(shape[a]) if a is not None else 1 for a in _kept(shape))


def _kept(shape) -> List[Optional[int]]:
    """The axes of `shape` that `squeezed` keeps, in order; None for a leading axis it adds."""
    if len(shape) < 2:
        return list(range(len(shape)))
    kept = [a for a in range(len(shape) - 1) if shape[a] != 1] + [len(shape) - 1]
    return [None] * (2 - len(kept)) + kept


def drop_unit_axes(
    model_proto, initializers: Dict[str, np.ndarray], shapes: Callable[[object], Dict[str, Tuple[int, ...]]]
):
    """The model with its activations squeezed, graph inputs and outputs too; `shapes(model)` infers a model's
    shapes."""
    shape_of = shapes(model_proto)
    model = copied(model_proto)
    graph = model.graph
    io = {value.name: value for value in (*graph.input, *graph.output)}
    rewrite = _Rewrite(graph, initializers, shape_of)
    rewrite.apply(model, initializers, rewrite.run(graph, set(io)))
    del graph.value_info[:]
    for name, value in io.items():
        if rewrite.changed(name):
            dims = value.type.tensor_type.shape.dim
            del dims[:]
            for d in rewrite.new[name]:
                dims.add().dim_value = int(d)
    return model


def _fixed(graph, initializers, shape_of) -> set:
    """The activations that keep their shapes: an unrewritten or unshaped op's, and those an elementwise op or a
    Transpose shares with one of them."""

    def values(node):
        return [name for name in (*node.input, *node.output) if name and name not in initializers]

    fixed = {
        name
        for node in graph.node
        if node.op_type not in REWRITTEN or any(name not in shape_of for name in values(node))
        for name in values(node)
    }
    grown = True
    while grown:
        grown = False
        for node in graph.node:
            names = values(node)
            if node.op_type not in ELEMENTWISE and node.op_type != 'Transpose':
                continue
            if any(n in fixed for n in names) and not all(n in fixed for n in names):
                fixed.update(names)
                grown = True
    return fixed


def _matmul_shape(lhs, rhs) -> Optional[Tuple[int, ...]]:
    if len(lhs) < 2 or len(rhs) < 2 or lhs[-1] != rhs[-2]:
        return None
    try:
        batch = np.broadcast_shapes(tuple(lhs[:-2]), tuple(rhs[:-2]))
    except ValueError:
        return None
    return (*batch, lhs[-2], rhs[-1])


class _Rewrite(GraphEdit):
    def __init__(self, graph, initializers, shape_of):
        super().__init__()
        fixed = _fixed(graph, initializers, shape_of)
        self.old = {name: tuple(int(d) for d in shape) for name, shape in shape_of.items() if name not in initializers}
        self.new = {name: shape if name in fixed else squeezed(shape) for name, shape in self.old.items()}
        self.initializers = initializers
        self.aliases: Dict[str, str] = {}

    def changed(self, name: str) -> bool:
        return name in self.old and self.new[name] != self.old[name]

    def axes(self, name: str) -> List[Optional[int]]:
        """Each old axis of `name`'s new axis; None where it is dropped."""
        if not self.changed(name):
            return list(range(len(self.old[name])))
        index = {a: i for i, a in enumerate(_kept(self.old[name])) if a is not None}
        return [index.get(a) for a in range(len(self.old[name]))]

    def resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def reshape(self, source: str, shape, out: str, name: str) -> None:
        self.emit('Reshape', [source, self.constant(f'{name}_shape', np.asarray(shape, np.int64))], [out], name)

    def run(self, graph, io: set) -> List:
        for node in graph.node:
            names = [name for name in (*node.input, *node.output) if name]
            if not any(self.changed(name) for name in names):
                self.copy(node, node.input)
            elif not self.rewrite(node):
                self.wrap(node)
        return self.alias_outputs(io)

    def copy(self, node, inputs, outputs=None, **attrs) -> None:
        """`node` reading `inputs` (aliases resolved), its attributes updated by `attrs`."""
        kept = {**attributes(node), **attrs}
        inputs = [self.resolve(name) if name else name for name in inputs]
        self.emit(node.op_type, inputs, outputs or node.output, node.name, **kept)

    def rewrite(self, node) -> bool:
        op = node.op_type
        if op in ELEMENTWISE:
            return self.elementwise(node)
        if op == 'Transpose':
            return self.transpose(node)
        if op in RESHAPES:
            return self.view(node, node.input[0], node.output[0])
        if op in AXIS_ATTRIBUTE:
            return self.along_axis(node)
        if op == 'Slice':
            return self.slice(node)
        if op == 'MatMul':  # with no batch on the right, the left's rows may be any of its axes
            lhs, rhs, out = node.input[0], node.input[1], node.output[0]
            rows_kept = len(self.old_shape(rhs)) == 2 or self.axes(lhs)[-2] is not None
            if rows_kept and _matmul_shape(self.new_shape(lhs), self.new_shape(rhs)) == self.new[out]:
                self.copy(node, node.input)
                return True
        return False

    def old_shape(self, name: str) -> Tuple[int, ...]:
        return self.old[name] if name in self.old else tuple(np.shape(self.initializers[name]))

    def new_shape(self, name: str) -> Tuple[int, ...]:
        return self.new[name] if name in self.new else tuple(np.shape(self.initializers[name]))

    def elementwise(self, node) -> bool:
        """Each input loses the axes its output drops (right-aligned): a constant is reshaped, an activation must be."""
        out = node.output[0]
        dropped = [a - len(self.old[out]) for a, n in enumerate(self.axes(out)) if n is None]
        inputs = []
        for name in node.input:
            if not name:
                inputs.append(name)
                continue
            old = self.old_shape(name)
            drop = tuple(a + len(old) for a in dropped if a + len(old) >= 0)
            if name in self.old:
                if self.new[name] != tuple(d for a, d in enumerate(old) if a not in drop):
                    return False
            elif drop:
                name = self.constant(f'{node.name}_{name}', np.squeeze(self.initializers[name], axis=drop))
            inputs.append(name)
        attrs = {}
        if node.op_type in ('QuantizeLinear', 'DequantizeLinear') and any(a.name == 'axis' for a in node.attribute):
            axis = self.axes(out)[int(attr(node, 'axis', 1)) % len(self.old[out])]
            if axis is None:
                return False
            attrs['axis'] = axis
        self.copy(node, inputs, **attrs)
        return True

    def transpose(self, node) -> bool:
        src, out = node.input[0], node.output[0]
        perm = [int(p) for p in attr(node, 'perm', list(reversed(range(len(self.old[src])))))]
        source, target = self.axes(src), self.axes(out)
        new_perm = [None] * len(self.new[out])
        for j, n in enumerate(target):
            if n is not None:
                new_perm[n] = source[perm[j]]
        if None not in new_perm and sorted(new_perm) == list(range(len(self.new[src]))):
            if new_perm != sorted(new_perm):
                self.copy(node, node.input, perm=new_perm)
                return True
            return self.view(node, src, out)
        moved = [p for p in perm if self.old[src][p] != 1]
        return moved == sorted(moved) and self.view(node, src, out)  # moves only unit axes: a reshape

    def view(self, node, src: str, out: str) -> bool:
        """A reshape of `src` into `out`: none where their new shapes agree."""
        if self.new[src] == self.new[out]:
            self.aliases[out] = self.resolve(src)
        else:
            self.reshape(self.resolve(src), self.new[out], out, node.name)
        return True

    def along_axis(self, node) -> bool:
        rank = len(self.old[node.input[0]])
        default = AXIS_ATTRIBUTE[node.op_type]
        axis = int(attr(node, 'axis', default)) % rank
        tensors = {
            'Concat': [*node.input, node.output[0]],
            'Split': [node.input[0], *node.output],
        }.get(node.op_type, [node.input[0], node.output[0]])
        if node.op_type == 'LayerNormalization':  # normalizes `axis` and those after it: from the first kept
            mapped = {next(n for n in self.axes(name)[axis:] if n is not None) for name in tensors}
        else:
            mapped = {self.axes(name)[axis] for name in tensors}
        if len(mapped) != 1 or None in mapped:
            return False
        inputs = list(node.input)
        if node.op_type == 'LayerNormalization':
            normalized = len(self.new[node.input[0]]) - next(iter(mapped))
            for i in (1, 2):
                if i < len(inputs) and inputs[i] in self.initializers:
                    value = np.asarray(self.initializers[inputs[i]])
                    if value.ndim > normalized:
                        inputs[i] = self.constant(f'{node.name}_{inputs[i]}', value.reshape(value.shape[-normalized:]))
        self.copy(node, inputs, axis=next(iter(mapped)))
        return True

    def slice(self, node) -> bool:
        src = node.input[0]
        rank = len(self.old[src])
        axes = (
            np.asarray(self.initializers[node.input[3]]).reshape(-1)
            if len(node.input) > 3 and node.input[3]
            else np.arange(np.asarray(self.initializers[node.input[1]]).size)
        )
        mapped = [self.axes(src)[int(a) % rank] for a in axes]
        if None in mapped or [self.axes(node.output[0])[int(a) % rank] for a in axes] != mapped:
            return False
        inputs = list(node.input) + [''] * (4 - len(node.input))
        inputs[3] = self.constant(f'{node.name}_axes', np.asarray(mapped, np.int64))
        self.copy(node, inputs)
        return True

    def wrap(self, node) -> None:
        """`node` on its own shapes, reshaped from its new inputs and to its new outputs."""
        inputs = []
        for i, name in enumerate(node.input):
            if name and self.changed(name):
                full = f'{node.name}_in{i}'
                self.reshape(self.resolve(name), self.old[name], full, f'{full}_reshape')
                name = full
            inputs.append(name)
        outputs = [f'{name}_full' if name and self.changed(name) else name for name in node.output]
        self.copy(node, inputs, outputs)
        for name, full in zip(node.output, outputs):
            if full != name:
                self.reshape(full, self.new[name], name, f'{name}_reshape')

    def alias_outputs(self, io: set) -> List:
        """The nodes, with each graph output an alias names written under its own name again."""
        renamed = {}
        for out in self.aliases:
            if out in io:
                source = self.resolve(out)
                if source in io:
                    raise NotImplementedError(f'{out}: a graph output that is {source!r} with its unit axes moved.')
                renamed[source] = out
        for node in self.nodes:
            node.input[:] = [renamed.get(self.resolve(n), self.resolve(n)) if n else n for n in node.input]
            node.output[:] = [renamed.get(n, n) for n in node.output]
        return self.nodes
