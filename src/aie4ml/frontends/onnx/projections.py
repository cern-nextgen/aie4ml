# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""A packed projection split per slice of its output, so each part is a Dense of its own (Q, K, V; a head's):

    MatMul(x, W) [+ bias] [-> f] -> Slice(..., a:b, last axis)   becomes   MatMul(x, W[:, a:b]) [+ bias[a:b]] [-> f]

with W a dequantized constant and f per-tensor ops (a QDQ, a scalar scale), copied per part."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import GraphEdit, attr, attributes, copied


def split_packed_projections(model_proto, initializers: Dict[str, np.ndarray], shape_of: Dict[str, Tuple[int, ...]]):
    model = copied(model_proto)
    graph = model.graph
    writer = {name: node for node in graph.node for name in node.output}
    readers: Dict[str, List] = {}
    for node in graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node)
    outputs = {value.name for value in graph.output}

    edit, removed = GraphEdit(), set()
    for matmul in graph.node:
        if matmul.op_type != 'MatMul':
            continue
        weight = _constant_weight(matmul.input[1], writer, initializers)
        if weight is None:
            continue
        add, bias = _bias(matmul.output[0], readers, writer, initializers)
        chain = _per_tensor_chain((add or matmul).output[0], readers, initializers)
        result = (chain[-1] if chain else add or matmul).output[0]
        slices = [_last_axis_slice(node, initializers, shape_of) for node in readers.get(result, [])]
        if not slices or any(s is None for s in slices) or result in outputs:
            continue
        for node, (start, end) in slices:
            base = f'{node.name}_part'
            ops = [matmul, *([add] if add else []), *chain]
            values = [f'{base}_{op.name}' for op in ops[:-1]] + [node.output[0]]
            w = _dequantized(weight, start, end, f'{base}_w', initializers, edit)
            edit.emit('MatMul', [matmul.input[0], w], [values[0]], base)
            if add:
                b = _dequantized(bias, start, end, f'{base}_b', initializers, edit)
                inputs = [values[0], b] if add.input[1] == matmul.output[0] else [b, values[0]]
                edit.emit('Add', inputs, [values[1]], f'{base}_bias')
            for at, op in enumerate(chain, start=len(ops) - len(chain)):
                inputs = [values[at - 1], *op.input[1:]]
                edit.emit(op.op_type, inputs, [values[at]], f'{base}_{op.name}', **attributes(op))
            removed.add(node.name)
        removed.update(n.name for n in (matmul, add, *chain) if n is not None)

    if edit.nodes:
        edit.apply(model, initializers, _live([n for n in graph.node if n.name not in removed] + edit.nodes, outputs))
    return model


def _constant_weight(name, writer, initializers) -> Optional[dict]:
    """The constant W a MatMul's rhs dequantizes (through a 2-D Transpose), with its quantizer's parameters."""
    node = writer.get(name)
    transposed = node is not None and node.op_type == 'Transpose' and list(attr(node, 'perm', [1, 0])) == [1, 0]
    if transposed:
        node = writer.get(node.input[0])
    if node is None or node.op_type != 'DequantizeLinear' or node.input[0] not in initializers:
        return None
    if np.asarray(initializers[node.input[0]]).ndim != 2:
        return None
    return {'node': node, 'transposed': transposed, 'columns': 0 if transposed else 1}


def _bias(name, readers, writer, initializers):
    """The Add of a dequantized constant vector to the MatMul's output, and that vector, where it is its only reader."""
    nodes = readers.get(name, [])
    if len(nodes) != 1 or nodes[0].op_type != 'Add':
        return None, None
    add = nodes[0]
    other = add.input[0] if add.input[1] == name else add.input[1]
    node = writer.get(other)
    if node is None or node.op_type != 'DequantizeLinear' or node.input[0] not in initializers:
        return None, None
    if np.asarray(initializers[node.input[0]]).ndim != 1:
        return None, None
    return add, {'node': node, 'transposed': False, 'columns': 0}


PER_TENSOR = frozenset({'QuantizeLinear', 'DequantizeLinear', 'Mul', 'Div'})


def _per_tensor_chain(name, readers, initializers) -> List:
    """The ops after `name`, each its value's only reader, that act on every element alike: a QDQ, a scalar scale."""
    chain = []
    while len(readers.get(name, [])) == 1:
        node = readers[name][0]
        if node.op_type not in PER_TENSOR or node.input[0] != name:
            break
        if any(c and (c not in initializers or np.asarray(initializers[c]).size != 1) for c in node.input[1:]):
            break
        chain.append(node)
        name = node.output[0]
    return chain


def _last_axis_slice(node, initializers, shape_of) -> Optional[Tuple[object, Tuple[int, int]]]:
    if node.op_type != 'Slice' or node.input[0] not in shape_of:
        return None
    params = [initializers.get(name) for name in node.input[1:]]
    if any(p is None for p in params[:2]) or (len(params) > 3 and params[3] is not None and np.any(params[3] != 1)):
        return None
    rank = len(shape_of[node.input[0]])
    axes = np.asarray(params[2]).reshape(-1) if len(params) > 2 and params[2] is not None else np.arange(1)
    if axes.size != 1 or int(axes[0]) % rank != rank - 1:
        return None
    extent = shape_of[node.input[0]][-1]
    start, end = (int(np.clip(int(np.asarray(p).reshape(-1)[0]), 0, extent)) for p in params[:2])
    return (node, (start, end)) if start < end else None


def _dequantized(source: dict, start: int, end: int, name: str, initializers, edit: GraphEdit) -> str:
    """The columns start:end of a dequantized constant (W or a bias), as a new constant and its DequantizeLinear."""
    node = source['node']
    quantized = np.take(initializers[node.input[0]], np.arange(start, end), axis=source['columns'])
    inputs = [edit.constant(f'{name}_q', quantized), *node.input[1:]]
    sliced_axis = int(attr(node, 'axis', 1)) % np.asarray(initializers[node.input[0]]).ndim == source['columns']
    for i, param in enumerate(node.input[1:], start=1):
        value = np.asarray(initializers[param])
        if value.size > 1 and sliced_axis:  # per-axis parameters along the split axis follow its slice
            inputs[i] = edit.constant(f'{name}_p{i}', value[start:end])
    out = f'{name}_dq'
    edit.emit('DequantizeLinear', inputs, [out], f'{name}_dequantize', **attributes(node))
    if source['transposed']:
        edit.emit('Transpose', [out], [f'{out}_t'], f'{name}_transpose', perm=[1, 0])
        out = f'{out}_t'
    return out


def _live(nodes, outputs) -> List:
    """`nodes` but those whose outputs nothing reads and no graph output is, transitively."""
    while True:
        read = {name for node in nodes for name in node.input} | set(outputs)
        live = [node for node in nodes if any(name in read for name in node.output)]
        if len(live) == len(nodes):
            return nodes
        nodes = live
