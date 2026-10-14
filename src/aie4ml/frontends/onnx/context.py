# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Import state and tensor plumbing shared by all ONNX handlers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ...aie_types import FloatFormat, FloatIntent, QuantIntent
from ...ir.context import AIEBackendContext
from ...ir.graph import LogicalIR, OpNode, TensorVar
from ..common import LAYER_NODE_SUFFIX
from .utils import intent_from_initializer

_ONNX_FLOAT = 1
_ONNX_BFLOAT16 = 16
_ONNX_FP8E4M3FN = 17

_FLOAT_INPUT_INTENT = {
    _ONNX_FLOAT: FloatIntent(width=32, format=FloatFormat.FP32),
    _ONNX_BFLOAT16: FloatIntent(width=16, format=FloatFormat.BF16),
    _ONNX_FP8E4M3FN: FloatIntent(width=8, format=FloatFormat.FP8_E4M3),
}


class OnnxImportContext:
    """Mutable state threaded through the ONNX handler dispatch."""

    def __init__(
        self,
        backend: AIEBackendContext,
        *,
        initializers: Dict[str, np.ndarray],
        input_shapes: Dict[str, Tuple[int, ...]],
        input_dtypes: Dict[str, int],
        shape_of: Dict[str, Tuple[int, ...]],
        layer_directives: Dict[str, Any],
    ):
        self.backend = backend
        self.graph: LogicalIR = backend.ir.logical
        self.device = backend.device
        self.initializers = initializers
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.shape_of = shape_of
        self.layer_directives = layer_directives
        self.used_directives: set[str] = set()

        self.value_tensors: Dict[str, TensorVar] = {}
        self.q_aliases: Dict[str, Tuple[str, Any, QuantIntent, Tuple[int, ...]]] = {}
        self.precision_mirrors: list[Tuple[str, str]] = []
        self.value_orders: Dict[str, Tuple[int, ...]] = {}

    # -- directives ---------------------------------------------------------

    def mirror_precision(self, target: str, source: str) -> None:
        """Declare that `target` carries the same quantization intent as `source`."""
        self.precision_mirrors.append((target, source))

    def take_directives(self, node_name: str) -> Dict[str, Any]:
        from ..common import normalize_directives

        directives = normalize_directives(node_name, self.layer_directives.get(node_name))
        if directives:
            self.used_directives.add(node_name)
        return directives

    def output_shape(self, name: str, node_name: str) -> Tuple[int, ...]:
        if name not in self.shape_of:
            raise ValueError(f'{node_name}: shape inference produced no shape for {name!r}.')
        return tuple(int(x) for x in self.shape_of[name])

    # -- tensor construction ------------------------------------------------

    def graph_input_tensor(self, name: str, shape: Sequence[int], intent: QuantIntent) -> TensorVar:
        tensor = self.graph.tensors.get(name)
        if tensor is None:
            tensor = TensorVar(name=name, shape=tuple(int(x) for x in shape), precision=intent)
            self.graph.add_tensor(tensor)
        elif tensor.precision is None:
            tensor.precision = intent
        elif tensor.precision != intent:
            raise ValueError(f'{name}: conflicting quantization intent for graph input.')
        return tensor

    def param_tensor(self, name: str, data: np.ndarray, intent: QuantIntent) -> TensorVar:
        tensor = self.graph.tensors.get(name)
        shape = tuple(int(x) for x in np.asarray(data).shape)
        if tensor is None:
            tensor = TensorVar(name=name, shape=shape, precision=intent, data=np.asarray(data, dtype=np.float64))
            self.graph.add_tensor(tensor)
            return tensor
        if tensor.data is None:
            raise ValueError(f'{name}: expected parameter tensor.')
        if tensor.precision != intent:
            raise ValueError(f'{name}: conflicting quantization intent for parameter tensor.')
        return tensor

    # -- source lookup ------------------------------------------------------

    # -- value order --------------------------------------------------------

    def order_of(self, name: str) -> Optional[Tuple[int, ...]]:
        """How ONNX value `name` views its canonical tensor, or None for the identity view."""
        return self.value_orders.get(name)

    def set_order(self, name: str, order: Optional[Sequence[int]], node_name: str) -> None:
        if order is None:
            self.value_orders.pop(name, None)
            return
        order = tuple(int(x) for x in order)
        if sorted(order) != list(range(len(order))):
            raise ValueError(f'{node_name}: {order} is not a permutation of rank {len(order)}.')
        if list(order) == sorted(order):
            self.value_orders.pop(name, None)
        else:
            self.value_orders[name] = order

    def canonical_axis(self, name: str, onnx_axis: int) -> int:
        """The canonical axis that ONNX value `name` calls `onnx_axis`: an axis a view op records."""
        order = self.value_orders.get(name)
        return int(order[onnx_axis]) if order is not None else int(onnx_axis)

    def canonical_shape(self, name: str, node_name: str) -> Tuple[int, ...]:
        """The canonical shape behind ONNX value `name`, whose shape is seen through its view."""
        view_shape = self.output_shape(name, node_name)
        order = self.value_orders.get(name) or tuple(range(len(view_shape)))
        shape = [0] * len(view_shape)
        for view_axis, canonical_axis in enumerate(order):
            shape[int(canonical_axis)] = int(view_shape[view_axis])
        return tuple(shape)

    def propagate_order(self, src: str, dst: str) -> None:
        """Carry a value's view across an op that changes neither shape nor axis order."""
        order = self.value_orders.get(src)
        if order is None:
            self.value_orders.pop(dst, None)
        else:
            self.value_orders[dst] = order

    def canonical_source(self, name: str, node_name: str) -> TensorVar:
        """An activation in its own axis order, materializing a folded view if one is pending.

        A handler that reads a tensor axis by axis calls this instead of `source_for`: a value
        that still carries a view becomes a `transpose` op, which the view passes fold into the
        consumer that can realize it -- and refuse on the consumer that cannot.
        """
        tensor = self.source_for(name, node_name)
        order = self.value_orders.get(name)
        if order is None:
            return tensor
        view_name = f'{name}_as_viewed'
        self.emit(
            'transpose',
            view_name,
            inputs=[tensor],
            outputs=[(view_name, self.output_shape(name, node_name), tensor.precision)],
            roles=['lhs'],
            metadata={
                'perm': [int(axis) for axis in order],
                'data_format': 'channels_last',
                'layer_class': 'Transpose',
                'source_layer': node_name,
            },
            directives={},
        )
        materialized = self.value_tensors[view_name]
        self.bind(name, materialized)  # the value is now in its own order for every later reader
        self.set_order(name, None, node_name)
        return materialized

    def common_order(self, names: Sequence[str], node_name: str) -> Optional[Tuple[int, ...]]:
        """The axis order shared by these values, for an op that reads them element for element."""
        orders = {self.value_orders.get(name) for name in names}
        if len(orders) > 1:
            raise ValueError(f'{node_name}: its inputs view their tensors in different axis orders ({orders}).')
        return next(iter(orders), None)

    def require_order(self, name: str, expected: Sequence[int], node_name: str) -> None:
        """Refuse a value that is not in the specific axis order this handler reads."""
        order = self.value_orders.get(name) or tuple(range(len(expected)))
        if tuple(order) != tuple(int(x) for x in expected):
            raise ValueError(
                f'{node_name}: input {name} views its tensor as {order}, but {node_name} reads '
                f'{tuple(int(x) for x in expected)}.'
            )

    def source_for(self, name: str, node_name: str) -> TensorVar:
        """An activation tensor; never a constant. Axis order is the handler's own contract: state
        it with `require_identity_order`, `require_order` or `propagate_order`."""
        if name in self.value_tensors:
            return self.value_tensors[name]
        if name in self.input_shapes:
            tensor = self.any_source_for(name, node_name)
            if tensor.is_parameter:
                raise ValueError(f'{node_name}: activation input {name} cannot be constant.')
            return tensor
        raise ValueError(f'{node_name}: unsupported input {name}. Expected a dequantized activation tensor.')

    def parameter_source_for(self, name: str, node_name: str) -> TensorVar:
        """A constant tensor, materialized from an initializer if needed."""
        if name in self.value_tensors:
            tensor = self.value_tensors[name]
            if not tensor.is_parameter:
                raise ValueError(f'{node_name}: parameter input {name} must be constant.')
            return tensor
        if name in self.initializers:
            data = np.asarray(self.initializers[name])
            return self.param_tensor(name, data, intent_from_initializer(data, node_name))
        raise ValueError(f'{node_name}: parameter input {name} must be a constant initializer.')

    def any_source_for(self, name: str, node_name: str) -> TensorVar:
        """Either an activation, a parameter, or a bare float graph input."""
        if name in self.value_tensors:
            return self.value_tensors[name]
        if name in self.initializers:
            return self.parameter_source_for(name, node_name)
        if name in self.input_shapes:
            elem_type = self.input_dtypes[name]
            intent = _FLOAT_INPUT_INTENT.get(elem_type)
            if intent is None:
                raise ValueError(
                    f'{node_name}: graph input "{name}" has unsupported ONNX elem_type {elem_type}. '
                    'Use QDQ wrapping for integer inputs.'
                )
            tensor = self.graph_input_tensor(name, self.input_shapes[name], intent)
            self.value_tensors[name] = tensor
            return tensor
        raise ValueError(f'{node_name}: unsupported input {name}.')

    # -- registration / emission -------------------------------------------

    def bind(self, name: str, tensor: TensorVar) -> None:
        self.value_tensors[name] = tensor

    def emit(
        self,
        op_type: str,
        node_name: str,
        *,
        inputs: Sequence[TensorVar],
        outputs: Sequence[Tuple[str, Sequence[int], Optional[Any]]],
        metadata: Dict[str, Any],
        directives: Dict[str, Any],
        roles: Optional[Sequence[str]] = None,
    ) -> OpNode:
        """Create a semantic OpNode, wire it, and register its outputs."""
        op = OpNode(name=f'{node_name}{LAYER_NODE_SUFFIX}', op_type=op_type)
        op.metadata.update(metadata)
        if roles is not None:
            op.metadata['input_roles'] = list(roles)
        op.directives.update(directives)

        for src in inputs:
            src.consumers.append(op)
        op.inputs.extend(inputs)

        for out_name, out_shape, out_precision in outputs:
            out_tensor = TensorVar(
                name=out_name,
                shape=tuple(int(x) for x in out_shape),
                precision=out_precision,
                producer=op,
            )
            self.graph.add_tensor(out_tensor)
            op.outputs.append(out_tensor)
            self.value_tensors[out_name] = out_tensor

        self.graph.add_node(op)
        return op
