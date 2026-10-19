from __future__ import annotations

from ....aie_types import FloatIntent
from ....ir import input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver
from ...utils.io import view_shape
from ...utils.precision import resolve_exact_storage_dtype


@family_resolver('add')
class AddFamilyResolver(FamilyResolver):
    op_type = 'add'

    def validate_structure(self, node, _device) -> None:
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        rhs_tensor = input_tensor_for_role(node, 'rhs')
        lhs_shape = tuple(int(x) for x in view_shape(node, lhs_tensor, 'inputs'))
        rhs_shape = tuple(int(x) for x in view_shape(node, rhs_tensor, 'inputs'))
        if len(lhs_shape) < 2:
            raise ValueError(f'{node.name}: elementwise Add requires rank>=2, got {len(lhs_shape)}.')
        if lhs_shape != rhs_shape:
            raise ValueError(
                f'{node.name}: elementwise Add requires exact-shape inputs, got {lhs_shape} and {rhs_shape}.'
            )
        tensors = (lhs_tensor, rhs_tensor, node.outputs[0])
        if any(isinstance(t.precision, FloatIntent) for t in tensors):
            if not all(isinstance(t.precision, FloatIntent) for t in tensors):
                raise ValueError(f'{node.name}: elementwise Add requires lhs/rhs/output to share float precision.')
        precision_lhs = resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name)
        precision_rhs = resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name)
        precision_out = resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name)
        if precision_lhs != precision_rhs or precision_lhs != precision_out:
            raise ValueError(f'{node.name}: elementwise Add requires lhs/rhs/output to use the same storage type.')
