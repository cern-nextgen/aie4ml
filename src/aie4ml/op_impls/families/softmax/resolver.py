from __future__ import annotations

from ....ir import input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver
from ...utils.io import view_shape
from ...utils.tiling import row_band_candidates


@family_resolver('softmax')
class SoftmaxFamilyResolver(FamilyResolver):
    op_type = 'softmax'

    def parallelism_candidates(self, node, device):
        return row_band_candidates(node, device)

    def validate_structure(self, node, _device) -> None:
        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]
        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        out_shape = tuple(int(x) for x in view_shape(node, out_tensor, 'outputs'))
        if len(in_shape) < 2:
            raise ValueError(f'{node.name}: Softmax requires rank>=2 tensors, got {len(in_shape)}.')
        if in_shape != out_shape:
            raise ValueError(f'{node.name}: Softmax input/output shapes must match, got {in_shape} and {out_shape}.')
        axis = int(node.metadata.get('axis', -1))
        if axis < 0:
            axis += len(in_shape)
        if axis != len(in_shape) - 1:
            raise ValueError(f'{node.name}: only last-axis Softmax is supported; got axis={axis}.')
