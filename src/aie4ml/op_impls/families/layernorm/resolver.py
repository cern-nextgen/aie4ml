from __future__ import annotations

import math

from ....ir import input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver
from ...utils.io import view_shape
from ...utils.tiling import row_band_candidates


@family_resolver('layer_norm')
class LayerNormFamilyResolver(FamilyResolver):
    op_type = 'layer_norm'

    def parallelism_candidates(self, node, device):
        return row_band_candidates(node, device)

    def validate_structure(self, node, _device) -> None:
        in_tensor = input_tensor_for_role(node, 'lhs')
        gamma_tensor = input_tensor_for_role(node, 'gamma')
        beta_tensor = input_tensor_for_role(node, 'beta')
        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        if len(in_shape) < 2:
            raise ValueError(f'{node.name}: LayerNorm requires rank>=2 input, got {len(in_shape)}.')
        axis = int(node.metadata.get('axis', -1))
        if axis < 0:
            axis += len(in_shape)
        if axis != len(in_shape) - 1:
            raise ValueError(f'{node.name}: integer LayerNorm only supports last-axis normalization; got axis={axis}.')
        full_inner = int(in_shape[-1])
        for role, tensor in (('gamma', gamma_tensor), ('beta', beta_tensor)):
            if not tensor.is_parameter:
                raise ValueError(f'{node.name}: LayerNorm {role} must be a constant parameter tensor.')
            param_elems = int(math.prod(tuple(int(x) for x in tensor.shape)))
            if param_elems != full_inner:
                raise ValueError(
                    f'{node.name}: LayerNorm {role} length {param_elems} must match inner axis={full_inner}.'
                )
