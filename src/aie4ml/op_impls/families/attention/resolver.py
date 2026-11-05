from __future__ import annotations

import numpy as np

from ....ir import input_tensor_for_role
from ....ir.graph import has_input_role
from ...family_registry import FamilyResolver, family_resolver
from ...utils.math import divisors
from ...utils.tiling import band_microtiling_candidates


@family_resolver('attention')
class AttentionFamilyResolver(FamilyResolver):
    op_type = 'attention'

    def parallelism_candidates(self, node, device):
        """Query chains (8 queries at least, one per array row), each one tile or two: one for the probabilities, one
        for the values and the division."""
        blocks = int(node.outputs[0].shape[0]) // 8
        rows = max(1, int(device.rows) - int(device.row_start))
        return tuple(
            {'contract': 'outer', 'cas_num': n, 'cas_length': c} for n in divisors(blocks) if n <= rows for c in (1, 2)
        )

    def microtiling_candidates(self, node, device):
        """The row bands a head projecting its own Q may write O in, so the Dense after it reads O directly; O is in
        Q's microtile otherwise."""
        return band_microtiling_candidates(node, device) if has_input_role(node, 'x') else ()

    def macs(self, node) -> int:
        tokens, head_dim = (int(x) for x in node.outputs[0].shape)
        projected = (
            3 * tokens * head_dim * int(input_tensor_for_role(node, 'x').shape[1]) if has_input_role(node, 'x') else 0
        )
        return 2 * tokens * tokens * head_dim + projected

    def validate_structure(self, node, _device) -> None:
        out = tuple(int(x) for x in node.outputs[0].shape)
        if has_input_role(node, 'x'):
            x = tuple(int(d) for d in input_tensor_for_role(node, 'x').shape)
            shapes = {f'w{r}': tuple(np.shape(input_tensor_for_role(node, f'w{r}').data)) for r in 'qkv'}
            if len(x) != 2 or x[0] != out[0] or set(shapes.values()) != {(x[1], out[1])}:
                raise ValueError(f'{node.name}: attention projects x {x} by {shapes} into O {out}.')
        else:
            shapes = {role: tuple(int(d) for d in input_tensor_for_role(node, role).shape) for role in ('q', 'k', 'v')}
            if set(shapes.values()) != {out}:
                raise ValueError(
                    f'{node.name}: attention takes Q, K, V and gives O of one (tokens, head_dim), got '
                    f'{shapes} and {out}.'
                )
        if len(out) != 2 or out[0] % 8 or out[1] % 8:
            raise ValueError(f'{node.name}: attention needs tokens and head_dim in whole 8s, got {out}.')
        if has_input_role(node, 'bias'):
            u = tuple(int(d) for d in input_tensor_for_role(node, 'bias').shape)
            if u != (out[0], out[0]):
                raise ValueError(f'{node.name}: attention adds a bias of (tokens, tokens) to its logits, got {u}.')
