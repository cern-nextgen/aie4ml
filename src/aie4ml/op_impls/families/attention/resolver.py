from __future__ import annotations

from ....ir import input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


@family_resolver('attention')
class AttentionFamilyResolver(FamilyResolver):
    op_type = 'attention'

    def parallelism_candidates(self, node, device):
        """Query chains (8 queries at least, one per array row), each one tile or two: one for the probabilities, one
        for the values and the division."""
        blocks = int(input_tensor_for_role(node, 'q').shape[0]) // 8
        rows = max(1, int(device.rows) - int(device.row_start))
        return tuple(
            {'contract': 'outer', 'cas_num': n, 'cas_length': c} for n in _divisors(blocks) if n <= rows for c in (1, 2)
        )

    def macs(self, node) -> int:
        tokens, head_dim = (int(x) for x in input_tensor_for_role(node, 'q').shape)
        return 2 * tokens * tokens * head_dim

    def validate_structure(self, node, _device) -> None:
        shapes = {role: tuple(int(x) for x in input_tensor_for_role(node, role).shape) for role in ('q', 'k', 'v')}
        shapes['output'] = tuple(int(x) for x in node.outputs[0].shape)
        if len(set(shapes.values())) != 1 or len(shapes['q']) != 2:
            raise ValueError(f'{node.name}: attention takes Q, K, V and gives O of one (tokens, head_dim), got {shapes}.')
        tokens, head_dim = shapes['q']
        if tokens % 8 or head_dim % 8:
            raise ValueError(f'{node.name}: attention needs tokens and head_dim in whole 8s, got {shapes["q"]}.')
