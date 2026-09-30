from __future__ import annotations

from typing import Any, Dict, List


def check_io_view(node, generation: str) -> None:
    """Check the tensors' ranks and the node's `io_view` trait. A tensor with no entry is viewed in its
    logical order; an entry holds only a `perm`, and buffer_order is derived (see TensorView)."""
    max_rank = 5 if 'AIE-MLV2' in (generation or '').upper() else 4
    tensors = {'inputs': node.inputs, 'outputs': node.outputs}
    for tensor in (*node.inputs, *node.outputs):
        if len(tensor.shape) > max_rank:
            raise ValueError(f'{node.name}: tensor rank {len(tensor.shape)} exceeds max {max_rank} for {generation}.')
    trait = node.traits.get('io_view')
    if trait is None:
        return
    if not set(trait.data) <= set(tensors):
        raise ValueError(f'{node.name}: io_view holds {sorted(trait.data)}, not inputs/outputs.')
    for direction, entries in trait.data.items():
        by_name = {tensor.name: tensor for tensor in tensors[direction]}
        for name, entry in entries.items():
            if name not in by_name:
                raise ValueError(f'{node.name}: io_view {direction} names {name!r}, which is not one of them.')
            if not set(entry) <= {'perm'}:
                raise ValueError(f'{node.name}: io_view entry for {name!r} holds {sorted(entry)}, not perm.')
            perm = entry.get('perm')
            if perm is not None and sorted(perm) != list(range(len(by_name[name].shape))):
                raise ValueError(f'{node.name}: io_view perm {perm} is not a permutation of {name!r}.')


def resolve_io_route(node) -> Dict[str, Any]:
    route = {'inputs': {}, 'outputs': {}}
    for tensor in node.inputs:
        route['inputs'][tensor.name] = 'auto'
    for tensor in node.outputs:
        route['outputs'][tensor.name] = 'auto'

    for direction, modes in node.directives.get('io_route', {}).items():
        unknown = sorted(set(modes) - set(route[direction]))
        if unknown:
            raise ValueError(
                f'{node.name}: io_route names {unknown}, which are not among its {direction} '
                f'{sorted(route[direction])}.'
            )
        route[direction].update(modes)
    return route


def view_shape(node, tensor, direction: str) -> List[int]:
    logical = [int(x) for x in tensor.shape]
    view = view_layout(node, tensor, direction)
    perm = view.get('perm')
    if perm is None:
        return logical
    return [int(logical[i]) for i in perm]


def view_layout(node, tensor, direction: str) -> Dict[str, Any]:
    """The tensor's `io_view` entry, checked by `check_io_view`; none means its logical order."""
    trait = node.traits.get('io_view')
    return {} if trait is None else trait.data.get(direction, {}).get(tensor.name, {})
