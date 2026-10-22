from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def check_io_view(node, generation: str) -> None:
    """Check the tensors' ranks and the node's `io_view`: each tensor it names is one the node reads or writes, its
    permutation of that tensor's rank. A tensor not named is viewed in its logical order."""
    max_rank = 5 if 'AIE-MLV2' in (generation or '').upper() else 4
    tensors = {'inputs': node.inputs, 'outputs': node.outputs}
    for tensor in (*node.inputs, *node.outputs):
        if len(tensor.shape) > max_rank:
            raise ValueError(f'{node.name}: tensor rank {len(tensor.shape)} exceeds max {max_rank} for {generation}.')
    view = node.attributes.get('io_view')
    if view is None:
        return
    for direction, perms in (('inputs', view.inputs), ('outputs', view.outputs)):
        by_name = {tensor.name: tensor for tensor in tensors[direction]}
        for name, perm in perms.items():
            if name not in by_name:
                raise ValueError(f'{node.name}: io_view {direction} names {name!r}, which is not one of them.')
            if len(perm) != len(by_name[name].shape):
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
    perm = view_perm(node, tensor, direction)
    if perm is None:
        return logical
    return [int(logical[i]) for i in perm]


def view_perm(node, tensor, direction: str) -> Optional[Tuple[int, ...]]:
    """The axis order the node's `io_view` reads or writes the tensor in; None for its logical order."""
    view = node.attributes.get('io_view')
    return None if view is None else getattr(view, direction).get(tensor.name)
