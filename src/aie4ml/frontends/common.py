from __future__ import annotations

from typing import Any, Dict

from ..ir.graph import ROUTE_MODES, STAGING_CONTRACTS, TENSOR_LAYOUTS
from ..op_impls.common_types import PORT_KIND_BUFFER, PORT_KINDS

_DIRECTIVE_FIELDS = {
    'placement': ('col', 'row'),
    'microtiling': ('microtile_m', 'microtile_k', 'microtile_n'),
    'parallelism': ('cas_num', 'cas_length', 'contract'),
    'io_route': ('inputs', 'outputs'),
    'hccs': ('B', 'S', 'Dmax', 'param_sets', 'inv_shift', 'use_clb'),
}
DIRECTIVES = frozenset({*_DIRECTIVE_FIELDS, 'layout', 'ports', 'approximation'})


def _group(name: str, key: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f'{name}: the {key} directive must be a dict.')
    fields = _DIRECTIVE_FIELDS[key]
    unknown = sorted(set(value) - set(fields))
    if unknown or not value:
        raise ValueError(f'{name}: {key} takes some of {list(fields)}, got {sorted(value)}.')
    return dict(value)


def normalize_directives(name: str, raw: Any) -> Dict[str, Any]:
    """Validate one layer's directives. Anything unknown is refused, never dropped."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f'{name}: layer directives must be a dict.')
    unknown = sorted(set(raw) - DIRECTIVES)
    if unknown:
        raise ValueError(f'{name}: unknown directive(s) {unknown}; expected some of {sorted(DIRECTIVES)}.')

    directives: Dict[str, Any] = {}

    if 'placement' in raw:
        placement = _group(name, 'placement', raw['placement'])
        if set(placement) != {'col', 'row'}:
            raise ValueError(f'{name}: placement needs both col and row.')
        directives['placement'] = {key: int(value) for key, value in placement.items()}

    if 'microtiling' in raw:
        directives['microtiling'] = {key: int(v) for key, v in _group(name, 'microtiling', raw['microtiling']).items()}

    if 'parallelism' in raw:
        parallelism = _group(name, 'parallelism', raw['parallelism'])
        directives['parallelism'] = {key: str(v) if key == 'contract' else int(v) for key, v in parallelism.items()}
        contract = directives['parallelism'].get('contract')
        if contract is not None and contract not in STAGING_CONTRACTS:
            raise ValueError(f'{name}: unknown contract {contract!r}; expected one of {sorted(STAGING_CONTRACTS)}.')

    if 'io_route' in raw:
        routes: Dict[str, Dict[str, str]] = {}
        for direction, modes in _group(name, 'io_route', raw['io_route']).items():
            if not isinstance(modes, dict):
                raise TypeError(f'{name}: io_route {direction} must map tensor names to route modes.')
            bad = {tensor: mode for tensor, mode in modes.items() if mode not in ROUTE_MODES}
            if bad:
                raise ValueError(f'{name}: io_route modes {bad}; expected one of {sorted(ROUTE_MODES)}.')
            routes[direction] = {str(tensor): str(mode) for tensor, mode in modes.items()}
        directives['io_route'] = routes

    if 'layout' in raw:
        layout = str(raw['layout'])
        if layout not in TENSOR_LAYOUTS:
            raise ValueError(f'{name}: unknown layout {layout!r}; expected one of {sorted(TENSOR_LAYOUTS)}.')
        directives['layout'] = layout

    if 'ports' in raw:
        # One kind for both directions, or one per direction; the default is a buffer.
        asked = raw['ports'] if isinstance(raw['ports'], dict) else {'inputs': raw['ports'], 'outputs': raw['ports']}
        ports = {direction: str(asked.get(direction, PORT_KIND_BUFFER)) for direction in ('inputs', 'outputs')}
        if set(asked) - set(ports) or set(ports.values()) - PORT_KINDS:
            raise ValueError(
                f'{name}: unknown ports {raw["ports"]!r}; expected one of {sorted(PORT_KINDS)}, or one per direction '
                "as {'inputs': ..., 'outputs': ...}."
            )
        directives['ports'] = ports

    if 'approximation' in raw:
        directives['approximation'] = str(raw['approximation'])

    if 'hccs' in raw:
        directives['hccs'] = _group(name, 'hccs', raw['hccs'])

    return directives
