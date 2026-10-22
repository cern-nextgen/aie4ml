from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..device_catalog import resolve_device
from ..ir.context import AIEBackendContext, ProjectConfig
from ..ir.graph import ROUTE_MODES, STAGING_CONTRACTS, TENSOR_LAYOUTS
from ..op_impls.common_types import PORT_KIND_BUFFER, PORT_KINDS

_DIRECTIVE_FIELDS = {
    'placement': ('col', 'row'),
    'microtiling': ('microtile_m', 'microtile_k', 'microtile_n'),
    'parallelism': ('cas_num', 'cas_length', 'contract'),
    'io_route': ('inputs', 'outputs'),
    'hccs': ('B', 'S', 'Dmax', 'param_sets', 'inv_shift', 'use_clb'),
}
DIRECTIVES = frozenset({*_DIRECTIVE_FIELDS, 'layout', 'ports', 'approximation', 'input_fold'})
LAYER_NODE_SUFFIX = '_aie'  # a layer's node is `<layer>_aie`


def create_context(part: Any, aie_config: Dict[str, Any], project_config: ProjectConfig) -> AIEBackendContext:
    """A model's backend context: the device `part` names, and its AIE configuration completed from the device
    catalog, the PL options normalized."""
    from ..system_plan import normalize_pl_config

    if not part:
        raise KeyError('the config names no Part (or AIEConfig.Part).')
    device, merged = resolve_device(part, dict(aie_config))
    resolved = normalize_pl_config({**merged, 'Part': str(part)})
    return AIEBackendContext(device=device, project_config=project_config, aie_config=resolved)


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

    if 'input_fold' in raw:
        if not isinstance(raw['input_fold'], bool):
            raise TypeError(f'{name}: input_fold must be True or False.')
        directives['input_fold'] = raw['input_fold']

    if 'approximation' in raw:
        directives['approximation'] = str(raw['approximation'])

    if 'hccs' in raw:
        directives['hccs'] = _group(name, 'hccs', raw['hccs'])

    return directives


def design_directives(
    pipeline: Path, directives: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """`directives` (LayerDirectives) with the design a lowering recorded in `pipeline`, its aie_pipeline.json,
    pinned: each layer's schedule and its place. Lowering the same model with them builds that design again; the
    kernels the compiler adds and the hand-overs it shares follow from the layers pinned. A directive the record
    contradicts is refused."""
    recorded = json.loads(Path(pipeline).read_text())
    placed = recorded['physical']['placements']
    merged = {layer: dict(given) for layer, given in (directives or {}).items()}
    for node in recorded['logical']:
        name = node['name']
        design = dict(node['schedule'])
        if name in placed:
            design['placement'] = {key: placed[name][key] for key in ('col', 'row')}
        if not design:
            continue
        layer = merged.setdefault(name.removesuffix(LAYER_NODE_SUFFIX), {})
        for key, value in design.items():
            given = layer.get(key)
            if given is not None and given != ({k: value.get(k) for k in given} if isinstance(value, dict) else value):
                raise ValueError(f'{name}: directive {key}={given} contradicts the recorded design ({value}).')
            layer[key] = value
    return merged
