# Copyright 2026 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Render the emitted physical AIE layout without invoking the toolchain."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .device_catalog import lookup_device
from .passes.utils import sanitize_identifier

_COLORS = (
    '#8ecae6',
    '#ffb703',
    '#90be6d',
    '#f28482',
    '#b8a1d9',
    '#84dcc6',
    '#f6bd60',
    '#a8dadc',
    '#f7a072',
    '#bde0fe',
)
_ANSI = (39, 214, 77, 203, 141, 43, 220, 44, 209, 117)
_KINDS = {'shared_memory': 'shared', 'dma': 'DMA', 'stream': 'stream', 'memtile': 'memtile'}
_KIND_ORDER = {'shared_memory': 0, 'dma': 1, 'stream': 2, 'memtile': 3}


class Layout(dict):
    """The emitted physical layout, displayable directly in a terminal or notebook."""

    def __str__(self) -> str:
        return format_terminal_layout(self)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        """Let notebooks show the full-width SVG without clipping terminal text."""
        return f'<div style="max-width:100%; overflow-x:auto">{format_svg_layout(self)}</div>'


def load_layout(project: Path) -> Layout:
    """Load and validate the placement information emitted by aie4ml."""
    project = Path(project)
    pipeline = project / 'aie_pipeline.json'
    if not project.exists():
        raise FileNotFoundError(f'{project}: project directory not found.')
    if not pipeline.exists():
        raise FileNotFoundError(f'{pipeline}: not found; emit the AIE project before requesting its layout.')
    try:
        document = json.loads(pipeline.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'{pipeline}: invalid JSON: {exc.msg} at line {exc.lineno}.') from exc
    except OSError as exc:
        raise OSError(f'{pipeline}: could not be read: {exc}.') from exc
    return Layout(layout_from_pipeline(document))


def layout_from_pipeline(document: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the graph-level physical layout from an emitted pipeline document."""
    if not isinstance(document, dict):
        raise ValueError('aie_pipeline.json: the root must be an object.')
    execution = document.get('execution')
    physical = document.get('physical')
    if not isinstance(execution, list) or not isinstance(physical, dict):
        raise ValueError('aie_pipeline.json: missing execution or physical IR.')
    placements = physical.get('placements')
    plan = physical.get('plan')
    if not isinstance(placements, dict) or not isinstance(plan, dict):
        raise ValueError('aie_pipeline.json: physical IR has no placements or materialized plan.')

    token_width = max(2, len(str(max(1, len(execution)))))
    layers = []
    occupancy: Dict[Tuple[int, int], str] = {}
    endpoint_ids: Dict[str, str] = {}
    order: Dict[str, int] = {}
    for index, entry in enumerate(execution, 1):
        if not isinstance(entry, dict) or not isinstance(entry.get('node'), str) or not entry['node']:
            raise ValueError(f'aie_pipeline.json: execution entry {index} has no node name.')
        name = entry['node']
        if name in order:
            raise ValueError(f'aie_pipeline.json: duplicate execution node {name!r}.')
        placement = placements.get(name)
        if not isinstance(placement, dict):
            raise ValueError(f'aie_pipeline.json: {name} has no physical placement.')
        col, row, width, height = (_nonnegative(placement, key, name) for key in ('col', 'row', 'width', 'height'))
        if width < 1 or height < 1:
            raise ValueError(f'aie_pipeline.json: {name} has invalid placement size {width} x {height}.')
        parallelism = (entry.get('config') or {}).get('parallelism') or {}
        if not isinstance(parallelism, dict):
            raise ValueError(f'aie_pipeline.json: {name} has malformed parallelism.')
        cas_num = _positive(parallelism.get('cas_num', 1), f'{name}.cas_num')
        cas_length = _positive(parallelism.get('cas_length', 1), f'{name}.cas_length')
        contract = str(parallelism.get('contract', 'inner'))
        token = str(index).zfill(token_width)
        layer = {
            'name': name,
            'token': token,
            'color': _COLORS[(index - 1) % len(_COLORS)],
            'ansi': _ANSI[(index - 1) % len(_ANSI)],
            'op_type': str(entry.get('op_type') or '?'),
            'variant': str(entry.get('variant_id') or '?'),
            'contract': contract,
            'cas_num': cas_num,
            'cas_length': cas_length,
            'col': col,
            'row': row,
            'width': width,
            'height': height,
            'tiles': width * height,
        }
        layers.append(layer)
        order[name] = index
        endpoint_id = sanitize_identifier(name)
        previous = endpoint_ids.setdefault(endpoint_id, name)
        if previous != name:
            raise ValueError(f'aie_pipeline.json: {previous!r} and {name!r} have the same generated identifier.')
        for y in range(row, row + height):
            for x in range(col, col + width):
                previous = occupancy.setdefault((x, y), name)
                if previous != name:
                    raise ValueError(f'aie_pipeline.json: {previous} and {name} overlap at tile ({x}, {y}).')

    if set(placements) != set(order):
        extra = sorted(set(placements) - set(order))
        raise ValueError(f'aie_pipeline.json: placements name non-execution graphs {extra}.')
    connections = _connections(plan, endpoint_ids, order)
    if layers:
        occupied = {
            'min_col': min(layer['col'] for layer in layers),
            'max_col': max(layer['col'] + layer['width'] - 1 for layer in layers),
            'min_row': min(layer['row'] for layer in layers),
            'max_row': max(layer['row'] + layer['height'] - 1 for layer in layers),
        }
    else:
        occupied = {'min_col': 0, 'max_col': -1, 'min_row': 0, 'max_row': -1}
    device, bounds = _device_geometry(document.get('device'))
    if bounds is None:
        bounds = occupied
    elif layers and (
        occupied['min_col'] < bounds['min_col']
        or occupied['max_col'] > bounds['max_col']
        or occupied['min_row'] < bounds['min_row']
        or occupied['max_row'] > bounds['max_row']
    ):
        raise ValueError(
            'aie_pipeline.json: a placement lies outside the device array '
            f'columns {bounds["min_col"]}-{bounds["max_col"]}, rows {bounds["min_row"]}-{bounds["max_row"]}.'
        )
    return {
        'layers': layers,
        'occupancy': occupancy,
        'connections': connections,
        'bounds': bounds,
        'occupied_bounds': occupied,
        'device': device,
    }


def _device_geometry(value: Any) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, int]]]:
    if value is None:
        return None, None
    # Accept the temporary development format so projects emitted while this feature settled still render.
    part = value.get('part') if isinstance(value, dict) else value
    if not isinstance(part, str) or not part:
        raise ValueError('aie_pipeline.json: device must name an AIE part.')
    facts = lookup_device(part)
    if not facts:
        raise ValueError(f'aie_pipeline.json: device {part!r} is not in aie_devices.json.')
    device = {'part': str(facts['Part']), 'generation': str(facts['Generation'])}
    bounds = {
        'min_col': 0,
        'max_col': int(facts['Columns']) - 1,
        'min_row': 0,
        'max_row': int(facts['Rows']) - 1,
    }
    return device, bounds


def _nonnegative(mapping: Dict[str, Any], key: str, name: str) -> int:
    if key not in mapping:
        raise ValueError(f'aie_pipeline.json: {name} placement has no {key}.')
    try:
        value = int(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'aie_pipeline.json: {name} placement {key} is not an integer.') from exc
    if value < 0:
        raise ValueError(f'aie_pipeline.json: {name} placement {key} is negative.')
    return value


def _positive(value: Any, field: str) -> int:
    result = _integer(value, field)
    if result < 1:
        raise ValueError(f'aie_pipeline.json: {field} must be positive.')
    return result


def _integer(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'aie_pipeline.json: {field} is not an integer.') from exc
    return result


def _endpoint(endpoint: Any, endpoint_ids: Dict[str, str]) -> Optional[str]:
    if not isinstance(endpoint, str):
        raise ValueError('aie_pipeline.json: a direct edge has a non-string endpoint.')
    root = endpoint.split('.', 1)[0]
    if root.startswith('ifm[') or root.startswith('ofm['):
        return None
    if root not in endpoint_ids:
        raise ValueError(f'aie_pipeline.json: direct edge endpoint {endpoint!r} names no execution graph.')
    return endpoint_ids[root]


def _connections(plan: Dict[str, Any], endpoint_ids: Dict[str, str], order: Dict[str, int]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str, str], int] = {}
    direct = plan.get('direct_edges', [])
    buffers = plan.get('buffers', [])
    if not isinstance(direct, list) or not isinstance(buffers, list):
        raise ValueError('aie_pipeline.json: materialized plan edges and buffers must be lists.')
    for edge in direct:
        if not isinstance(edge, dict):
            raise ValueError('aie_pipeline.json: malformed direct edge.')
        source, target = _endpoint(edge.get('source'), endpoint_ids), _endpoint(edge.get('target'), endpoint_ids)
        if source is None or target is None:
            continue
        kind = edge.get('realization')
        if kind not in _KINDS or kind == 'memtile':
            raise ValueError(f'aie_pipeline.json: {source} -> {target} has unknown direct realization {kind!r}.')
        counts[(source, target, kind)] = counts.get((source, target, kind), 0) + 1

    for buffer in buffers:
        if not isinstance(buffer, dict):
            raise ValueError('aie_pipeline.json: malformed memory-tile buffer.')
        writers, readers = buffer.get('writers'), buffer.get('readers')
        if not isinstance(writers, list) or not isinstance(readers, list):
            raise ValueError(f'aie_pipeline.json: memory-tile buffer {buffer.get("name", "?")} has no endpoints.')
        sources = _op_impl_endpoints(writers, 'source_type', 'source_endpoint', order)
        targets = _op_impl_endpoints(readers, 'target_type', 'target_endpoint', order)
        for source in sources:
            for target in targets:
                counts[(source, target, 'memtile')] = counts.get((source, target, 'memtile'), 0) + 1

    return [
        {'source': source, 'target': target, 'kind': kind, 'count': count}
        for (source, target, kind), count in sorted(
            counts.items(), key=lambda item: (order[item[0][0]], order[item[0][1]], _KIND_ORDER[item[0][2]])
        )
    ]


def _op_impl_endpoints(
    endpoints: Iterable[Dict[str, Any]], type_key: str, endpoint_key: str, order: Dict[str, int]
) -> List[str]:
    found = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise ValueError('aie_pipeline.json: malformed memory-tile endpoint.')
        if endpoint.get(type_key) != 'op_impl':
            continue
        metadata = endpoint.get(endpoint_key)
        name = metadata.get('op_impl') if isinstance(metadata, dict) else None
        if name not in order:
            raise ValueError(f'aie_pipeline.json: memory-tile endpoint names no execution graph: {name!r}.')
        found.add(name)
    return sorted(found, key=order.__getitem__)


def format_terminal_layout(layout: Dict[str, Any], terminal_width: int = 120, color: bool = False) -> str:
    """Render a compact, width-aware terminal view of a physical layout."""
    layers = layout['layers']
    bounds, occupancy = layout['bounds'], layout['occupancy']
    if bounds['max_col'] < bounds['min_col'] or bounds['max_row'] < bounds['min_row']:
        return 'AIE layout\n    no placed kernel graphs'
    by_name = {layer['name']: layer for layer in layers}
    token_width = max((len(layer['token']) for layer in layers), default=2)
    cell_width = max(token_width, len(str(bounds['max_col']))) + 1
    row_label_width = max(5, len(str(bounds['max_row'])) + 4)
    panel_capacity = max(1, (max(20, int(terminal_width)) - row_label_width - 2) // cell_width)
    column_count = bounds['max_col'] - bounds['min_col'] + 1
    panel_count = (column_count + panel_capacity - 1) // panel_capacity
    columns_per_panel = (column_count + panel_count - 1) // panel_count
    total = (bounds['max_col'] - bounds['min_col'] + 1) * (bounds['max_row'] - bounds['min_row'] + 1)
    device = layout.get('device')
    title = 'AIE layout'
    if device:
        title += f'  |  {device["generation"]}  |  {device["part"]}'
    out = [title]
    if device:
        utilization = 100.0 * len(occupancy) / total
        out.append(
            f'    array columns {bounds["min_col"]}-{bounds["max_col"]}, '
            f'rows {bounds["min_row"]}-{bounds["max_row"]}  |  '
            f'{len(occupancy)} of {total} tiles occupied ({utilization:.1f}%)'
        )
    else:
        out.append(
            f'    occupied region columns {bounds["min_col"]}-{bounds["max_col"]}, '
            f'rows {bounds["min_row"]}-{bounds["max_row"]}; device geometry was not emitted'
        )
    for first in range(bounds['min_col'], bounds['max_col'] + 1, columns_per_panel):
        last = min(bounds['max_col'], first + columns_per_panel - 1)
        out.extend(['', f'    columns {first}-{last}'])
        panel_width = (last - first + 1) * cell_width
        out.append(' ' * (row_label_width + 1) + ''.join(f'{col:>{cell_width}}' for col in range(first, last + 1)))
        out.append(' ' * row_label_width + '┌' + '─' * panel_width + '┐')
        for row in range(bounds['max_row'], bounds['min_row'] - 1, -1):
            cells = []
            for col in range(first, last + 1):
                name = occupancy.get((col, row))
                token = by_name[name]['token'] if name else '·'
                cell = f'{token:^{cell_width}}'
                if color and name:
                    cell = f'\x1b[38;5;16;48;5;{by_name[name]["ansi"]}m{cell}\x1b[0m'
                elif color:
                    cell = f'\x1b[38;5;240m{cell}\x1b[0m'
                cells.append(cell)
            out.append(f'{f"r{row}":>{row_label_width}}│' + ''.join(cells) + '│')
        out.append(' ' * row_label_width + '└' + '─' * panel_width + '┘')

    out.extend(['', 'Layers  (split is cas_num × cas_length)'])
    for layer in layers:
        token = layer['token']
        if color:
            token = f'\x1b[38;5;16;48;5;{layer["ansi"]}m {token}\x1b[0m'
        tile_word = 'tile' if layer['tiles'] == 1 else 'tiles'
        out.append(
            f'    {token}  {layer["name"]}  {layer["variant"]}  '
            f'{layer["contract"]} {layer["cas_num"]}×{layer["cas_length"]}  '
            f'{layer["tiles"]} {tile_word} at ({layer["col"]},{layer["row"]}) '
            f'{layer["width"]}×{layer["height"]}'
        )
    out.extend(['', 'Connections  (graph level)'])
    grouped = _group_connections(layout['connections'])
    if not grouped:
        out.append('    none')
    for source, target, kinds in grouped:
        details = ', '.join(f'{_KINDS[kind]} ×{count}' for kind, count in kinds)
        out.append(f'    {by_name[source]["token"]} {source} -> {by_name[target]["token"]} {target}: {details}')
    return '\n'.join(out)


def _group_connections(connections: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for edge in connections:
        grouped.setdefault((edge['source'], edge['target']), []).append((edge['kind'], int(edge['count'])))
    return [(source, target, kinds) for (source, target), kinds in grouped.items()]


def format_svg_layout(layout: Dict[str, Any]) -> str:
    """Render the same graph-level layout as a standalone SVG."""
    layers = layout['layers']
    if not layers:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="80">'
            '<text x="20" y="45">No placed AIE graphs</text></svg>\n'
        )
    bounds, occupancy = layout['bounds'], layout['occupancy']
    by_name = {layer['name']: layer for layer in layers}
    cell, left, top = 34, 54, 76
    cols = bounds['max_col'] - bounds['min_col'] + 1
    rows = bounds['max_row'] - bounds['min_row'] + 1
    grid_width, grid_height = cols * cell, rows * cell
    legend_y = top + grid_height + 42
    connection_rows = max(1, len(_group_connections(layout['connections'])))
    height = legend_y + 24 * (len(layers) + connection_rows + 3)
    width = max(680, left + grid_width + 36)

    def point(col: float, row: float) -> Tuple[float, float]:
        return left + (col - bounds['min_col']) * cell, top + (bounds['max_row'] - row) * cell

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#374151"/></marker></defs>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="18" ' 'font-weight="bold">AIE physical layout</text>',
        f'<text x="20" y="48" font-family="sans-serif" font-size="12" fill="#475569">'
        f'array columns {bounds["min_col"]}-{bounds["max_col"]}, '
        f'rows {bounds["min_row"]}-{bounds["max_row"]}</text>',
    ]
    for col in range(bounds['min_col'], bounds['max_col'] + 1):
        x, _ = point(col, bounds['max_row'])
        lines.append(
            f'<text x="{x + cell / 2}" y="{top - 10}" text-anchor="middle" '
            f'font-family="monospace" font-size="11">{col}</text>'
        )
    for row in range(bounds['max_row'], bounds['min_row'] - 1, -1):
        _, y = point(bounds['min_col'], row)
        lines.append(
            f'<text x="{left - 10}" y="{y + 22}" text-anchor="end" '
            f'font-family="monospace" font-size="11">r{row}</text>'
        )
        for col in range(bounds['min_col'], bounds['max_col'] + 1):
            x, _ = point(col, row)
            name = occupancy.get((col, row))
            fill = by_name[name]['color'] if name else '#f8fafc'
            if name:
                layer = by_name[name]
                title = escape(
                    f'{name}: {layer["variant"]}, {layer["contract"]} '
                    f'{layer["cas_num"]}x{layer["cas_length"]}, tile ({col},{row})'
                )
                lines.extend(
                    [
                        f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" '
                        f'stroke="#cbd5e1"><title>{title}</title></rect>',
                        f'<text x="{x + cell / 2}" y="{y + 22}" text-anchor="middle" '
                        f'font-family="monospace" font-size="11">{layer["token"]}</text>',
                    ]
                )
            else:
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="#cbd5e1"><title>tile ({col},{row})</title></rect>'
                )

    for source, target, kinds in _group_connections(layout['connections']):
        a, b = by_name[source], by_name[target]
        x1, y1 = point(a['col'] + a['width'] / 2, a['row'] + (a['height'] - 1) / 2)
        x2, y2 = point(b['col'] + b['width'] / 2, b['row'] + (b['height'] - 1) / 2)
        label = escape(', '.join(f'{_KINDS[k]} ×{n}' for k, n in kinds))
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" '
            f'stroke-width="2" opacity="0.7" marker-end="url(#arrow)">'
            f'<title>{escape(source)} -&gt; {escape(target)}: {label}</title></line>'
        )

    lines.append(
        f'<text x="20" y="{legend_y}" font-family="sans-serif" font-size="14" '
        'font-weight="bold">Layers (split is cas_num × cas_length)</text>'
    )
    y = legend_y + 24
    for layer in layers:
        description = escape(
            f'{layer["token"]}  {layer["name"]}  {layer["variant"]}  '
            f'{layer["contract"]} {layer["cas_num"]}×{layer["cas_length"]}  '
            f'{layer["tiles"]} tiles at ({layer["col"]},{layer["row"]}) {layer["width"]}×{layer["height"]}'
        )
        lines.extend(
            [
                f'<rect x="20" y="{y - 13}" width="14" height="14" ' f'fill="{layer["color"]}" stroke="#64748b"/>',
                f'<text x="42" y="{y}" font-family="monospace" font-size="12">{description}</text>',
            ]
        )
        y += 24
    lines.append(
        f'<text x="20" y="{y + 8}" font-family="sans-serif" font-size="14" '
        'font-weight="bold">Connections (graph level)</text>'
    )
    y += 32
    grouped = _group_connections(layout['connections'])
    if not grouped:
        lines.append(f'<text x="42" y="{y}" font-family="monospace" font-size="12">none</text>')
    for source, target, kinds in grouped:
        details = ', '.join(f'{_KINDS[kind]} ×{count}' for kind, count in kinds)
        lines.append(
            f'<text x="42" y="{y}" font-family="monospace" font-size="12">'
            f'{escape(source)} -&gt; {escape(target)}: {escape(details)}</text>'
        )
        y += 24
    lines.append('</svg>')
    return '\n'.join(lines) + '\n'


__all__ = ['Layout', 'format_svg_layout', 'format_terminal_layout', 'layout_from_pipeline', 'load_layout']
