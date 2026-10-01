from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

from ...errors import ConfigRefused
from ...ir.graph import TENSOR_LAYOUTS
from ..common_types import PORT_KIND_BUFFER
from .tensor_view import microtile_from_staging


@dataclass(frozen=True)
class ParallelismConfig:
    """Parallelism contract for any op variant.

    contract   which axis cas_num partitions (same vocabulary as output_staging_contract):
               'inner' -> the inner/feature axis; every tile keeps the full outer extent.
               'outer' -> the outer/row axis; every tile keeps the full inner extent, so a
                          tile's output stays whole and a consumer can take it directly.
    cas_num    number of kernel chains partitioning that axis.
    cas_length number of kernel columns partitioning the shared / reduction axis.
    """

    cas_num: int
    cas_length: int = 1
    contract: str = 'inner'


def parse_directives(directives) -> tuple[dict, dict]:
    """Return (io_route, parallel_cfg) from a directives dict."""
    return dict(directives.get('io_route', {})), dict(directives.get('parallelism') or {})


def extract_inner_outer(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return (full_inner, outer_prefix, last_outer) for any ND shape.

    full_inner   = shape[-1]
    last_outer   = shape[-2]  (1 when rank < 2)
    outer_prefix = prod(shape[:-2])  (1 when rank <= 2)
    """
    if not shape:
        raise ValueError('extract_inner_outer requires rank >= 1.')
    full_inner = int(shape[-1])
    last_outer = int(shape[-2]) if len(shape) >= 2 else 1
    outer_prefix = int(math.prod(shape[:-2])) if len(shape) > 2 else 1
    return full_inner, outer_prefix, last_outer


def requested_layout(node) -> str:
    """Return and validate the explicitly requested layout, or ``linear`` when omitted."""
    layout = str(node.directives.get('layout', 'linear'))
    if layout not in TENSOR_LAYOUTS:
        raise ValueError(f'{node.name}: unknown layout {layout!r}; expected one of {sorted(TENSOR_LAYOUTS)}.')
    return layout


def requested_port_kinds(node) -> Tuple[str, str]:
    """The ADF port kinds (inputs, outputs) asked of this node's activation ports (buffers when omitted)."""
    ports = node.directives.get('ports', {})
    return tuple(ports.get(direction, PORT_KIND_BUFFER) for direction in ('inputs', 'outputs'))


def layout_variant_matches(node, layout: str) -> bool:
    """Whether a layout variant may participate in selection.

    With no directive all layouts participate, allowing ``plevel`` to express the preferred
    implementation. An explicit layout remains an exact user constraint.
    """
    if 'layout' in node.directives:
        return requested_layout(node) == layout
    return True


def inherited_microtile(node, input_contracts):
    """The microtile a producer already wrote one of this node's inputs in, or None.

    The tiling counterpart of find_tile_split's partition inheritance: adopting the producer's
    shape is what makes the hand-off direct.
    """
    for tensor in node.inputs:
        tc = input_contracts.get(tensor.name)
        if tc is None or not tc.port_staging:
            continue
        microtile = microtile_from_staging(tc.port_staging[0])
        if microtile is not None:
            return microtile
    return None


def find_tile_split(
    *,
    partition_size: int,
    max_rows: int,
    bank_bytes: int,
    tile_bytes_fn: Callable[[int], int],
    parallel_cfg: dict,
    input_contracts: dict,
    primary_tensor_name: str,
    contract: str,
    descending: bool = False,
    require_match: bool = False,
) -> tuple[int, int]:
    """Find (cas_num, tile_size) that splits partition_size and fits bank_bytes.

    Preference order: user override in parallel_cfg > producer port count from
    input_contracts > auto-search. Producer preference is a soft hint for 'outer'
    (avoids memtile insertion by matching producer port count) and must be present
    for 'inner' when require_match=True.

    tile_bytes_fn(tile_size) must return the worst-case byte count across all
    kernel buffers for that tile. The caller captures outer extents and per-element
    sizes in a closure — the function receives only the tile size on the partition axis.

    descending=True searches from max parallelism downward (prefer maximum split).
    require_match=True raises immediately if no cas_num is available from user or
    producer — use for contracts where the split is dictated by the producer.
    """
    user = parallel_cfg.get('cas_num')
    if user is not None:
        requested = int(user)
    else:
        ic = input_contracts.get(primary_tensor_name)
        requested = len(ic.port_staging) if (ic is not None and ic.contract == contract) else None

    if require_match and requested is None:
        raise ConfigRefused(
            f'{contract!r} contract requires a matching producer port count; '
            'ensure the producer op is resolved before this one.'
        )

    limit = min(max_rows, partition_size)

    if requested is not None:
        candidates: range | list = [requested]
    elif descending:
        candidates = range(limit, 0, -1)
    else:
        candidates = range(1, limit + 1)

    for cas_num in candidates:
        if partition_size % cas_num != 0:
            continue
        tile_size = partition_size // cas_num
        if tile_bytes_fn(tile_size) <= bank_bytes:
            return int(cas_num), int(tile_size)

    raise ConfigRefused(
        f'No legal {contract} parallelism: partition_size={partition_size} cannot be split '
        f'into cas_num<={max_rows} where tile fits {bank_bytes}B bank.'
    )


def build_io_views(
    node,
    tensors_in: list,
    tensors_out: list,
    *,
    full_inner: int,
    full_outer: int,
    tile_inner: int,
    tile_outer: int,
    tile_inner_raw: int,
    tile_outer_raw: int,
    microtile=None,
) -> dict:
    """Build io_views for all tensors sharing the same partition geometry.

    `microtile` blocks every view it builds. Pass it when the op reads and writes the same
    tiling; a kernel that differs per tensor builds those views itself.
    """
    from .tensor_view import build_tensor_view

    views = {}
    for tensor in tensors_in:
        views[tensor.name] = build_tensor_view(
            node,
            tensor,
            'inputs',
            full_inner=full_inner,
            tile_inner=tile_inner,
            tile_inner_raw=tile_inner_raw,
            full_outer=full_outer,
            tile_outer=tile_outer,
            tile_outer_raw=tile_outer_raw,
            microtile=microtile,
        )
    for tensor in tensors_out:
        views[tensor.name] = build_tensor_view(
            node,
            tensor,
            'outputs',
            full_inner=full_inner,
            tile_inner=tile_inner,
            tile_inner_raw=tile_inner_raw,
            full_outer=full_outer,
            tile_outer=tile_outer,
            tile_outer_raw=tile_outer_raw,
            microtile=microtile,
        )
    return views
