from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Sequence

STORAGE_LAYOUT_LINEAR = 'linear'
STORAGE_LAYOUT_MICROTILED = 'microtiled'
# Chunks of the inner axis with the chunk index outermost ([c/B][...][B]); the chunk size B is
# the descriptor's inner tiling_dimension. The wire order stays linear; conv frames use it.
STORAGE_LAYOUT_INNER_BLOCKED = 'inner_blocked'
STORAGE_LAYOUTS = frozenset({STORAGE_LAYOUT_LINEAR, STORAGE_LAYOUT_MICROTILED, STORAGE_LAYOUT_INNER_BLOCKED})


@dataclass(frozen=True)
class TensorView:
    """Per-tensor shape contract across three axis spaces:

      LOGICAL  host order (TensorVar shape); host-facing io_tiling/io_boundary live here.
      VIEW     kernel order = LOGICAL after `perm`; inner axis [-1], outer [-2].
               full/tile/tile_raw live here.
      BUFFER   memory order = LOGICAL reversed; BD fields (buffer_dimension,
               tiling_dimension, offset, tile_traversal) live here.

    `buffer_order` lists LOGICAL axes, so a VIEW-order shape relabels to LOGICAL before
    it applies (see `ordered_view_shape`). Identity `perm` collapses all three spaces,
    so the distinction only surfaces under transpose.

    logical             unpadded, LOGICAL order (the only non-derivable shape).
    full/tile/tile_raw  padded / per-port slice / raw slice, VIEW order.
    perm                io_view permutation (VIEW axis -> LOGICAL axis), or None.
    origin              where the logical data starts inside `full`, VIEW order; None = 0.
                        Padding before the data is a conv frame's zero border.
    """

    logical: tuple[int, ...]
    full: tuple[int, ...]
    tile: tuple[int, ...]
    tile_raw: tuple[int, ...]
    perm: tuple[int, ...] | None = None
    microtile: 'MicrotileShape | None' = None
    origin: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        rank = len(self.logical)
        for name in ('logical', 'full', 'tile', 'tile_raw'):
            shape = getattr(self, name)
            if len(shape) != rank:
                raise ValueError(f'TensorView.{name} {shape} does not have the logical rank {rank}.')
            if any(int(extent) <= 0 for extent in shape):
                raise ValueError(f'TensorView.{name} {shape} must have positive extents.')
        if self.perm is not None and sorted(self.perm) != list(range(rank)):
            raise ValueError(f'TensorView.perm {self.perm} is not a permutation of rank {rank}.')
        origin = self.origin or tuple(0 for _ in range(rank))
        if len(origin) != rank or any(int(x) < 0 for x in origin):
            raise ValueError(f'TensorView.origin {self.origin} must hold a non-negative offset per axis.')
        # A port's tile fits the padded tensor, and its unpadded extent fits the tile it describes.
        if any(int(t) > int(f) for t, f in zip(self.tile, self.full)):
            raise ValueError(f'TensorView.tile {self.tile} does not fit its padded shape {self.full}.')
        if any(int(r) > int(t) for r, t in zip(self.tile_raw, self.tile)):
            raise ValueError(f'TensorView.tile_raw {self.tile_raw} exceeds the tile {self.tile} it measures.')
        # `full`/`tile` are VIEW order and `logical` is LOGICAL order, so relabel before comparing.
        logical_in_view = _logical_to_view(self.logical, self.perm)
        if any(int(o) + int(e) > int(f) for o, e, f in zip(origin, logical_in_view, self.full)):
            raise ValueError(
                f'TensorView: the logical data {logical_in_view} at origin {origin} does not fit {self.full}.'
            )

    # Convenience accessors for the generic 2-D execution/hardware model.
    # inner = last axis [-1], outer = [-2] (1 on a rank-1 view).

    @property
    def rank(self) -> int:
        return len(self.logical)

    @property
    def is_transposed(self) -> bool:
        """True when `perm` moves the inner axis, so VIEW and BUFFER disagree on what is fastest."""
        return self.perm is not None and int(self.perm[-1]) != self.rank - 1

    @property
    def buffer_order(self) -> tuple[int, ...]:
        """LOGICAL axis per buffer position — the fixed axis-reversal convention."""
        return tuple(reversed(range(self.rank)))

    @property
    def full_inner(self) -> int:
        """Fully padded inner-dim (last axis) extent."""
        return int(self.full[-1])

    @property
    def full_outer(self) -> int:
        """Fully padded outer-dim (second-to-last axis) extent."""
        return int(self.full[-2]) if self.rank >= 2 else 1

    @property
    def compacted_full_outer(self) -> int:
        """2D kernel-contract outer extent after compacting all non-inner dims."""
        return int(math.prod(self.full[:-1])) if self.rank >= 2 else 1

    @property
    def tile_inner(self) -> int:
        """Per-port tile extent along the inner dim."""
        return int(self.tile[-1])

    @property
    def tile_outer(self) -> int:
        """Per-port tile extent along the outer dim."""
        return int(self.tile[-2]) if self.rank >= 2 else 1

    @property
    def compacted_tile_outer(self) -> int:
        """2D per-port kernel outer extent after compacting all non-inner dims."""
        return int(math.prod(self.tile[:-1])) if self.rank >= 2 else 1

    @property
    def logical_inner(self) -> int:
        """The tensor's unpadded inner-dim extent."""
        return int(_logical_to_view(self.logical, self.perm)[-1])

    @property
    def logical_outer(self) -> int:
        """The tensor's unpadded outer-dim extent."""
        return int(_logical_to_view(self.logical, self.perm)[-2]) if self.rank >= 2 else 1

    @property
    def tile_raw_inner(self) -> int:
        """Per-port unaligned tile extent along the inner dim."""
        return int(self.tile_raw[-1])

    @property
    def tile_raw_outer(self) -> int:
        """Per-port unaligned tile extent along the outer dim."""
        return int(self.tile_raw[-2]) if self.rank >= 2 else 1


def _view_to_logical(shape: Sequence[int], perm: Sequence[int] | None) -> tuple[int, ...]:
    """Relabel a VIEW-order shape (full/tile/tile_raw) to LOGICAL order."""
    if perm is None:
        return tuple(int(x) for x in shape)
    relabelled = [0 for _ in shape]
    for view_axis, logical_axis in enumerate(perm):
        relabelled[int(logical_axis)] = int(shape[view_axis])
    return tuple(relabelled)


def _logical_to_view(shape: Sequence[int], perm: Sequence[int] | None) -> tuple[int, ...]:
    """Relabel a LOGICAL-order shape to VIEW order (inverse of `_view_to_logical`)."""
    if perm is None:
        return tuple(int(x) for x in shape)
    return tuple(int(shape[int(logical_axis)]) for logical_axis in perm)


def ordered_view_shape(view: TensorView, kind: str) -> list[int]:
    """`kind` field in BUFFER order. VIEW-order kinds (full/tile/tile_raw) relabel to
    LOGICAL first, since `buffer_order` lists LOGICAL axes; 'logical' is already there.
    """
    shape = getattr(view, kind)
    if kind != 'logical':
        shape = _view_to_logical(shape, view.perm)
    return [int(shape[index]) for index in view.buffer_order]


def staging_tile_shape(desc: Mapping[str, Any]) -> tuple[int, ...]:
    if 'tiling_dimension' not in desc:
        raise ValueError('staging descriptor is missing tiling_dimension.')
    shape = [int(x) for x in desc['tiling_dimension']]
    for entry in desc.get('tile_traversal', ()):
        dim = int(entry['dimension'])
        shape[dim] = int(entry['stride']) * int(entry['wrap'])
    return tuple(shape)


def map_view_axis(view: TensorView, axis: int) -> int:
    if view.perm is None:
        return int(axis)
    return int(view.perm[axis])


def canonical_buffer_axes(view: TensorView) -> tuple[int, int, list[int]]:
    """Return (inner_dim, outer_dim, traversal_order) in buffer space.

    inner_dim       buffer axis for the last logical axis (inner / kernel-vectorized).
    outer_dim       buffer axis for the second-to-last logical axis (outer / work).
    traversal_order [inner_dim, outer_dim] + remaining axes sorted.
    """
    rank = view.rank
    inner_axis = map_view_axis(view, rank - 1)
    outer_axis = map_view_axis(view, rank - 2)
    inner_dim = view.buffer_order.index(int(inner_axis))
    outer_dim = view.buffer_order.index(int(outer_axis))
    tail_dims = sorted(dim for dim in range(rank) if dim not in (inner_dim, outer_dim))
    return inner_dim, outer_dim, [inner_dim, outer_dim] + tail_dims


def make_staging_descriptor(
    *,
    access: str,
    view: TensorView,
    tiling_dimension: Sequence[int],
    offset: Sequence[int],
    tile_traversal: Sequence[Mapping[str, int]],
    inner_dim: int,
    outer_dim: int,
    slice_dim: int | None = None,
    boundary_shape: str | None = None,
    io_boundary_shape: str | None = None,
    io_tiling_dimension: Sequence[int] | None = None,
    logical_origin: Mapping[int, int] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    # Axis space (see TensorView): BUFFER order = buffer_dimension, tiling_dimension,
    # offset, tile_traversal, inner/outer/slice_dimension (the BD walk — must agree).
    # LOGICAL order = io_tiling_dimension, io_boundary_dimension (host-facing).
    # tiling_dimension is ADF's per-BD chunk, NOT TensorView.tile.
    descriptor: Dict[str, Any] = {
        'access': access,
        'storage_layout': STORAGE_LAYOUT_MICROTILED if view.microtile is not None else STORAGE_LAYOUT_LINEAR,
        'buffer_dimension': ordered_view_shape(view, 'full'),
        'tiling_dimension': [int(x) for x in tiling_dimension],
        'offset': [int(x) for x in offset],
        'tile_traversal': [{str(k): int(v) for k, v in entry.items()} for entry in tile_traversal],
        'slice_dimension': int(inner_dim if slice_dim is None else slice_dim),
        'inner_dimension': int(inner_dim),
        'outer_dimension': int(outer_dim),
        # Where this port's window starts in the logical tensor, per axis and signed: the op that
        # partitions the tensor states it, because only it knows how its ports divide the work.
        # Negative means the window opens on padding, before the data.
        'logical_origin': [int((logical_origin or {}).get(dim, 0)) for dim in range(len(tiling_dimension))],
    }

    if boundary_shape is not None:
        descriptor['boundary_dimension'] = ordered_view_shape(view, boundary_shape)
    if io_boundary_shape is not None:
        descriptor['io_boundary_dimension'] = ordered_view_shape(view, io_boundary_shape)
    if io_tiling_dimension is not None:
        descriptor['io_tiling_dimension'] = [int(x) for x in io_tiling_dimension]
    if extras:
        descriptor.update(dict(extras))
    validate_staging_descriptor(descriptor)
    return descriptor


_STAGING_VECTORS = (
    'buffer_dimension',
    'tiling_dimension',
    'offset',
    'boundary_dimension',
    'io_boundary_dimension',
    'io_tiling_dimension',
    'logical_origin',
)


def validate_staging_descriptor(desc: Mapping[str, Any]) -> None:
    """Structural verifier for a staging descriptor, the contract every transport pass reads."""
    missing = [
        key
        for key in ('access', 'storage_layout', 'buffer_dimension', 'tiling_dimension', 'offset', 'logical_origin')
        if key not in desc
    ]
    if missing:
        raise ValueError(f'staging descriptor is missing {", ".join(missing)}.')
    if desc['access'] not in ('read', 'write'):
        raise ValueError(f'staging descriptor has unknown access {desc["access"]!r}.')
    if desc['storage_layout'] not in STORAGE_LAYOUTS:
        raise ValueError(f'staging descriptor has unknown storage_layout {desc["storage_layout"]!r}.')
    rank = len(desc['buffer_dimension'])
    for key in _STAGING_VECTORS:
        if key in desc and len(desc[key]) != rank:
            raise ValueError(f'staging descriptor {key} has rank {len(desc[key])}, expected {rank}.')
    if any(int(x) <= 0 for x in desc['buffer_dimension']):
        raise ValueError(f'staging descriptor buffer_dimension {desc["buffer_dimension"]} must be positive.')
    for dim, (chunk, extent, offset) in enumerate(
        zip(desc['tiling_dimension'], desc['buffer_dimension'], desc['offset'])
    ):
        if int(chunk) <= 0 or int(chunk) > int(extent):
            raise ValueError(f'staging descriptor tiling_dimension[{dim}]={chunk} does not fit extent {extent}.')
        if not 0 <= int(offset) < int(extent):
            raise ValueError(f'staging descriptor offset[{dim}]={offset} is outside extent {extent}.')
    for key in ('slice_dimension', 'inner_dimension', 'outer_dimension'):
        if key in desc and not 0 <= int(desc[key]) < rank:
            raise ValueError(f'staging descriptor {key}={desc[key]} is not an axis of rank {rank}.')
    # A column-phased frame groups its columns by residue: a fact of the inner-blocked layout, part of
    # what two descriptors must agree on to hand a buffer over directly.
    if 'column_phases' in desc:
        phases = desc['column_phases']
        if not isinstance(phases, int) or phases < 2:
            raise ValueError(f'staging descriptor column_phases={phases!r} must be an integer of at least 2.')
        if desc['storage_layout'] != STORAGE_LAYOUT_INNER_BLOCKED:
            raise ValueError('staging descriptor column_phases applies only to an inner-blocked layout.')
    # The BD walk: each axis is traversed at most once, and the window it sweeps stays inside the
    # buffer. An axis with no traversal entry transfers its chunk once, at its offset.
    walked = set()
    for entry in desc.get('tile_traversal', ()):
        dim, stride, wrap = int(entry['dimension']), int(entry['stride']), int(entry['wrap'])
        if not 0 <= dim < rank:
            raise ValueError(f'staging descriptor traversal dimension {dim} is not an axis of rank {rank}.')
        if dim in walked:
            raise ValueError(f'staging descriptor traverses axis {dim} more than once.')
        walked.add(dim)
        if stride <= 0 or wrap <= 0:
            raise ValueError(f'staging descriptor traversal on axis {dim} must have positive stride and wrap.')
        span = int(desc['offset'][dim]) + stride * (wrap - 1) + int(desc['tiling_dimension'][dim])
        if span > int(desc['buffer_dimension'][dim]):
            raise ValueError(
                f'staging descriptor walks {span} elements along axis {dim}, past its extent '
                f'{desc["buffer_dimension"][dim]}.'
            )
    for dim in range(rank):
        if dim in walked:
            continue
        span = int(desc['offset'][dim]) + int(desc['tiling_dimension'][dim])
        if span > int(desc['buffer_dimension'][dim]):
            raise ValueError(
                f'staging descriptor transfers {span} elements along the untraversed axis {dim}, past its '
                f'extent {desc["buffer_dimension"][dim]}.'
            )


class AxisPlan(NamedTuple):
    """DMA plan for one BUFFER axis: chunk = per-BD transfer size (tiling_dimension),
    stride/wrap = traversal, offset = port window shift."""

    chunk: int
    stride: int
    wrap: int
    offset: int = 0


def build_staging_descriptor(
    view: TensorView,
    *,
    access: str,
    plans: Mapping[int, AxisPlan],
    order: Sequence[int],
    io_tiling_base: str = 'logical',
    io_tiling_overrides: Mapping[int, int] | None = None,
    boundary_shape: str | None = None,
    io_boundary_shape: str | None = 'logical',
    slice_dim: int | None = None,
    logical_origin: Mapping[int, int] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Single BD-staging emitter: owns the BUFFER-order arrays and the LOGICAL-order
    io_tiling_dimension so no family hand-rolls the axis-space projection.

    `plans` maps a BUFFER axis -> AxisPlan; `order` lists all BUFFER axes in traversal
    order (axes absent from `plans` stream the whole extent, chunk 1). `io_tiling_base`
    seeds io_tiling_dimension; `io_tiling_overrides` set per-axis raw extents.
    """
    buffer_dimension = ordered_view_shape(view, 'full')
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)

    io_tiling_dimension = ordered_view_shape(view, io_tiling_base)
    for dim, raw in (io_tiling_overrides or {}).items():
        io_tiling_dimension[int(dim)] = int(raw)

    tiling_dimension = [1 for _ in buffer_dimension]
    offset = [0 for _ in buffer_dimension]
    tile_traversal = []
    for dim in order:
        plan = plans.get(int(dim))
        if plan is None:
            tile_traversal.append({'dimension': int(dim), 'stride': 1, 'wrap': int(buffer_dimension[dim])})
            continue
        tiling_dimension[int(dim)] = int(plan.chunk)
        offset[int(dim)] = int(plan.offset)
        tile_traversal.append({'dimension': int(dim), 'stride': int(plan.stride), 'wrap': int(plan.wrap)})

    return make_staging_descriptor(
        access=access,
        view=view,
        tiling_dimension=tiling_dimension,
        offset=offset,
        tile_traversal=tile_traversal,
        inner_dim=inner_dim,
        outer_dim=outer_dim,
        slice_dim=slice_dim,
        logical_origin=logical_origin,
        boundary_shape=boundary_shape,
        io_boundary_shape=io_boundary_shape,
        io_tiling_dimension=io_tiling_dimension,
        extras=extras,
    )


def describe_partition_staging(view, port: int, access: str, contract: str):
    """Partition pattern: split one axis across ports, stream the rest. Shared by
    row-wise/reduction ops. `contract` picks the partitioned axis:
    'inner' the kernel axis,'outer' the work axis.

    A view carrying a `microtile` refines the traversal without changing the partition: each
    axis is walked in microtile-sized chunks rather than one whole slice, which is the order
    aie::mmul reads and writes. A tensor staged that way hands over to a matmul with no memtile
    in between, since the transport layer compares descriptors.

    A transposed view is only half-realized here -- the grid is walked in view order, but each
    microtile is still stored the other way round. Resolve refuses that unless the variant
    declares `kernel_transposes_microtile`.
    """
    microtile = view.microtile
    inner_dim, outer_dim, _ = canonical_buffer_axes(view)
    is_inner = contract == 'inner'
    partition_dim = inner_dim if is_inner else outer_dim
    traverse_dim = outer_dim if is_inner else inner_dim

    part_raw = view.tile_raw_inner if is_inner else view.tile_raw_outer
    trav_raw = view.tile_raw_outer if is_inner else view.tile_raw_inner
    trav_full = view.full_outer if is_inner else view.full_inner
    traverse_wrap = max(1, int(trav_full) // max(1, int(trav_raw))) if is_inner else 1

    lead, second = (inner_dim, outer_dim) if microtile is not None else (partition_dim, traverse_dim)
    order = [lead, second] + [d for d in range(view.rank) if d not in (lead, second)]

    return build_staging_descriptor(
        view,
        access=access,
        plans=(
            {
                partition_dim: AxisPlan(int(part_raw), int(part_raw), 1, int(port) * int(part_raw)),
                traverse_dim: AxisPlan(int(trav_raw), int(trav_raw), int(traverse_wrap)),
            }
            if microtile is None
            else {
                inner_dim: AxisPlan(
                    int(microtile.inner),
                    int(microtile.inner),
                    max(1, int(view.tile_raw_inner) // int(microtile.inner)),
                    int(port) * int(part_raw) if is_inner else 0,
                ),
                outer_dim: AxisPlan(
                    int(microtile.outer),
                    int(microtile.outer),
                    max(1, int(view.tile_raw_outer) // int(microtile.outer)),
                    0 if is_inner else int(port) * int(part_raw),
                ),
            }
        ),
        order=order,
        io_tiling_base='tile_raw',
        slice_dim=partition_dim,
        logical_origin={partition_dim: int(port) * int(part_raw)},
        boundary_shape='logical' if access == 'read' else None,
    )


@dataclass(frozen=True)
class MicrotileShape:
    """Innermost block a tiled tensor is stored in, in the compiler's axis vocabulary:
    `outer` along the work axis (matmul's M), `inner` along the feature axis (its N)."""

    outer: int
    inner: int


def microtile_from_staging(desc: Mapping[str, Any]):
    """Decode a microtiled staging layout, or return ``None`` for linear storage.

    Not how an op reads its own layout -- that is `view.microtile`. This is the edge decoder: a
    TensorView never crosses, so a consumer only sees the producer's published staging contract.
    """
    storage_layout = desc.get('storage_layout')
    # Inner-blocked splits only the inner axis and hoists the block index above the others, so it
    # carries no (outer, inner) microtile a consumer could adopt.
    if storage_layout in (STORAGE_LAYOUT_LINEAR, STORAGE_LAYOUT_INNER_BLOCKED):
        return None
    if storage_layout not in STORAGE_LAYOUTS:
        raise ValueError(f'Unknown or missing staging storage_layout {storage_layout!r}.')
    if storage_layout != STORAGE_LAYOUT_MICROTILED:
        raise NotImplementedError(f'Cannot decode staging storage_layout {storage_layout!r} as a microtile.')

    required = ('tiling_dimension', 'buffer_dimension', 'inner_dimension', 'outer_dimension')
    missing = [field for field in required if field not in desc]
    if missing:
        raise ValueError(f'Microtiled staging descriptor is missing {", ".join(missing)}.')
    tiling = desc['tiling_dimension']
    inner = desc['inner_dimension']
    outer = desc['outer_dimension']
    return MicrotileShape(outer=int(tiling[int(outer)]), inner=int(tiling[int(inner)]))


def build_tensor_view(
    node,
    tensor,
    direction: str,
    *,
    full_inner: int,
    tile_inner: int,
    tile_inner_raw: int,
    full_outer: int,
    tile_outer: int | None = None,
    tile_outer_raw: int | None = None,
    microtile: 'MicrotileShape | None' = None,
) -> TensorView:
    """Build a rank-preserving TensorView from resolved IO layout and per-port extents.

    Only the explicit inner axis and, for rank >= 2, the explicit outer axis are
    rewritten here. Compacted 2D kernel extents are derived later from the rank-preserving view.
    """

    from .io import view_layout, view_shape  # local import avoids circular dependency

    logical = tuple(int(x) for x in tensor.shape)
    # view-order base shape (= logical under perm); seeds the padded/sliced shapes.
    view = tuple(int(x) for x in view_shape(node, tensor, direction))
    layout = view_layout(node, tensor, direction)
    rank = len(view)

    full = list(view)
    full[-1] = int(full_inner)
    if rank >= 2:
        full[-2] = int(full_outer)

    tile = list(full)
    tile[-1] = int(tile_inner)
    if tile_outer is not None and rank >= 2:
        tile[-2] = int(tile_outer)

    tile_raw = list(view)
    tile_raw[-1] = int(tile_inner_raw)
    if tile_outer_raw is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer_raw)
    elif tile_outer is not None and rank >= 2:
        tile_raw[-2] = int(tile_outer)

    return TensorView(
        logical=logical,
        full=tuple(full),
        tile=tuple(tile),
        tile_raw=tuple(tile_raw),
        perm=None if layout.get('perm') is None else tuple(int(x) for x in layout['perm']),
        microtile=microtile,
    )
