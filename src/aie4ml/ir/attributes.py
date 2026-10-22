"""The attributes passes attach to a logical node, by name (`OpNode.attributes`): what was fused or folded into it.
Each is immutable and checks itself when made, so readers take its fields as they are."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .graph import VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS, ExecutionView, ViewPart

POOL_KINDS = frozenset({'max', 'average', 'sum'})
OUTPUT_VIEWS = frozenset({VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS})


@dataclass(frozen=True)
class FusedActivation:
    """The activation the node applies to its output; a ReLU is the one that fuses."""

    activation: str

    def __post_init__(self) -> None:
        if self.activation != 'relu':
            raise ValueError(f'only a ReLU fuses, not {self.activation!r}.')


@dataclass(frozen=True)
class FusedPool:
    """A pool of `kind` over the node's output: a window over its spatial axes, or over runs of its rows, read in
    `input_precision`."""

    kind: str
    kernel_shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dilations: Tuple[int, ...]
    pads: Tuple[int, ...]
    input_precision: Any

    def __post_init__(self) -> None:
        if self.kind not in POOL_KINDS:
            raise ValueError(f'unknown pool kind {self.kind!r}; expected one of {sorted(POOL_KINDS)}.')
        axes = len(self.kernel_shape)
        if (len(self.strides), len(self.dilations), len(self.pads)) != (axes, axes, 2 * axes):
            raise ValueError(f'a {axes}-axis pool window has strides, dilations and pads per axis, got {self}.')


@dataclass(frozen=True)
class OutputView:
    """The view of its output the node writes in place of the tensor (`OUTPUT_VIEWS`)."""

    kind: str

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_VIEWS:
            raise ValueError(f'unknown output view {self.kind!r}; expected one of {sorted(OUTPUT_VIEWS)}.')


@dataclass(frozen=True)
class IOView:
    """The axis order the node reads (`inputs`) or writes (`outputs`) each named tensor in, as folded transposes
    leave it; a tensor not named is in its logical order."""

    inputs: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    outputs: Dict[str, Tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, perm in (*self.inputs.items(), *self.outputs.items()):
            if sorted(perm) != list(range(len(perm))):
                raise ValueError(f'the io_view of {name!r}, {perm}, is not a permutation.')


@dataclass(frozen=True)
class FoldedView:
    """A slice, split or concat folded into the node's readers: each output, by name, the windows of the tensors it
    is made of along `axis`."""

    kind: str
    axis: int
    outputs: Dict[str, Tuple[ViewPart, ...]]

    def __post_init__(self) -> None:
        for name, parts in self.outputs.items():
            ExecutionView(self.kind, name, self.axis, parts)  # the view each output is, checked as execution reads it


@dataclass(frozen=True)
class Scale:
    """A constant scale folded into the node: as `output_scale` on what it writes, as `input_scale` on what it reads."""

    scale: float
