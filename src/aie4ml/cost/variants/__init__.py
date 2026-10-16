"""How calibration treats each op type, kept out of the variants: optional, one entry per op type.

A kernel's complete `kernel_params` are its truth. `features` names the workload fields its cost grows with; every
other field sets its code group, but those `not_code` names with evidence that the kernel's code does not read them.
A field wrongly named there shows only if calibration samples two of its values at one coordinate. `space` is how
calibration (aie4ml.cost.calibrate) exercises the op type: discrete choices, candidate shapes, a model builder
selecting a given variant, and multi-layer models for transport; lowering decides what is legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

Model = Tuple[Any, Dict[str, Any], Dict[str, Any]]  # an ONNX model, its layer directives and its simulation inputs


@dataclass(frozen=True)
class CalibrationSpace:
    choices: Dict[str, Tuple[Any, ...]]  # discrete choices a model of the op type makes
    shape: Dict[str, Tuple[int, ...]]  # candidate values of each shape axis, per tile
    lengths: Tuple[int, ...]  # chain lengths giving every cascade role
    build: Callable[[str, Dict[str, Any], int, Any], Model]  # (name, choices and shape, chain length, variant)
    handoffs: Callable[[Any, Dict[str, Any]], Dict[str, Model]]  # multi-layer models of a variant and choice, by name


@dataclass(frozen=True)
class CostDescriptor:
    features: Tuple[str, ...]
    not_code: Tuple[str, ...]  # paths into the flattened kernel parameters, `*` standing for one name
    space: CalibrationSpace


# Of every op type: every kernel saturates (its template fixes the mode) and sets its rounding mode in a control
# register, so neither sets the code.
NOT_CODE = ('rounding_mode', 'precision.*.rounding', 'precision.*.saturation')


from .dense import DENSE  # noqa: E402

DESCRIPTORS: Dict[str, CostDescriptor] = {'dense': DENSE}
