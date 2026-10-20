# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Generic numeric utilities for quantization arithmetic."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .aie_types import RoundingMode, SaturationMode


def dtype_for_precision(width: Optional[int], signed: bool) -> np.dtype:
    """Map a bit-width and signedness to the smallest fitting NumPy dtype."""
    if width is None:
        return np.int32

    if signed:
        if width <= 8:
            return np.int8
        if width <= 16:
            return np.int16
        if width <= 32:
            return np.int32
        return np.int64

    if width <= 8:
        return np.uint8
    if width <= 16:
        return np.uint16
    if width <= 32:
        return np.uint32
    return np.uint64


def wrap_to_width(values: np.ndarray, width: int, signed: bool) -> np.ndarray:
    """Wrap integer values to the given bit width (modular / two's-complement)."""
    modulus = 1 << width
    wrapped = np.mod(values, modulus)
    if signed:
        sign_bit = 1 << (width - 1)
        wrapped = np.where(wrapped >= sign_bit, wrapped - modulus, wrapped)
    return wrapped


def apply_rounding(values: np.ndarray, mode: RoundingMode) -> np.ndarray:
    """Apply a fixed-point rounding mode to floating-point values."""
    if mode in (RoundingMode.TRN, RoundingMode.RND_MIN_INF):
        return np.floor(values)
    if mode in (RoundingMode.TRN_ZERO, RoundingMode.RND_ZERO):
        return np.trunc(values)
    if mode == RoundingMode.RND_INF:
        return np.ceil(values)
    if mode == RoundingMode.RND_CONV:
        return np.round(values)
    if mode == RoundingMode.RND:
        return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))

    raise ValueError(f'Unsupported rounding mode {mode}')


def handle_overflow(
    values: np.ndarray,
    width: int,
    signed: bool,
    mode: SaturationMode,
) -> np.ndarray:
    """Apply an overflow/saturation mode to integer values."""
    if mode == SaturationMode.WRAP:
        return wrap_to_width(values, width, signed)

    dtype = dtype_for_precision(width, signed)
    info = np.iinfo(dtype)

    if mode == SaturationMode.SAT:
        return np.clip(values, info.min, info.max)

    if mode == SaturationMode.SAT_ZERO:
        clipped = np.clip(values, 0, info.max)
        clipped[values < 0] = 0
        return clipped

    if mode == SaturationMode.SAT_SYM:
        sym_min = -info.max if info.min < -info.max else info.min
        return np.clip(values, sym_min, info.max)

    raise ValueError(f'Unsupported saturation mode {mode}')


def quantize_to_int(
    array: np.ndarray,
    frac_bits: int,
    target_bits: int,
    signed: bool = True,
    rounding_mode=None,
    saturation_mode=None,
) -> np.ndarray:
    """Float data as fixed-point integers at `frac_bits`, rounded and saturated to `target_bits`."""
    if array is None:
        return None
    scaled = np.ldexp(np.asarray(array, dtype=np.float64), int(frac_bits))
    rounded = apply_rounding(scaled, rounding_mode)
    integers = rounded.astype(np.int64)
    processed = handle_overflow(integers, target_bits, signed, saturation_mode)
    dtype = dtype_for_precision(target_bits, signed)
    return processed.astype(dtype, copy=False)
