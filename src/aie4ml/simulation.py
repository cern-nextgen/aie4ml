# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .aie_types import FLOAT_FORMATS, AIEDataType
from .ir import get_backend_context
from .quant_utils import apply_rounding, dtype_for_precision, handle_overflow

log = logging.getLogger(__name__)


def _is_float_format(fmt: str) -> bool:
    return (fmt or '') in FLOAT_FORMATS


@dataclass
class IOPortLayout:
    direction: str
    port: int
    tensor: str
    descriptor: Dict
    staging: Dict
    dtype: AIEDataType

    @property
    def rank(self) -> int:
        return len(self.staging['io_boundary_dimension'])

    @property
    def slice_dimension(self) -> int:
        return int(self.staging['slice_dimension'])

    @property
    def io_boundary_dimension(self) -> List[int]:
        return [int(x) for x in self.staging['io_boundary_dimension']]

    @property
    def io_tiling_dimension(self) -> List[int]:
        return [int(x) for x in self.staging['io_tiling_dimension']]

    @property
    def offset(self) -> List[int]:
        # Host-visible tensor slicing must use the kernel staging descriptor.
        # Physical plan offsets may be shard-rebased for memtile units>1.
        return [int(x) for x in self.staging['offset']]

    @property
    def transfer_bytes(self) -> int:
        """Bytes one inference moves through this port: its tile, or more when the port frames the
        tile in whole transfer units. The padding follows the tile and is not part of the tensor."""
        element = int(self.dtype.width) // 8
        tile = int(np.prod(self.numpy_tile_shape)) * element
        framed = int(self.staging.get('transfer_bytes', tile))
        if framed < tile or framed % element:
            raise ValueError(
                f'{self.tensor}: a {framed}-byte transfer cannot carry a {tile}-byte tile of {element}-byte elements.'
            )
        return framed

    @property
    def logical_origin(self) -> List[int]:
        # Where this port's window starts in the tensor, signed: negative means it opens on
        # padding, before the data (a conv frame's zero border).
        return [int(x) for x in self.staging['logical_origin']]

    @property
    def tiling_dimension(self) -> List[int]:
        # The shape a PLIO file carries per iteration: the logical IO tile for a DMA-fed buffer
        # port, the whole padded port tile for a stream port. `io_tiling_dimension` bounds the
        # logical elements inside it either way.
        return [int(x) for x in self.staging['tiling_dimension']]

    @property
    def numpy_boundary_shape(self) -> Tuple[int, ...]:
        return tuple(self.io_boundary_dimension[::-1])

    @property
    def numpy_tile_shape(self) -> Tuple[int, ...]:
        return tuple(self.tiling_dimension[::-1])


@dataclass
class IOLayout:
    inputs: Dict[str, List[IOPortLayout]]
    outputs: Dict[str, List[IOPortLayout]]

    def input_tensors(self) -> List[str]:
        return list(self.inputs.keys())

    def output_tensors(self) -> List[str]:
        return list(self.outputs.keys())


def build_io_layout(model) -> IOLayout:
    """
    Build the canonical per-port layout published by the physical plan.
    """
    ctx = get_backend_context(model)
    plan = ctx.ir.physical.plan
    io_ports = plan.get('io_ports')
    if not isinstance(io_ports, list):
        raise RuntimeError('Physical plan is missing io_ports required for IO layout reconstruction.')

    inputs: Dict[str, List[IOPortLayout]] = {}
    outputs: Dict[str, List[IOPortLayout]] = {}

    for item in io_ports:
        direction = item.get('direction')
        if direction not in ('input', 'output'):
            raise RuntimeError(f'Physical plan contains invalid IO direction {direction!r}.')
        tensor = item.get('tensor')
        if not tensor:
            raise RuntimeError('Physical plan contains an IO port without a tensor name.')
        staging = item.get('staging')
        dtype = _dtype_from_plan(item.get('dtype'))
        if staging is None:
            raise RuntimeError(f'{tensor}: physical plan is missing graph-{direction} staging data.')
        if dtype is None:
            raise RuntimeError(f'{tensor}: physical plan is missing graph-{direction} dtype data.')

        port = IOPortLayout(
            direction=direction,
            port=int(item['port']),
            tensor=tensor,
            descriptor=item['descriptor'],
            staging=staging,
            dtype=dtype,
        )
        (inputs if direction == 'input' else outputs).setdefault(tensor, []).append(port)

    for t in inputs:
        inputs[t] = sorted(inputs[t], key=lambda p: p.port)
    for t in outputs:
        outputs[t] = sorted(outputs[t], key=lambda p: p.port)

    return IOLayout(inputs=inputs, outputs=outputs)


def _dtype_from_plan(data: Dict | None) -> AIEDataType | None:
    if data is None:
        return None
    return AIEDataType.from_dict(data)


def prepare_inputs(layout: IOLayout, X, iterations: int, quantize: bool = True) -> Dict[str, np.ndarray]:
    if len(layout.inputs) == 1 and not isinstance(X, dict):
        tensor = next(iter(layout.inputs.keys()))
        X = {tensor: X}

    prepared: Dict[str, np.ndarray] = {}

    for tensor, ports in layout.inputs.items():
        p0 = ports[0]
        expected = p0.numpy_boundary_shape
        arr = np.asarray(X[tensor])

        if tuple(arr.shape) == expected:
            arr = np.repeat(arr[np.newaxis, ...], iterations, axis=0)
        elif arr.ndim == len(expected) + 1 and tuple(arr.shape[1:]) == expected and arr.shape[0] == iterations:
            pass
        elif arr.ndim == len(expected) + 1 and tuple(arr.shape[1:]) == expected and arr.shape[0] == 1:
            arr = np.repeat(arr, iterations, axis=0)
        else:
            raise ValueError(
                f'{tensor}: expected shape {expected}, (1, *{expected}) or ({iterations}, *{expected}); '
                f'got {tuple(arr.shape)}'
            )

        if tuple(arr.shape[1:]) != expected:
            raise ValueError(f'{tensor}: expected input shape {expected}, got {tuple(arr.shape[1:])}')

        if _is_float_format(p0.dtype.format):
            prepared[tensor] = np.asarray(arr, dtype=np.float32)
        elif quantize:
            prepared[tensor] = _quantize_to_int(
                arr,
                dtype=p0.dtype,
            )
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f'{tensor}: quantize=False requires integer inputs')
            prepared[tensor] = arr.astype(dtype_for_precision(p0.dtype.width, p0.dtype.signed), copy=False)

    return prepared


def write_input_files(output_dir: Path, layout: IOLayout, prepared_inputs: Dict[str, np.ndarray], plio_width_bits: int):
    data_dir = Path(output_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    for tensor, ports in layout.inputs.items():
        data = prepared_inputs[tensor]
        for p in ports:
            vals_per_line = max(1, int(plio_width_bits) // int(p.dtype.width))
            values = _framed(_extract_port_tile(data, p), p)
            file_path = data_dir / f'ifm_c{p.port}.txt'
            with open(file_path, 'w') as handle:
                _write_values(handle, values, vals_per_line)


def _write_values(stream, values, vals_per_line):
    if vals_per_line <= 0:
        vals_per_line = len(values)

    is_float = np.issubdtype(np.asarray(values).dtype, np.floating)
    for idx, value in enumerate(values):
        if idx and idx % vals_per_line == 0:
            stream.write('\n')
        elif idx:
            stream.write(' ')
        stream.write(f'{float(value):e}' if is_float else str(int(value)))

    stream.write('\n')


def collect_outputs(output_dir: Path, sim_mode: str, layout: IOLayout) -> Dict[str, np.ndarray]:
    data_dir = Path(output_dir) / f'{sim_mode}simulator_output/data'
    outputs: Dict[str, np.ndarray] = {}

    for tensor, ports in layout.outputs.items():
        first = ports[0]
        first_tile = _read_output_file(data_dir / f'y_p{first.port}.txt', first)
        buf_dtype = np.float64 if _is_float_format(first.dtype.format) else np.int64
        out = np.zeros((first_tile.shape[0], *first.numpy_boundary_shape), dtype=buf_dtype)
        _insert_port_tile(out, first_tile, first)

        for p in ports[1:]:
            tile = _read_output_file(data_dir / f'y_p{p.port}.txt', p)
            _insert_port_tile(out, tile, p)

        outputs[tensor] = out

    return outputs


def dequantize_outputs(layout: IOLayout, outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for tensor, arr in outputs.items():
        p0 = layout.outputs[tensor][0]
        out[tensor] = np.ldexp(arr.astype(np.float64, copy=False), -int(p0.dtype.frac))
    return out


def _quantize_to_int(data: np.ndarray, dtype: AIEDataType) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(dtype_for_precision(dtype.width, dtype.signed), copy=False)

    scaled = np.ldexp(np.asarray(data, dtype=np.float64), int(dtype.frac))
    rounded = apply_rounding(scaled, dtype.rounding)
    integers = rounded.astype(np.int64, copy=False)
    clipped = handle_overflow(integers, int(dtype.width), bool(dtype.signed), dtype.saturation)
    return clipped.astype(dtype_for_precision(dtype.width, dtype.signed), copy=False)


def _framed(tiles: np.ndarray, port: IOPortLayout) -> np.ndarray:
    """The values a port moves, one transfer per iteration: each iteration's tile, then the zeros
    that fill it to the port's transfer size. `tiles` carries a leading iteration axis."""
    per_iteration = tiles.reshape(tiles.shape[0], -1)
    padding = port.transfer_bytes // (int(port.dtype.width) // 8) - per_iteration.shape[1]
    if padding == 0:
        return per_iteration.reshape(-1)
    zeros = np.zeros((per_iteration.shape[0], padding), dtype=per_iteration.dtype)
    return np.concatenate([per_iteration, zeros], axis=1).reshape(-1)


def _extract_port_tile(data: np.ndarray, port: IOPortLayout) -> np.ndarray:
    """The tile one port carries: its window over the tensor, clipped, the rest left zero."""
    tile = np.zeros((data.shape[0], *port.numpy_tile_shape), dtype=data.dtype)
    src, dst = _window_slices(port, data.shape[1:])
    tile[(slice(None), *dst)] = data[(slice(None), *src)]
    return tile


def _insert_port_tile(out: np.ndarray, tile: np.ndarray, port: IOPortLayout) -> None:
    """The inverse: keep the part of the tile that lands inside the tensor."""
    src, dst = _window_slices(port, out.shape[1:])
    out[(slice(None), *src)] = tile[(slice(None), *dst)]


def _window_slices(port: IOPortLayout, tensor_shape):
    """Intersect a port's window with the tensor, per axis, in numpy order.

    The window starts at `logical_origin` (signed) and spans the transfer shape; everything
    outside the tensor is padding the host leaves at zero.
    """
    tensor_slices, tile_slices = [], []
    for axis, extent in enumerate(tensor_shape):
        dim = port.rank - 1 - axis  # numpy axis -> descriptor axis (buffer order)
        start = int(port.logical_origin[dim])
        span = int(port.numpy_tile_shape[axis])
        bound = min(int(extent), int(port.numpy_boundary_shape[axis]))
        first = max(0, start)
        last = min(bound, start + span)
        take = max(0, last - first)
        tensor_slices.append(slice(first, first + take))
        tile_slices.append(slice(first - start, first - start + take))
    return tuple(tensor_slices), tuple(tile_slices)


def _read_output_file(path: Path, port: IOPortLayout) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Expected simulator output {path} not found.')

    tokens = path.read_text().split()
    clean: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.upper() == 'TLAST':
            i += 1
            continue
        if tok.upper() == 'T' and i + 2 < len(tokens):
            i += 3
            continue
        clean.append(tok)
        i += 1

    if _is_float_format(port.dtype.format):
        values = np.array([float(t) for t in clean], dtype=np.float64)
    else:
        values = np.array([int(t) for t in clean], dtype=np.int64)
    per_iter = int(np.prod(port.numpy_tile_shape, dtype=np.int64))
    iters = values.size // per_iter
    values = values[: iters * per_iter]
    return values.reshape(iters, *port.numpy_tile_shape)


def run_simulation_target(output_dir, make_target):
    cmd = ['make', make_target]
    result = subprocess.run(cmd, cwd=Path(output_dir), text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Make target "{make_target}" failed in {output_dir}.')
