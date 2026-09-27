"""Dense: its workload features, the kernel parameters its code does not read, and how calibration exercises it."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from . import CalibrationSpace, CostDescriptor, Model

FRAC = 4


def _digest(value) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]


_BITS = {8: 'INT8', 16: 'INT16', 32: 'INT32'}


def _quant(helper, TensorProto, prefix: str, bits: int, frac: int = FRAC) -> list:
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', getattr(TensorProto, _BITS[bits]), [], [0]),
    ]


def _dense_layers(name: str, rows: int, widths, layers, point: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Dense layers L1, L2, ... from widths[0] to widths[-1] features through integer QDQ, each with the point's
    operand widths, bias and ReLU; `layers` gives each its directives."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    act, weight, out = point['act_bits'], point['weight_bits'], point['out_bits']
    inits = _quant(helper, TensorProto, 'a0', act)
    nodes = [helper.make_node('DequantizeLinear', ['x_q', 'a0_scale', 'a0_zp'], ['a0'])]
    for i, (k, n) in enumerate(zip(widths, widths[1:]), start=1):
        dtype = np.int8 if weight == 8 else np.int16
        inits += [
            numpy_helper.from_array(rng.integers(-4, 5, size=(k, n)).astype(dtype), f'w{i}_q'),
            *_quant(helper, TensorProto, f'w{i}', weight),
            *_quant(helper, TensorProto, f'a{i}', act if i < len(widths) - 1 else out),
        ]
        nodes.append(helper.make_node('DequantizeLinear', [f'w{i}_q', f'w{i}_scale', f'w{i}_zp'], [f'w{i}']))
        operands = [f'a{i - 1}', f'w{i}']
        if point['bias']:
            inits += [
                numpy_helper.from_array(rng.integers(-64, 64, size=(n,), dtype=np.int32), f'b{i}_q'),
                *_quant(helper, TensorProto, f'b{i}', 32, frac=2 * FRAC),
            ]
            nodes.append(helper.make_node('DequantizeLinear', [f'b{i}_q', f'b{i}_scale', f'b{i}_zp'], [f'b{i}']))
            operands.append(f'b{i}')
        nodes.append(helper.make_node('Gemm' if point['bias'] else 'MatMul', operands, [f'mm{i}'], name=f'L{i}'))
        result = f'mm{i}'
        if point['relu']:
            nodes.append(helper.make_node('Relu', [result], [f'mm{i}_r']))
            result = f'mm{i}_r'
        nodes.append(helper.make_node('QuantizeLinear', [result, f'a{i}_scale', f'a{i}_zp'], [f'a{i}_q']))
        last = i == len(widths) - 1
        nodes.append(
            helper.make_node('DequantizeLinear', [f'a{i}_q', f'a{i}_scale', f'a{i}_zp'], ['y' if last else f'a{i}'])
        )
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info('x_q', getattr(TensorProto, _BITS[act]), [rows, widths[0]])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [rows, widths[-1]])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 21)], ir_version=10)
    feeds = {'x_q': np.zeros((rows, widths[0]), np.int8 if act == 8 else np.int16)}
    return model, {f'L{i + 1}': directives for i, directives in enumerate(layers)}, feeds


def _dense_model(name: str, point: Dict[str, Any], length: int, variant) -> Model:
    """One Dense layer whose `length` tiles each compute `point`'s rows, k and n."""
    directives = {
        'ports': variant.port_kind,
        'parallelism': {'cas_num': 1, 'cas_length': length, 'contract': variant.contract},
    }
    model, layers, feeds = _dense_layers(name, point['rows'], (point['k'] * length, point['n']), [directives], point)
    return model, layers, feeds


def _dense_handoffs(variant, choice: Dict[str, Any]) -> Dict[str, Model]:
    """Multi-layer models of one choice whose layers hand over through shared memory and by DMA, their first layer
    split in the given (cas_num, cas_length) and every later one in the given cas_num."""
    point = dict(choice)
    designs = {  # through shared memory only, by DMA only, and both
        'handoff_96_80_64_32': ((96, 80, 64, 32), [(1, 1), (1,), (1,)]),
        'handoff_192_96_80_dma': ((192, 96, 80), [(2, 2), (2,)]),
        'handoff_192_96_80': ((192, 96, 80), [(2, 2), (1,)]),
        'handoff_240_128_48': ((240, 128, 48), [(2, 2), (1,)]),
    }
    models = {}
    for name, (widths, splits) in designs.items():
        layers = [
            {
                'ports': variant.port_kind,
                'parallelism': {'cas_num': split[0], **({'cas_length': split[1]} if len(split) > 1 else {})},
            }
            for split in splits
        ]
        models[f'{name}_{_digest(choice)}'] = _dense_layers(name, 8, widths, layers, point)
    return models


DENSE = CostDescriptor(
    features=('full_outer', 'tile_inner_lhs', 'tile_inner_rhs'),
    not_code=(
        # a run-time operand: shifts 0 to 31 compiled to one schedule in every cascade role; a fraction only sets it
        'shift',
        'precision.*.frac',
        # signed or unsigned is a scale to the array; the width (precision.*.width) sets the code
        'precision.*.format',
        # read by the graph only -- ports, placement, static_asserts -- never by a kernel's run code
        'parallelism.cas_num',
        'parallelism.cas_length',
        'full_inner_lhs',
        'full_inner_rhs',
        'tile_inner_lhs_raw',
        'tile_inner_rhs_raw',
        'bank_mem_bytes',
        'alternating_horizontal',
        # the design's tensor names and routes; the kernel's own tile is its features
        'io_views',
        'io_route',
    ),
    space=CalibrationSpace(
        choices={
            'act_bits': (8, 16),
            'weight_bits': (8, 16),
            'out_bits': (8, 16, 32),
            'bias': (False, True),
            'relu': (False, True),
        },
        shape={'rows': (4, 8), 'k': tuple(range(16, 513, 16)), 'n': tuple(range(16, 257, 16))},
        lengths=(1, 3),
        build=_dense_model,
        handoffs=_dense_handoffs,
    ),
)
