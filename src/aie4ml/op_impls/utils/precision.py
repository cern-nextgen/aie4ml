from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from ...aie_types import FLOAT_FORMATS, AIEDataType, FloatIntent, QuantIntent, RoundingMode
from ...ir import input_role, input_tensor_for_role

ROUNDING_TOKEN_MAP: Dict[RoundingMode, str] = {
    RoundingMode.TRN: 'floor',
    RoundingMode.RND_MIN_INF: 'floor',
    RoundingMode.RND_INF: 'ceil',
    RoundingMode.RND: 'symmetric_inf',
    RoundingMode.TRN_ZERO: 'symmetric_zero',
    RoundingMode.RND_ZERO: 'symmetric_zero',
    RoundingMode.RND_CONV: 'conv_even',
}


def to_quant_intent(precision: Any) -> QuantIntent:
    if isinstance(precision, QuantIntent):
        return precision
    if isinstance(precision, AIEDataType):
        return QuantIntent(
            width=int(precision.width),
            frac=int(precision.frac),
            signed=bool(precision.signed),
            rounding=precision.rounding,
            saturation=precision.saturation,
        )
    raise TypeError(f'Unsupported precision representation {type(precision)}')


def resolve_storage_width(width: int, *, allowed: Tuple[int, ...], namespace: str, layer_name: str) -> int:
    width = int(width)
    if width <= 0:
        raise ValueError(f'{layer_name}: invalid {namespace} width {width}.')
    for candidate in allowed:
        if width <= candidate:
            return candidate
    raise ValueError(f'{layer_name}: {namespace} width {width} exceeds supported widths {allowed}.')


def resolve_storage_dtype(
    intent: QuantIntent,
    *,
    allowed: Tuple[int, ...],
    namespace: str,
    layer_name: str,
) -> AIEDataType:
    storage_width = resolve_storage_width(intent.width, allowed=allowed, namespace=namespace, layer_name=layer_name)
    return AIEDataType(
        format=f'{"int" if intent.signed else "uint"}{storage_width}',
        frac=int(intent.frac),
        rounding=intent.rounding,
        saturation=intent.saturation,
    )


def resolve_exact_storage_dtype(precision: Any, *, namespace: str, layer_name: str) -> AIEDataType:
    if isinstance(precision, FloatIntent):
        return AIEDataType(
            format=precision.format.value,
        )
    return resolve_storage_dtype(
        to_quant_intent(precision), allowed=(4, 8, 16, 32), namespace=namespace, layer_name=layer_name
    )


def infer_accumulator_tag(
    device: Any,
    lhs_dtype: Optional[AIEDataType],
    rhs_dtype: Optional[AIEDataType],
) -> Optional[str]:
    if lhs_dtype is None or rhs_dtype is None:
        return None

    if lhs_dtype.format in FLOAT_FORMATS or rhs_dtype.format in FLOAT_FORMATS:
        if lhs_dtype.format not in FLOAT_FORMATS or rhs_dtype.format not in FLOAT_FORMATS:
            raise ValueError(
                f'No accumulator tag registered for mixed float/integer precisions '
                f'({lhs_dtype.format!r}, {rhs_dtype.format!r}).'
            )
        return 'accfloat'

    lhs_w = int(getattr(lhs_dtype, 'width', 0) or 0)
    rhs_w = int(getattr(rhs_dtype, 'width', 0) or 0)
    is_ml = device.generation.startswith('AIE-ML')

    if not is_ml:
        if lhs_w <= 8 and rhs_w <= 8:
            return 'acc48'
        if lhs_w <= 16 and rhs_w <= 16:
            return 'acc48'
        if lhs_w <= 32 and rhs_w <= 32:
            return 'acc80'
        raise ValueError(
            f'No accumulator tag registered for AIE generation "{device.generation}" '
            f'with lhs {lhs_w}-bit and rhs {rhs_w}-bit precisions.'
        )

    if max(lhs_w, rhs_w) <= 8:
        return 'acc32'
    if {lhs_w, rhs_w} in ({8, 16}, {16, 8}):
        return 'acc32'
    if max(lhs_w, rhs_w) <= 16:
        return 'acc64'
    if max(lhs_w, rhs_w) <= 32:
        return 'acc64'
    raise ValueError(
        f'No accumulator tag registered for AIE generation "{device.generation}" '
        f'with lhs {lhs_w}-bit and rhs {rhs_w}-bit precisions.'
    )


def aie_rounding_token(source) -> str:
    mode = getattr(source, 'rounding_mode', None) or getattr(source, 'rounding', None) or RoundingMode.TRN
    token = ROUNDING_TOKEN_MAP.get(mode)
    if token is None:
        raise ValueError(f'Unsupported rounding mode {mode} for AIE kernel.')
    return token


def resolve_accumulator_output_shift(
    lhs_precision: Any,
    output_precision: Any,
    rhs_precision: Any = None,
) -> int:
    """Right-shift from accumulator fixed-point to output fixed-point.

    For matmul: acc_frac = lhs_frac + rhs_frac; for elementwise add: rhs_precision=None → rhs_frac=0.
    """
    lhs_frac = to_quant_intent(lhs_precision).frac
    out_frac = to_quant_intent(output_precision).frac
    rhs_frac = to_quant_intent(rhs_precision).frac if rhs_precision is not None else 0
    shift = int(lhs_frac + rhs_frac - out_frac)
    if shift < 0:
        raise ValueError(f'output frac {out_frac} exceeds the accumulator frac {lhs_frac + rhs_frac}: no left shift.')
    return shift


def element_bytes(dtype: Optional[AIEDataType]) -> int:
    if not dtype or not getattr(dtype, 'width', None):
        return 1
    return max(1, (int(dtype.width) + 7) // 8)


def storage_bytes_for_spec(spec: Any) -> int:
    return max(1, int((int(spec.width) + 7) // 8))


# --------------------------------------------------------------------------- #
# Quantized epilogue: what every lhs/rhs op resolves before it configures a kernel.
# --------------------------------------------------------------------------- #


def resolve_operand_precision(node, device) -> tuple[Dict[str, AIEDataType], str]:
    lhs_tensor = input_tensor_for_role(node, 'lhs')
    rhs_tensor = input_tensor_for_role(node, 'rhs')
    out_tensor = node.outputs[0]
    if any(t.precision is None for t in (lhs_tensor, rhs_tensor, out_tensor)):
        raise ValueError(f'{node.name}: missing precision metadata for {node.op_type}.')

    resolved = {
        'lhs': resolve_exact_storage_dtype(lhs_tensor.precision, namespace='lhs', layer_name=node.name),
        'rhs': resolve_exact_storage_dtype(rhs_tensor.precision, namespace='rhs', layer_name=node.name),
        'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
    }

    if isinstance(lhs_tensor.precision, FloatIntent):
        if not all(isinstance(t.precision, FloatIntent) for t in (lhs_tensor, rhs_tensor, out_tensor)):
            raise ValueError(f'{node.name}: float {node.op_type} requires lhs/rhs/output to share float precision.')
        return resolved, 'accfloat'

    if int(resolved['lhs'].width) <= 8 and int(resolved['rhs'].width) > 8:
        raise RuntimeError(
            f'{node.name}: unsupported int8 x int16 precision mix; its accumulator output shift '
            'may be negative, which the current kernels do not support.'
        )

    acc_tag = infer_accumulator_tag(device, resolved['lhs'], resolved['rhs'])
    return resolved, acc_tag


def resolve_bias_dtype(node, precision: Dict[str, AIEDataType]) -> AIEDataType:
    """Resolve the bias accumulator dtype for dense-family ops."""
    is_float = precision['lhs'].format in FLOAT_FORMATS
    if is_float:
        return AIEDataType(format='float32', frac=0)
    bias_tensor = next((t for t in node.inputs if t.is_parameter and input_role(node, t.name) == 'bias'), None)
    frac = int(precision['lhs'].frac) + int(precision['rhs'].frac)
    if bias_tensor is not None and bias_tensor.precision is not None:
        bias_intent = to_quant_intent(bias_tensor.precision)
        return AIEDataType(format='int32', frac=frac, rounding=bias_intent.rounding, saturation=bias_intent.saturation)
    return AIEDataType(format='int32', frac=frac)


def resolve_output_scale_shift(node, *, is_float: bool) -> int:
    trait = node.traits.get('output_scale')
    if trait is None:
        return 0
    scale = float(trait.data['scale'])
    if is_float:
        raise NotImplementedError(f'{node.name}: fused output scaling is not implemented for float MatMul-family ops.')
    if scale <= 0.0 or scale > 1.0:
        raise ValueError(f'{node.name}: fused output scale must be in the range (0, 1], got {scale}.')
    shift = int(round(-math.log2(scale)))
    if not math.isclose(scale, math.ldexp(1.0, -shift), rel_tol=0.0, abs_tol=1e-12):
        raise NotImplementedError(
            f'{node.name}: fused output scale {scale} is not a power of two; '
            'fixed-point multiplier scaling is not implemented.'
        )
    return shift
