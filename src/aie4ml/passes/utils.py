"""Shared helpers for AIE backend passes."""

from __future__ import annotations

from typing import Any


def sanitize_identifier(name: str, prefix: str = 'id') -> str:
    """Return a C/C++ friendly identifier derived from ``name``."""

    if not name:
        return prefix

    filtered = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name))
    filtered = filtered.lstrip('_') or '_'
    if filtered[0].isdigit():
        filtered = f'{prefix}_{filtered}'
    return filtered


def lookup_layer(model: Any, name: str):
    """Return the hls4ml layer instance with the given name, or None if missing."""
    try:
        return model.get_layer(name)
    except AttributeError:
        for layer in model.get_layers():
            if getattr(layer, 'name', None) == name:
                return layer
    return None


def assert_true_pointwise(layer) -> None:
    """A Conv1D that stands in for a Dense must really be pointwise; hls4ml's multi-dense
    replacement only ever produces that shape."""
    filt_w = layer.get_attr('filt_width')
    stride_w = layer.get_attr('stride_width')
    pad_l = layer.get_attr('pad_left')
    pad_r = layer.get_attr('pad_right')
    if not (filt_w == 1 and stride_w == 1 and pad_l == 0 and pad_r == 0):
        raise ValueError(f'{layer.name}: PointwiseConv1D is not true pointwise.')
    if layer.get_attr('in_width') != layer.get_attr('out_width'):
        raise ValueError(f'{layer.name}: PointwiseConv1D is not true pointwise.')


def is_pointwise_dense(layer) -> bool:
    """Whether a convolution layer is a Dense in disguise: a 1x1 window over every pixel.

    A Conv2D that is not pointwise is an ordinary convolution and lowers to the conv2d family.
    """
    if layer.class_name == 'Conv1D':
        assert_true_pointwise(layer)
        return True
    if layer.class_name != 'Conv2D':
        return False
    return (
        (layer.get_attr('filt_height'), layer.get_attr('filt_width')) == (1, 1)
        and (layer.get_attr('stride_height'), layer.get_attr('stride_width')) == (1, 1)
        and not any(layer.get_attr(f'pad_{side}', 0) for side in ('top', 'bottom', 'left', 'right'))
        and (layer.get_attr('in_height'), layer.get_attr('in_width'))
        == (layer.get_attr('out_height'), layer.get_attr('out_width'))
    )


def keeps_values(source, result) -> bool:
    """Whether an op that only moves or selects values (a reshape, a max pool) can carry `source`'s quantization into
    `result`: they must share a representation; how `result` would round or saturate never applies."""
    from ..aie_types import QuantIntent

    if isinstance(source, QuantIntent) and isinstance(result, QuantIntent):
        return (source.width, source.frac, source.signed) == (result.width, result.frac, result.signed)
    return source == result
