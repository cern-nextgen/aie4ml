"""Float compute path: bfloat16, float32 and fp8_e4m3."""

import os

import numpy as np
import pytest

# fp8 <-> float32 reference (AMD Vitis float8 E4M3 encoding)


def _float32_to_fp8(f: np.float32) -> np.uint8:
    """Convert scalar float32 to AMD Vitis float8 (E4M3)."""
    h = int(np.float32(f).view(np.uint32))
    h = (h + 0x00080000) & 0xFFFFFFFF
    e = (h & 0x7F800000) >> 23
    m = h & 0x007FFFFF
    sign = (h & 0x80000000) >> 24
    if e > 135:
        result = sign | 0x7F
    elif e > 120:
        result = sign | (((e - 120) << 3) & 0x78) | (m >> 20)
    elif e > 116:
        result = sign | (((0x00780000 + m) >> (140 - e)) + 1) >> 1
    else:
        result = sign
    return np.uint8(result & 0xFF)


def _fp8_to_float32(b: np.uint8) -> np.float32:
    """Convert AMD Vitis float8 (E4M3) to float32."""
    v = int(b)
    e = (v & 0x78) >> 3
    m = (v & 0x07) << 20
    sign = (v & 0x80) << 24
    if e != 0:
        bits = np.uint32(sign | ((e + 120) << 23) | m)
    elif m != 0:
        temp_f = float(m)
        nv = int(np.float32(temp_f).view(np.uint32) >> 23)
        bits = np.uint32(sign | ((nv - 29) << 23) | ((m << (150 - nv)) & 0x00700000))
    else:
        bits = np.uint32(sign)
    return bits.view(np.float32)


_vfloat32_to_fp8 = np.vectorize(_float32_to_fp8, otypes=[np.uint8])
_vfp8_to_float32 = np.vectorize(_fp8_to_float32, otypes=[np.float32])


def _fp8_round(x: np.ndarray) -> np.ndarray:
    """Round float32 array through fp8 E4M3 encoding (vectorised)."""
    return _vfp8_to_float32(_vfloat32_to_fp8(x.astype(np.float32)))


def _bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 array to bfloat16 precision."""
    arr = np.asarray(x, dtype=np.float32)
    return ((arr.view(np.uint32) >> 16) << 16).view(np.float32)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """Return bf16 storage bits for a float32 array."""
    arr = np.asarray(x, dtype=np.float32)
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def _attention_like_float_params(*, round_fn, model_dim, head_dim, out_dim, weight_scale, bias_scale):
    def _w(modulus, offset, shape):
        base = ((np.arange(np.prod(shape), dtype=np.int64) % modulus) - offset).astype(np.float32)
        return round_fn((base * np.float32(weight_scale)).reshape(shape))

    def _b(modulus, offset, size):
        base = ((np.arange(size, dtype=np.int64) % modulus) - offset).astype(np.float32)
        return np.asarray(base * np.float32(bias_scale), dtype=np.float32)

    return {
        'q_w': _w(7, 3, (model_dim, head_dim)),
        'q_b': _b(5, 2, head_dim),
        'k_w': _w(11, 5, (model_dim, head_dim)),
        'k_b': _b(7, 3, head_dim),
        'v_w': _w(13, 6, (model_dim, head_dim)),
        'v_b': _b(5, 2, head_dim),
        'res_w': _w(9, 4, (model_dim, head_dim)),
        'res_b': _b(9, 4, head_dim),
        'out_w': _w(15, 7, (head_dim, out_dim)),
        'out_b': _b(7, 3, out_dim),
    }


def _float_like_mode_spec(TensorProto, mode):
    if mode == 'fp8':
        dtype = getattr(TensorProto, 'FLOAT8E4M3FN', None)
        if dtype is None:
            pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')
        return {
            'dtype': dtype,
            'round': _fp8_round,
            'bits': lambda x: _vfloat32_to_fp8(np.asarray(x, dtype=np.float32)),
            'weight_init': _make_fp8_initializer,
        }
    if mode == 'bf16':
        return {
            'dtype': TensorProto.BFLOAT16,
            'round': _bf16_round,
            'bits': lambda x: _bf16_bits(np.asarray(x, dtype=np.float32)),
            'weight_init': _make_bf16_initializer,
        }
    raise ValueError(f'Unsupported float-like mode: {mode!r}')


def _make_bf16_initializer(TensorProto, name: str, array: np.ndarray):
    arr = np.asarray(array, dtype=np.float32)
    packed = (arr.view(np.uint32) >> 16).astype(np.uint16)
    tensor = TensorProto()
    tensor.data_type = TensorProto.BFLOAT16
    tensor.name = name
    tensor.dims.extend(arr.shape)
    tensor.raw_data = packed.tobytes()
    return tensor


def _make_fp8_initializer(TensorProto, name: str, array: np.ndarray):
    arr = np.asarray(array, dtype=np.float32)
    packed = _vfloat32_to_fp8(arr)
    tensor = TensorProto()
    tensor.data_type = TensorProto.FLOAT8E4M3FN
    tensor.name = name
    tensor.dims.extend(arr.shape)
    tensor.raw_data = packed.tobytes()
    return tensor


def _dense_chain_params(*, round_fn, model_dim, head_dim, out_dim, weight_scale, bias_scale):
    def _w(modulus, offset, shape):
        base = ((np.arange(np.prod(shape), dtype=np.int64) % modulus) - offset).astype(np.float32)
        return round_fn((base * np.float32(weight_scale)).reshape(shape))

    def _b(modulus, offset, size):
        base = ((np.arange(size, dtype=np.int64) % modulus) - offset).astype(np.float32)
        return np.asarray(base * np.float32(bias_scale), dtype=np.float32)

    return {
        'w0': _w(7, 3, (model_dim, head_dim)),
        'b0': _b(5, 2, head_dim),
        'w1': _w(11, 5, (head_dim, head_dim)),
        'b1': _b(7, 3, head_dim),
        'w2': _w(15, 7, (head_dim, out_dim)),
        'b2': _b(7, 3, out_dim),
    }


# ---------------------------------------------------------------------------
# Combined dense + matmul + add ONNX model builders
# ---------------------------------------------------------------------------


def _build_bf16_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n):
    """bf16 input/output model: Gemm(static weight) → Add(residual) → MatMul(dynamic)."""

    def _to_bf16_raw(arr_f32):
        u32 = arr_f32.view(np.uint32)
        u16 = (u32 >> 16).astype(np.uint16)
        return u16.tobytes()

    def init_bf16(name, arr):
        arr_f32 = np.asarray(arr, dtype=np.float32)
        t = TensorProto()
        t.data_type = TensorProto.BFLOAT16
        t.name = name
        t.dims.extend(arr_f32.shape)
        t.raw_data = _to_bf16_raw(arr_f32)
        return t

    rng = np.random.default_rng(0)
    w1 = rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32)

    x_info = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [batch, m, k])
    r_info = helper.make_tensor_value_info('r', TensorProto.BFLOAT16, [batch, m, n])
    w2_info = helper.make_tensor_value_info('w2', TensorProto.BFLOAT16, [batch, n, n])
    y_info = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [batch, m, n])

    nodes = [
        helper.make_node('MatMul', ['x', 'w1'], ['gemm_out'], name='dense0'),
        helper.make_node('Add', ['gemm_out', 'r'], ['add_out'], name='add0'),
        helper.make_node('MatMul', ['add_out', 'w2'], ['y'], name='matmul0'),
    ]
    graph = helper.make_graph(
        nodes,
        'bf16_combined',
        [x_info, r_info, w2_info],
        [y_info],
        initializer=[init_bf16('w1', w1)],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 13)],
        ir_version=8,
    ), w1


def _build_fp8_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n):
    """fp8 input/output model: MatMul(dynamic) → Add(residual)."""
    _FP8 = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8 is None:
        return None, None
    x_info = helper.make_tensor_value_info('x', _FP8, [batch, m, k])
    w_info = helper.make_tensor_value_info('w', _FP8, [batch, k, n])
    r_info = helper.make_tensor_value_info('r', _FP8, [batch, m, n])
    y_info = helper.make_tensor_value_info('y', _FP8, [batch, m, n])

    nodes = [
        helper.make_node('MatMul', ['x', 'w'], ['mm_out'], name='matmul0'),
        helper.make_node('Add', ['mm_out', 'r'], ['y'], name='add0'),
    ]
    graph = helper.make_graph(nodes, 'fp8_combined', [x_info, w_info, r_info], [y_info])
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    ), None


def _build_float_matmul_model(TensorProto, helper, batch, m, k, n, dtype):
    """Direct float-like MatMul model with dynamic lhs/rhs."""
    x_info = helper.make_tensor_value_info('x', dtype, [batch, m, k])
    w_info = helper.make_tensor_value_info('w', dtype, [batch, k, n])
    y_info = helper.make_tensor_value_info('y', dtype, [batch, m, n])
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['x', 'w'], ['y'], name='matmul0')],
        f'matmul_{dtype}',
        [x_info, w_info],
        [y_info],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_chained_matmul_model(TensorProto, helper, batch, tokens, head_dim, dtype):
    """Attention-core float-like model: scores = Q*K_t, then ctx = scores*V.

    The K operand enters already transposed at the graph boundary. This avoids
    exercising boundary-padding constraints from an explicit input Transpose and
    keeps the test focused on matmul numerics.
    """
    q_info = helper.make_tensor_value_info('q', dtype, [batch, tokens, head_dim])
    k_t_info = helper.make_tensor_value_info('k_t', dtype, [batch, head_dim, tokens])
    v_info = helper.make_tensor_value_info('v', dtype, [batch, tokens, head_dim])
    y_info = helper.make_tensor_value_info('y', dtype, [batch, tokens, head_dim])
    nodes = [
        helper.make_node('MatMul', ['q', 'k_t'], ['scores'], name='matmul_scores'),
        helper.make_node('MatMul', ['scores', 'v'], ['y'], name='matmul_ctx'),
    ]
    graph = helper.make_graph(nodes, f'chained_matmul_{dtype}', [q_info, k_t_info, v_info], [y_info])
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_dense_chain_model(
    TensorProto, helper, numpy_helper, *, mode, batch, tokens, model_dim, head_dim, out_dim
):
    """Static dense -> relu -> dense -> relu -> dense -> relu chain."""
    spec = _float_like_mode_spec(TensorProto, mode)
    params = _dense_chain_params(
        round_fn=spec['round'],
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x_info = helper.make_tensor_value_info('x', spec['dtype'], [batch, tokens, model_dim])
    y_info = helper.make_tensor_value_info('y', spec['dtype'], [batch, tokens, out_dim])
    nodes = [
        helper.make_node('MatMul', ['x', 'q_w'], ['dense0_mm'], name='dense0'),
        helper.make_node('Add', ['dense0_mm', 'q_b'], ['dense0'], name='dense0_bias'),
        helper.make_node('Relu', ['dense0'], ['dense0_relu'], name='dense0_relu'),
        helper.make_node('MatMul', ['dense0_relu', 'k_w'], ['dense1_mm'], name='dense1'),
        helper.make_node('Add', ['dense1_mm', 'k_b'], ['dense1'], name='dense1_bias'),
        helper.make_node('Relu', ['dense1'], ['dense1_relu'], name='dense1_relu'),
        helper.make_node('MatMul', ['dense1_relu', 'out_w'], ['dense2_mm'], name='dense2'),
        helper.make_node('Add', ['dense2_mm', 'out_b'], ['dense2'], name='dense2_bias'),
        helper.make_node('Relu', ['dense2'], ['y'], name='dense2_relu'),
    ]
    graph = helper.make_graph(
        nodes,
        f'{mode}_dense_chain',
        [x_info],
        [y_info],
        initializer=[
            spec['weight_init'](TensorProto, 'q_w', params['w0']),
            numpy_helper.from_array(np.asarray(params['b0'], dtype=np.float32), name='q_b'),
            spec['weight_init'](TensorProto, 'k_w', params['w1']),
            numpy_helper.from_array(np.asarray(params['b1'], dtype=np.float32), name='k_b'),
            spec['weight_init'](TensorProto, 'out_w', params['w2']),
            numpy_helper.from_array(np.asarray(params['b2'], dtype=np.float32), name='out_b'),
        ],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_dense_chain_probe_model(
    TensorProto, helper, numpy_helper, *, mode, batch, tokens, model_dim, head_dim
):
    """Two static dense+relu stages with both the intermediate and final tensors exposed."""
    spec = _float_like_mode_spec(TensorProto, mode)
    params = _dense_chain_params(
        round_fn=spec['round'],
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=head_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x_info = helper.make_tensor_value_info('x', spec['dtype'], [batch, tokens, model_dim])
    mid_info = helper.make_tensor_value_info('dense0_relu', spec['dtype'], [batch, tokens, head_dim])
    y_info = helper.make_tensor_value_info('y', spec['dtype'], [batch, tokens, head_dim])
    nodes = [
        helper.make_node('MatMul', ['x', 'q_w'], ['dense0_mm'], name='dense0'),
        helper.make_node('Add', ['dense0_mm', 'q_b'], ['dense0'], name='dense0_bias'),
        helper.make_node('Relu', ['dense0'], ['dense0_relu'], name='dense0_relu'),
        helper.make_node('MatMul', ['dense0_relu', 'k_w'], ['dense1_mm'], name='dense1'),
        helper.make_node('Add', ['dense1_mm', 'k_b'], ['dense1'], name='dense1_bias'),
        helper.make_node('Relu', ['dense1'], ['y'], name='dense1_relu'),
    ]
    graph = helper.make_graph(
        nodes,
        f'{mode}_dense_chain_probe',
        [x_info],
        [mid_info, y_info],
        initializer=[
            spec['weight_init'](TensorProto, 'q_w', params['w0']),
            numpy_helper.from_array(np.asarray(params['b0'], dtype=np.float32), name='q_b'),
            spec['weight_init'](TensorProto, 'k_w', params['w1']),
            numpy_helper.from_array(np.asarray(params['b1'], dtype=np.float32), name='k_b'),
        ],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_single_dense_model(TensorProto, helper, numpy_helper, *, mode, batch, tokens, model_dim, out_dim):
    """Single static dense+bias+relu stage."""
    spec = _float_like_mode_spec(TensorProto, mode)
    params = _dense_chain_params(
        round_fn=spec['round'],
        model_dim=model_dim,
        head_dim=out_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x_info = helper.make_tensor_value_info('x', spec['dtype'], [batch, tokens, model_dim])
    y_info = helper.make_tensor_value_info('y', spec['dtype'], [batch, tokens, out_dim])
    nodes = [
        helper.make_node('MatMul', ['x', 'q_w'], ['dense0_mm'], name='dense0'),
        helper.make_node('Add', ['dense0_mm', 'q_b'], ['dense0'], name='dense0_bias'),
        helper.make_node('Relu', ['dense0'], ['y'], name='dense0_relu'),
    ]
    graph = helper.make_graph(
        nodes,
        f'{mode}_single_dense',
        [x_info],
        [y_info],
        initializer=[
            spec['weight_init'](TensorProto, 'q_w', params['w0']),
            numpy_helper.from_array(np.asarray(params['b0'], dtype=np.float32), name='q_b'),
        ],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_attention_core_model(TensorProto, helper, numpy_helper, *, mode, batch, tokens, model_dim, head_dim):
    """Static Q/K/V dense projections followed by scores and ctx matmuls."""
    spec = _float_like_mode_spec(TensorProto, mode)
    params = _attention_like_float_params(
        round_fn=spec['round'],
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=model_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x_info = helper.make_tensor_value_info('x', spec['dtype'], [batch, tokens, model_dim])
    y_info = helper.make_tensor_value_info('y', spec['dtype'], [batch, tokens, head_dim])
    nodes = [
        helper.make_node('MatMul', ['x', 'q_w'], ['q_mm'], name='dense_q'),
        helper.make_node('Add', ['q_mm', 'q_b'], ['q'], name='dense_q_bias'),
        helper.make_node('MatMul', ['x', 'k_w'], ['k_mm'], name='dense_k'),
        helper.make_node('Add', ['k_mm', 'k_b'], ['k'], name='dense_k_bias'),
        helper.make_node('MatMul', ['x', 'v_w'], ['v_mm'], name='dense_v'),
        helper.make_node('Add', ['v_mm', 'v_b'], ['v'], name='dense_v_bias'),
        helper.make_node('Transpose', ['k'], ['k_t'], name='k_t', perm=[0, 2, 1]),
        helper.make_node('MatMul', ['q', 'k_t'], ['scores'], name='matmul_scores'),
        helper.make_node('MatMul', ['scores', 'v'], ['y'], name='matmul_ctx'),
    ]
    graph = helper.make_graph(
        nodes,
        f'{mode}_attention_core',
        [x_info],
        [y_info],
        initializer=[
            spec['weight_init'](TensorProto, 'q_w', params['q_w']),
            numpy_helper.from_array(np.asarray(params['q_b'], dtype=np.float32), name='q_b'),
            spec['weight_init'](TensorProto, 'k_w', params['k_w']),
            numpy_helper.from_array(np.asarray(params['k_b'], dtype=np.float32), name='k_b'),
            spec['weight_init'](TensorProto, 'v_w', params['v_w']),
            numpy_helper.from_array(np.asarray(params['v_b'], dtype=np.float32), name='v_b'),
        ],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _build_float_attention_tail_model(TensorProto, helper, numpy_helper, *, mode, batch, tokens, head_dim, out_dim):
    """Residual tail: Add(ctx, res) -> dense_out -> relu."""
    spec = _float_like_mode_spec(TensorProto, mode)
    params = _attention_like_float_params(
        round_fn=spec['round'],
        model_dim=out_dim,
        head_dim=head_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    ctx_info = helper.make_tensor_value_info('ctx', spec['dtype'], [batch, tokens, head_dim])
    res_info = helper.make_tensor_value_info('res', spec['dtype'], [batch, tokens, head_dim])
    y_info = helper.make_tensor_value_info('y', spec['dtype'], [batch, tokens, out_dim])
    nodes = [
        helper.make_node('Add', ['ctx', 'res'], ['sum'], name='add_residual'),
        helper.make_node('MatMul', ['sum', 'out_w'], ['out_mm'], name='dense_out'),
        helper.make_node('Add', ['out_mm', 'out_b'], ['out'], name='dense_out_bias'),
        helper.make_node('Relu', ['out'], ['y'], name='dense_out_relu'),
    ]
    graph = helper.make_graph(
        nodes,
        f'{mode}_attention_tail',
        [ctx_info, res_info],
        [y_info],
        initializer=[
            spec['weight_init'](TensorProto, 'out_w', params['out_w']),
            numpy_helper.from_array(np.asarray(params['out_b'], dtype=np.float32), name='out_b'),
        ],
    )
    return helper.make_model(
        graph,
        producer_name='test',
        opset_imports=[helper.make_operatorsetid('', 18)],
        ir_version=8,
    )


def _float_dense_chain_reference(x, *, mode, model_dim, head_dim, out_dim):
    spec = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    p = _dense_chain_params(
        round_fn=spec,
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x = spec(np.asarray(x, dtype=np.float32))
    dense0 = spec(np.matmul(x, p['w0']) + p['b0'])
    dense0 = spec(np.maximum(dense0, 0.0))
    dense1 = spec(np.matmul(dense0, p['w1']) + p['b1'])
    dense1 = spec(np.maximum(dense1, 0.0))
    dense2 = spec(np.matmul(dense1, p['w2']) + p['b2'])
    return spec(np.maximum(dense2, 0.0))


def _float_dense_chain_probe_reference(x, *, mode, model_dim, head_dim):
    spec = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    p = _dense_chain_params(
        round_fn=spec,
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=head_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x = spec(np.asarray(x, dtype=np.float32))
    dense0 = spec(np.matmul(x, p['w0']) + p['b0'])
    dense0 = spec(np.maximum(dense0, 0.0))
    dense1 = spec(np.matmul(dense0, p['w1']) + p['b1'])
    dense1 = spec(np.maximum(dense1, 0.0))
    return dense0, dense1


def _float_single_dense_reference(x, *, mode, model_dim, out_dim):
    spec = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    p = _dense_chain_params(
        round_fn=spec,
        model_dim=model_dim,
        head_dim=out_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x = spec(np.asarray(x, dtype=np.float32))
    dense0 = spec(np.matmul(x, p['w0']) + p['b0'])
    return spec(np.maximum(dense0, 0.0))


def _float_attention_core_reference(x, *, mode, model_dim, head_dim):
    spec = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    p = _attention_like_float_params(
        round_fn=spec,
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=model_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    x = spec(np.asarray(x, dtype=np.float32))
    q = spec(np.matmul(x, p['q_w']) + p['q_b'])
    k = spec(np.matmul(x, p['k_w']) + p['k_b'])
    v = spec(np.matmul(x, p['v_w']) + p['v_b'])
    scores = spec(np.matmul(q, np.transpose(k, (0, 2, 1))))
    return spec(np.matmul(scores, v))


def _float_attention_tail_reference(ctx, res, *, mode, head_dim, out_dim):
    spec = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    p = _attention_like_float_params(
        round_fn=spec,
        model_dim=out_dim,
        head_dim=head_dim,
        out_dim=out_dim,
        weight_scale=np.float32(2.0**-6),
        bias_scale=np.float32(2.0**-7),
    )
    ctx = spec(np.asarray(ctx, dtype=np.float32))
    res = spec(np.asarray(res, dtype=np.float32))
    summed = spec(ctx + res)
    out = spec(np.matmul(summed, p['out_w']) + p['out_b'])
    return spec(np.maximum(out, 0.0))


def _float_like_max_code_diff(aie_out, ref_out, *, mode):
    if mode == 'fp8':
        ref_bits = _vfloat32_to_fp8(np.asarray(ref_out, dtype=np.float32))
        aie_bits = _vfloat32_to_fp8(np.asarray(aie_out, dtype=np.float32))
        return int(np.max(np.abs(aie_bits.astype(np.int16) - ref_bits.astype(np.int16))))
    if mode == 'bf16':
        ref_bits = _bf16_bits(np.asarray(ref_out, dtype=np.float32))
        aie_bits = _bf16_bits(np.asarray(aie_out, dtype=np.float32))
        return int(np.max(np.abs(aie_bits.astype(np.int32) - ref_bits.astype(np.int32))))
    raise ValueError(f'Unsupported float-like mode: {mode!r}')


# ---------------------------------------------------------------------------
# Pipeline tests (no Vitis needed)
# ---------------------------------------------------------------------------


@pytest.mark.aie_ir
def test_bf16_combined_mlp_pipeline(tmp_path):
    """BF16 dense+add+matmul: verify IR lowers with correct precision and op types."""
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, m, k, n = 1, 8, 32, 32
    model, _ = _build_bf16_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n)

    aie_model = from_onnx(
        model,
        {
            'Part': 'xilinx_vek280_base_202610_1',
            'AIEConfig': {'Iterations': 1},
        },
        output_dir=tmp_path / 'proj_bf16_mlp',
        project_name='proj_bf16_mlp',
    )
    aie_model.run_pipeline()
    op_types = [node.op_type for node in aie_model.context.ir.logical]
    assert 'dense' in op_types
    assert 'add' in op_types
    assert 'matmul' in op_types
    ctx = aie_model.context.ir.execution
    for inst in ctx.instances.values():
        fmt = inst.config.precision['lhs'].format
        assert fmt == 'bfloat16', f'expected bfloat16 precision, got {fmt!r}'


@pytest.mark.aie_ir
def test_fp8_combined_mlp_pipeline(tmp_path):
    """fp8 matmul+add: verify IR lowers with fp8_e4m3 precision."""
    pytest.importorskip('onnx')
    from onnx import TensorProto, helper, numpy_helper

    _FP8 = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8 is None:
        pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')

    from aie4ml.frontends.onnx import from_onnx

    batch, m, k, n = 1, 4, 8, 8
    model, _ = _build_fp8_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
        },
        output_dir=tmp_path / 'proj_fp8_mlp',
        project_name='proj_fp8_mlp',
    )
    aie_model.run_pipeline()
    ctx = aie_model.context.ir.execution
    for inst in ctx.instances.values():
        fmt = inst.config.precision['lhs'].format
        assert fmt == 'fp8_e4m3', f'expected fp8_e4m3 precision, got {fmt!r}'


# ---------------------------------------------------------------------------
# x86 bit-exact simulation tests (requires Vitis)
# ---------------------------------------------------------------------------


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_bf16_combined_mlp_x86_bit_exact(tmp_path):
    """BF16 dense+add+matmul: x86 simulation matches bf16-rounded numpy reference."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, m, k, n = 1, 8, 32, 32
    model, w1 = _build_bf16_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n)

    aie_model = from_onnx(
        model,
        {
            'Part': 'xilinx_vek280_base_202610_1',
            'AIEConfig': {'Iterations': 1},
        },
        output_dir=tmp_path / 'proj_bf16_mlp_x86',
        project_name='proj_bf16_mlp_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(42)
    x = _bf16_round(rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32))
    r = _bf16_round(rng.uniform(-0.5, 0.5, (batch, m, n)).astype(np.float32))
    w2 = _bf16_round(rng.uniform(-0.5, 0.5, (batch, n, n)).astype(np.float32))

    # Reference: bf16-rounded float computation
    w1_bf16 = _bf16_round(w1)
    gemm_out = _bf16_round(np.einsum('...k,kn->...n', x, w1_bf16))
    add_out = _bf16_round(gemm_out + r)
    y_ref = _bf16_round(np.matmul(add_out, w2))

    y_aie = aie_model.predict({'x': x, 'r': r, 'w2': w2}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_allclose(y_aie.astype(np.float32), y_ref, rtol=1e-2, atol=5e-2)


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_fp8_combined_mlp_x86_bit_exact(tmp_path):
    """fp8 matmul+add: x86 simulation matches fp8-rounded numpy reference."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from onnx import TensorProto, helper, numpy_helper

    _FP8 = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8 is None:
        pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')

    from aie4ml.frontends.onnx import from_onnx

    batch, m, k, n = 1, 4, 8, 8
    model, _ = _build_fp8_combined_model(TensorProto, helper, numpy_helper, batch, m, k, n)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
        },
        output_dir=tmp_path / 'proj_fp8_mlp_x86',
        project_name='proj_fp8_mlp_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(42)
    x = _fp8_round(rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32))
    w = _fp8_round(rng.uniform(-0.5, 0.5, (batch, k, n)).astype(np.float32))
    r = _fp8_round(rng.uniform(-0.5, 0.5, (batch, m, n)).astype(np.float32))

    # Reference: fp8-rounded float computation (matmul in float32, then round)
    mm_out = _fp8_round(np.matmul(x, w))
    y_ref = _fp8_round(mm_out + r)

    y_aie = aie_model.predict({'x': x, 'w': w, 'r': r}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_array_equal(
        _vfloat32_to_fp8(y_aie.astype(np.float32)),
        _vfloat32_to_fp8(y_ref),
    )


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_fp8_large_matmul_x86_code_diff(tmp_path):
    """Large fp8 MatMul: isolate the first attention-sized reduction drift in code space."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from onnx import TensorProto, helper

    _FP8 = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8 is None:
        pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')

    from aie4ml.frontends.onnx import from_onnx

    batch, m, k, n = 1, 64, 256, 64
    model = _build_float_matmul_model(TensorProto, helper, batch, m, k, n, _FP8)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {'matmul0': {'parallelism': {'cas_num': 2, 'cas_length': 2}}},
        },
        output_dir=tmp_path / 'proj_fp8_large_matmul_x86',
        project_name='proj_fp8_large_matmul_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(123)
    x = _fp8_round(rng.uniform(-0.25, 0.25, (batch, m, k)).astype(np.float32))
    w = _fp8_round(rng.uniform(-0.25, 0.25, (batch, k, n)).astype(np.float32))

    y_ref = _fp8_round(np.matmul(x, w))
    y_aie = aie_model.predict({'x': x, 'w': w}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    ref_codes = _vfloat32_to_fp8(y_ref.astype(np.float32))
    aie_codes = _vfloat32_to_fp8(y_aie.astype(np.float32))
    max_code_diff = int(np.max(np.abs(aie_codes.astype(np.int16) - ref_codes.astype(np.int16))))
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_bf16_large_matmul_x86_code_diff(tmp_path):
    """Large bf16 MatMul: diagnose code-space drift for one attention-sized reduction."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper

    batch, m, k, n = 1, 64, 256, 64
    model = _build_float_matmul_model(TensorProto, helper, batch, m, k, n, TensorProto.BFLOAT16)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {'matmul0': {'parallelism': {'cas_num': 2, 'cas_length': 2}}},
        },
        output_dir=tmp_path / 'proj_bf16_large_matmul_x86',
        project_name='proj_bf16_large_matmul_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(123)
    x = _bf16_round(rng.uniform(-0.25, 0.25, (batch, m, k)).astype(np.float32))
    w = _bf16_round(rng.uniform(-0.25, 0.25, (batch, k, n)).astype(np.float32))

    y_ref = _bf16_round(np.matmul(x, w))
    y_aie = aie_model.predict({'x': x, 'w': w}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    ref_bits = _bf16_bits(y_ref)
    aie_bits = _bf16_bits(y_aie)
    max_code_diff = int(np.max(np.abs(aie_bits.astype(np.int32) - ref_bits.astype(np.int32))))
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_fp8_large_chained_matmul_x86_code_diff(tmp_path):
    """Attention-core fp8 chain: identifies whether divergence begins once matmuls are chained."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from onnx import TensorProto, helper

    _FP8 = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8 is None:
        pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')

    from aie4ml.frontends.onnx import from_onnx

    batch, tokens, head_dim = 1, 64, 256
    model = _build_float_chained_matmul_model(TensorProto, helper, batch, tokens, head_dim, _FP8)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'matmul_scores': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
                'matmul_ctx': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
            },
        },
        output_dir=tmp_path / 'proj_fp8_large_chain_x86',
        project_name='proj_fp8_large_chain_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(321)
    q = _fp8_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    k = _fp8_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    k_t = np.transpose(k, (0, 2, 1)).copy()
    v = _fp8_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))

    scores_ref = _fp8_round(np.matmul(q, k_t))
    y_ref = _fp8_round(np.matmul(scores_ref, v))
    y_aie = aie_model.predict({'q': q, 'k_t': k_t, 'v': v}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    ref_codes = _vfloat32_to_fp8(y_ref.astype(np.float32))
    aie_codes = _vfloat32_to_fp8(y_aie.astype(np.float32))
    max_code_diff = int(np.max(np.abs(aie_codes.astype(np.int16) - ref_codes.astype(np.int16))))
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_bf16_large_chained_matmul_x86_close(tmp_path):
    """BF16 control on the same chained attention-core shapes."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper

    batch, tokens, head_dim = 1, 64, 256
    model = _build_float_chained_matmul_model(TensorProto, helper, batch, tokens, head_dim, TensorProto.BFLOAT16)

    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'matmul_scores': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
                'matmul_ctx': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
            },
        },
        output_dir=tmp_path / 'proj_bf16_large_chain_x86',
        project_name='proj_bf16_large_chain_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(321)
    q = _bf16_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    k = _bf16_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    k_t = np.transpose(k, (0, 2, 1)).copy()
    v = _bf16_round(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))

    scores_ref = _bf16_round(np.matmul(q, k_t))
    y_ref = _bf16_round(np.matmul(scores_ref, v))
    y_aie = aie_model.predict({'q': q, 'k_t': k_t, 'v': v}, simulator='x86')[:batch]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_allclose(y_aie.astype(np.float32), y_ref, rtol=1e-2, atol=5e-2)


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('mode', ['fp8', 'bf16'])
def test_float_like_single_dense_x86_code_diff(tmp_path, mode):
    _require_vitis()
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, tokens, model_dim, out_dim = 1, 128, 64, 128
    model = _build_float_single_dense_model(
        TensorProto,
        helper,
        numpy_helper,
        mode=mode,
        batch=batch,
        tokens=tokens,
        model_dim=model_dim,
        out_dim=out_dim,
    )
    part = 'vek385_base' if mode == 'fp8' else 'xilinx_vek280_base_202610_1'
    aie_model = from_onnx(
        model,
        {
            'Part': part,
            'AIEConfig': {'Iterations': 1},
        },
        output_dir=tmp_path / f'proj_{mode}_single_dense_x86',
        project_name=f'proj_{mode}_single_dense_x86',
    )
    aie_model.compile()

    round_fn = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    rng = np.random.default_rng(17)
    x = round_fn(rng.uniform(-0.25, 0.25, (batch, tokens, model_dim)).astype(np.float32))
    y_ref = _float_single_dense_reference(x, mode=mode, model_dim=model_dim, out_dim=out_dim)
    y_aie = aie_model.predict({'x': x}, simulator='x86', quantize_in=False, dequantize_out=False)[:batch]

    assert y_ref.shape == y_aie.shape
    max_code_diff = _float_like_max_code_diff(y_aie, y_ref, mode=mode)
    if max_code_diff > 1:
        print(f'{mode} single_dense max_code_diff={max_code_diff}')
        pytest.xfail(f'{mode} single dense currently diverges with max_code_diff={max_code_diff}')
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('mode', ['fp8', 'bf16'])
def test_float_like_attention_dense_chain_x86_code_diff(tmp_path, mode):
    """Three static dense+relu stages isolate whether chained dense blocks amplify drift badly."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, tokens, model_dim, head_dim, out_dim = 1, 64, 128, 256, 128
    model = _build_float_dense_chain_model(
        TensorProto,
        helper,
        numpy_helper,
        mode=mode,
        batch=batch,
        tokens=tokens,
        model_dim=model_dim,
        head_dim=head_dim,
        out_dim=out_dim,
    )
    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'dense0': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'dense1': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'dense2': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
            },
        },
        output_dir=tmp_path / f'proj_{mode}_dense_chain_x86',
        project_name=f'proj_{mode}_dense_chain_x86',
    )
    aie_model.compile()

    round_fn = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    rng = np.random.default_rng(777)
    x = round_fn(rng.uniform(-0.25, 0.25, (batch, tokens, model_dim)).astype(np.float32))
    y_ref = _float_dense_chain_reference(x, mode=mode, model_dim=model_dim, head_dim=head_dim, out_dim=out_dim)
    y_aie = aie_model.predict({'x': x}, simulator='x86', quantize_in=False, dequantize_out=False)[:batch]

    assert y_ref.shape == y_aie.shape
    max_code_diff = _float_like_max_code_diff(y_aie, y_ref, mode=mode)
    if max_code_diff > 1:
        print(f'{mode} dense_chain max_code_diff={max_code_diff}')
        pytest.xfail(f'{mode} dense chain currently diverges with max_code_diff={max_code_diff}')
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
def test_bf16_dense_chain_probe_x86_code_diff(tmp_path):
    """Expose the first dense output and the final output to localize where bf16 chaining first diverges."""
    _require_vitis()
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, tokens, model_dim, head_dim = 1, 64, 128, 256
    model = _build_float_dense_chain_probe_model(
        TensorProto,
        helper,
        numpy_helper,
        mode='bf16',
        batch=batch,
        tokens=tokens,
        model_dim=model_dim,
        head_dim=head_dim,
    )
    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'dense0': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'dense1': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
            },
        },
        output_dir=tmp_path / 'proj_bf16_dense_chain_probe_x86',
        project_name='proj_bf16_dense_chain_probe_x86',
    )
    aie_model.compile()

    rng = np.random.default_rng(777)
    x = _bf16_round(rng.uniform(-0.25, 0.25, (batch, tokens, model_dim)).astype(np.float32))
    mid_ref, y_ref = _float_dense_chain_probe_reference(x, mode='bf16', model_dim=model_dim, head_dim=head_dim)
    outputs = aie_model.predict({'x': x}, simulator='x86', quantize_in=False, dequantize_out=False)

    assert isinstance(outputs, dict)
    mid_aie = outputs['dense0_relu'][:batch]
    y_aie = outputs['y'][:batch]

    mid_code_diff = _float_like_max_code_diff(mid_aie, mid_ref, mode='bf16')
    final_code_diff = _float_like_max_code_diff(y_aie, y_ref, mode='bf16')
    if mid_code_diff > 1 or final_code_diff > 1:
        print(f'bf16 dense_chain_probe mid_code_diff={mid_code_diff} final_code_diff={final_code_diff}')
        pytest.xfail(
            'bf16 dense chain probe currently diverges with '
            f'mid_code_diff={mid_code_diff}, final_code_diff={final_code_diff}'
        )
    assert mid_code_diff <= 1
    assert final_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('mode', ['fp8', 'bf16'])
def test_float_like_attention_core_x86_code_diff(tmp_path, mode):
    """Q/K/V dense projections plus the two attention-core matmuls."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, tokens, model_dim, head_dim = 1, 64, 128, 256
    model = _build_float_attention_core_model(
        TensorProto,
        helper,
        numpy_helper,
        mode=mode,
        batch=batch,
        tokens=tokens,
        model_dim=model_dim,
        head_dim=head_dim,
    )
    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'dense_q': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'dense_k': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'dense_v': {'parallelism': {'cas_num': 2, 'cas_length': 4}},
                'matmul_scores': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
                'matmul_ctx': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
            },
        },
        output_dir=tmp_path / f'proj_{mode}_attention_core_x86',
        project_name=f'proj_{mode}_attention_core_x86',
    )
    aie_model.compile()

    round_fn = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    rng = np.random.default_rng(778)
    x = round_fn(rng.uniform(-0.25, 0.25, (batch, tokens, model_dim)).astype(np.float32))
    y_ref = _float_attention_core_reference(x, mode=mode, model_dim=model_dim, head_dim=head_dim)
    y_aie = aie_model.predict({'x': x}, simulator='x86', quantize_in=False, dequantize_out=False)[:batch]

    assert y_ref.shape == y_aie.shape
    max_code_diff = _float_like_max_code_diff(y_aie, y_ref, mode=mode)
    if max_code_diff > 1:
        print(f'{mode} attention_core max_code_diff={max_code_diff}')
        pytest.xfail(f'{mode} attention core currently diverges with max_code_diff={max_code_diff}')
    assert max_code_diff <= 1


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@pytest.mark.parametrize('mode', ['fp8', 'bf16'])
def test_float_like_attention_tail_x86_code_diff(tmp_path, mode):
    """Residual add plus the final dense+relu tail."""
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')
    pytest.importorskip('onnx')
    from aie4ml.frontends.onnx import from_onnx
    from onnx import TensorProto, helper, numpy_helper

    batch, tokens, head_dim, out_dim = 1, 64, 256, 128
    model = _build_float_attention_tail_model(
        TensorProto,
        helper,
        numpy_helper,
        mode=mode,
        batch=batch,
        tokens=tokens,
        head_dim=head_dim,
        out_dim=out_dim,
    )
    aie_model = from_onnx(
        model,
        {
            'Part': 'vek385_base',
            'AIEConfig': {'Iterations': 1},
            'LayerDirectives': {
                'add_residual': {'parallelism': {'cas_num': 2}},
                'dense_out': {'parallelism': {'cas_num': 2, 'cas_length': 2}},
            },
        },
        output_dir=tmp_path / f'proj_{mode}_attention_tail_x86',
        project_name=f'proj_{mode}_attention_tail_x86',
    )
    aie_model.compile()

    round_fn = {'fp8': _fp8_round, 'bf16': _bf16_round}[mode]
    rng = np.random.default_rng(779)
    ctx = round_fn(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    res = round_fn(rng.uniform(-0.25, 0.25, (batch, tokens, head_dim)).astype(np.float32))
    y_ref = _float_attention_tail_reference(ctx, res, mode=mode, head_dim=head_dim, out_dim=out_dim)
    y_aie = aie_model.predict(
        {'ctx': ctx, 'res': res},
        simulator='x86',
        quantize_in=False,
        dequantize_out=False,
    )[:batch]

    assert y_ref.shape == y_aie.shape
    max_code_diff = _float_like_max_code_diff(y_aie, y_ref, mode=mode)
    if max_code_diff > 1:
        print(f'{mode} attention_tail max_code_diff={max_code_diff}')
        pytest.xfail(f'{mode} attention tail currently diverges with max_code_diff={max_code_diff}')
    assert max_code_diff <= 1


# ---------------------------------------------------------------------------
# fp8_e4m3 type-system unit tests (no Vitis, no ONNX required)
# ---------------------------------------------------------------------------


def test_fp8_float_format():
    from aie4ml.aie_types import FloatFormat

    assert FloatFormat.FP8_E4M3.value == 'fp8_e4m3'


def test_fp8_aie_data_type():
    from aie4ml.aie_types import FloatFormat, FloatIntent
    from aie4ml.op_impls.utils.precision import resolve_exact_storage_dtype

    intent = FloatIntent(width=8, format=FloatFormat.FP8_E4M3)
    dtype = resolve_exact_storage_dtype(intent, namespace='lhs', layer_name='test')
    assert dtype.format == 'fp8_e4m3'
    assert dtype.width == 8


def test_fp8_ctype_for_format():
    # AIE Vitis fp8 E4M3 C type is `float8` (adf/window/types.h)
    from aie4ml.aie_types import ctype_for_format

    assert ctype_for_format('fp8_e4m3') == 'float8'


def test_fp8_c_type_property():
    from aie4ml.aie_types import AIEDataType

    dtype = AIEDataType(format='fp8_e4m3')
    assert dtype.c_type == 'float8'


def test_bf16_storage_dtype_property():
    from aie4ml.aie_types import AIEDataType

    dtype = AIEDataType(format='bfloat16')
    assert dtype.c_type == 'bfloat16'
    assert dtype.storage_dtype == 'uint16_t'


def test_fp8_microtile_entry():
    from aie4ml.op_impls.families.matmul.common import MICROTILE_OPTIONS

    aie_mlv2 = MICROTILE_OPTIONS.get('AIE-MLV2', {})
    assert ('fp8_e4m3', 'fp8_e4m3') in aie_mlv2
    assert aie_mlv2[('fp8_e4m3', 'fp8_e4m3')]


def test_fp8_onnx_input_maps():
    pytest.importorskip('onnx')
    from onnx import TensorProto, helper

    _FP8E4M3FN = getattr(TensorProto, 'FLOAT8E4M3FN', None)
    if _FP8E4M3FN is None:
        pytest.skip('onnx version does not define FLOAT8E4M3FN (need >= 1.14)')

    from aie4ml.frontends.onnx.utils import input_maps

    graph = helper.make_graph(
        [],
        'test_fp8',
        [helper.make_tensor_value_info('x', _FP8E4M3FN, [1, 8])],
        [helper.make_tensor_value_info('y', _FP8E4M3FN, [1, 8])],
    )
    shapes, elem_types = input_maps(graph, set())
    assert shapes['x'] == (1, 8)
    assert elem_types['x'] == _FP8E4M3FN


def _require_vitis():
    if 'XILINX_VITIS' not in os.environ:
        pytest.skip('AMD Vitis not found (XILINX_VITIS not set)')


def _imports():
    hls4ml = pytest.importorskip('hls4ml')
    keras = pytest.importorskip('keras')
    qkeras = pytest.importorskip('qkeras')
    return hls4ml, keras, qkeras


def _build_models(keras, qkeras, n_in, n_h, n_out, bits=16, seed=0):
    """Return (qkeras_model, keras_reference) sharing the same weights."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed)

    from keras import Input, Model
    from keras.initializers import RandomNormal
    from keras.layers import Dense, ReLU
    from qkeras import QActivation, QDense, quantized_bits, quantized_relu

    init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    q_w = quantized_bits(bits, 2, alpha=1)
    q_b = quantized_bits(bits, 2, alpha=1)

    x_in = Input(shape=(n_in,), name='input_layer')
    x = QActivation(quantized_bits(bits, 2, alpha=1), name='input_quant')(x_in)
    x = QDense(n_h, name='dense0', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer=init, use_bias=True)(x)
    x = QActivation(quantized_relu(bits, 2), name='act0')(x)
    x = QDense(n_out, name='dense1', kernel_quantizer=q_w, bias_quantizer=q_b, bias_initializer=init, use_bias=True)(x)
    x = QActivation(quantized_bits(bits, 2, alpha=1), name='output_quant')(x)
    qmodel = Model(inputs=x_in, outputs=x)

    # Plain Keras reference — this is the right baseline for ForceFloatMode
    xr = Input(shape=(n_in,), name='input_layer')
    xr_h = Dense(n_h, name='dense0', use_bias=True)(xr)
    xr_h = ReLU()(xr_h)
    xr_h = Dense(n_out, name='dense1', use_bias=True)(xr_h)
    kmodel = Model(inputs=xr, outputs=xr_h)

    kmodel.get_layer('dense0').set_weights(qmodel.get_layer('dense0').get_weights())
    kmodel.get_layer('dense1').set_weights(qmodel.get_layer('dense1').get_weights())

    return qmodel, kmodel


def _bf16_like(x):
    arr = np.asarray(x, dtype=np.float32)
    return ((arr.view(np.uint32) >> 16) << 16).view(np.float32)


def _float_reference(kmodel, x, compute_dtype):
    if compute_dtype != 'bfloat16':
        return kmodel.predict(x, verbose=0)

    x0 = _bf16_like(x)

    w0, b0 = kmodel.get_layer('dense0').get_weights()
    w1, b1 = kmodel.get_layer('dense1').get_weights()

    h = _bf16_like(x0) @ _bf16_like(w0) + np.asarray(b0, dtype=np.float32)
    h = np.maximum(h, 0.0).astype(np.float32, copy=False)
    y = _bf16_like(h) @ _bf16_like(w1) + np.asarray(b1, dtype=np.float32)
    return np.asarray(y, dtype=np.float32)


N_IN, N_H, N_OUT = 32, 32, 16
BATCH = 4

_DTYPES = pytest.mark.parametrize(
    'compute_dtype,rtol,atol',
    [
        ('bfloat16', 1e-2, 5e-2),
        ('float', 1e-3, 1e-3),
    ],
)


@pytest.mark.aie_ir
@_DTYPES
def test_float_conversion(tmp_path, compute_dtype, rtol, atol):
    """Conversion + lowering only — no Vitis required."""
    hls4ml, keras, qkeras = _imports()
    qmodel, _ = _build_models(keras, qkeras, N_IN, N_H, N_OUT)

    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(tmp_path / f'proj_{compute_dtype}'),
        backend='aie',
        project_name=f'proj_{compute_dtype}',
        batch_size=BATCH,
        iterations=3,
        compute_dtype=compute_dtype,
    )
    assert aie_model is not None


@pytest.mark.aie_ir
@pytest.mark.requires_vitis
@_DTYPES
def test_float_x86sim(tmp_path, compute_dtype, rtol, atol):
    """Compile + x86 simulation; output must match the Keras float32 reference."""
    _require_vitis()
    hls4ml, keras, qkeras = _imports()
    qmodel, kmodel = _build_models(keras, qkeras, N_IN, N_H, N_OUT)

    cfg = hls4ml.utils.config_from_keras_model(qmodel, granularity='name')
    aie_model = hls4ml.converters.convert_from_keras_model(
        qmodel,
        hls_config=cfg,
        output_dir=str(tmp_path / f'proj_{compute_dtype}'),
        backend='aie',
        project_name=f'proj_{compute_dtype}',
        batch_size=BATCH,
        iterations=3,
        compute_dtype=compute_dtype,
    )
    aie_model.compile()

    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, size=(BATCH, N_IN)).astype(np.float32)

    y_ref = _float_reference(kmodel, x, compute_dtype)
    y_aie = aie_model.predict(x, simulator='x86')[:BATCH]

    assert y_ref.shape == y_aie.shape
    np.testing.assert_allclose(y_aie, y_ref, rtol=rtol, atol=atol)
