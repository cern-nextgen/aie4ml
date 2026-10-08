from __future__ import annotations

import numpy as np
import pytest
from aie4ml.cost.specialization import kernel_specialization
from helpers import TensorProto, dq, helper, lower, make_model, numpy_helper, qdq, qparams

PART = 'xcvp2802-vsva5601-2MHP-e-S'
COMPILER = 'X-2025.06-TGT-260312'


def _conv(tmp_path, name, size, follow=False):
    """x -> conv 3x3, 8 -> 8 [-> conv 1x1]: tensors and nodes named after `name`."""
    nodes, inits = [], [*qparams('x'), *qparams(f'{name}w'), *qparams(f'{name}o')]
    dq(nodes, 'x_q', 'x', 'x')
    nodes.append(helper.make_node('Transpose', ['x'], [f'{name}_in'], perm=[0, 3, 1, 2]))
    nodes.append(helper.make_node('DequantizeLinear', [f'{name}w_q', f'{name}w_scale', f'{name}w_zp'], [f'{name}_w']))
    inits.append(numpy_helper.from_array(np.ones((8, 8, 3, 3), np.int8), f'{name}w_q'))
    nodes.append(helper.make_node('Conv', [f'{name}_in', f'{name}_w'], [f'{name}_c'], name=name, pads=[1, 1, 1, 1]))
    qdq(nodes, f'{name}_c', f'{name}_a', f'{name}o')
    last = f'{name}_a'
    if follow:
        inits += [*qparams('fw'), *qparams('fo'), numpy_helper.from_array(np.ones((8, 8, 1, 1), np.int8), 'fw_q')]
        nodes.append(helper.make_node('DequantizeLinear', ['fw_q', 'fw_scale', 'fw_zp'], ['f_w']))
        nodes.append(helper.make_node('Conv', [last, 'f_w'], ['f_c'], name='follow'))
        qdq(nodes, 'f_c', 'f_a', 'fo')
        last = 'f_a'
    nodes.append(helper.make_node('Transpose', [last], ['y'], perm=[0, 2, 3, 1]))
    model = make_model(
        name,
        nodes=nodes,
        inputs=[('x_q', TensorProto.INT8, [1, size, size, 8])],
        outputs=[('y', TensorProto.FLOAT, [1, size, size, 8])],
        initializers=inits,
    )
    ctx = lower(model, tmp_path, part=PART, project=name)
    return kernel_specialization(ctx.ir.execution.get(f'{name}_aie'))


def test_a_kernel_is_identified_by_what_it_is_compiled_from(tmp_path):
    first = _conv(tmp_path, 'a', 8)
    # Other names, another model around it, another placement: the same compiled kernel.
    same = _conv(tmp_path, 'b', 8, follow=True)
    wider = _conv(tmp_path, 'c', 16)
    key = first.key('single', PART, COMPILER)
    assert same.key('single', PART, COMPILER) == key
    assert wider.key('single', PART, COMPILER) != key
    assert first.key('single', PART, 'V-2024.06-TGT-250729') != key
    with pytest.raises(ValueError, match="no 'first' kernel"):
        first.key('first', PART, COMPILER)
