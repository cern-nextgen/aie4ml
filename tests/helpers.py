from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

pytest.importorskip('onnx')

from aie4ml.frontends.onnx import from_onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402,F401  (re-exported for tests)

#: Ships with Vitis 2025.2
PART = 'xilinx_vek280_base_202610_1'


# --------------------------------------------------------------------------- #
# building QDQ models
# --------------------------------------------------------------------------- #


def qparams(prefix: str, *, frac: int = 4, unsigned: bool = False) -> list:
    """The scale/zero-point initializers a Q or DQ node needs, for a power-of-two scale."""
    zp = TensorProto.UINT8 if unsigned else TensorProto.INT8
    return [
        helper.make_tensor(f'{prefix}_scale', TensorProto.FLOAT, [], [float(2.0**-frac)]),
        helper.make_tensor(f'{prefix}_zp', zp, [], [0]),
    ]


def qdq(nodes: list, src: str, dst: str, prefix: str) -> None:
    """Append the Q->DQ pair that pins `src`'s quantization intent, producing `dst`."""
    q = f'{prefix}_q'
    nodes.append(helper.make_node('QuantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [q], name=q))
    nodes.append(
        helper.make_node('DequantizeLinear', [q, f'{prefix}_scale', f'{prefix}_zp'], [dst], name=f'{prefix}_dq')
    )


def dq(nodes: list, src: str, dst: str, prefix: str) -> None:
    """Append the DQ that lifts an int8 input into the float domain the ops declare."""
    nodes.append(
        helper.make_node('DequantizeLinear', [src, f'{prefix}_scale', f'{prefix}_zp'], [dst], name=f'{prefix}_dq')
    )


def make_model(name: str, *, nodes, inputs, outputs, initializers, opset: int = 17):
    """A checked model from built nodes. `inputs`/`outputs` are (name, dtype, shape) triples."""
    import onnx

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info(n, t, s) for n, t, s in outputs],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', opset)], ir_version=9)
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------- #
# lowering and reading back
# --------------------------------------------------------------------------- #


def lower(
    model,
    tmp_path: Path,
    directives: Optional[Dict[str, Any]] = None,
    *,
    part: str = PART,
    batch: int = 1,
    project: str = 'proj',
):
    config = {
        'Part': part,
        'AIEConfig': {'BatchSize': batch, 'Iterations': 1},
        'LayerDirectives': dict(directives or {}),
    }
    aie_model = from_onnx(model, config, output_dir=Path(tmp_path) / project, project_name=project)
    aie_model.run_pipeline()
    return aie_model.context


def config_of(ctx, node: str):
    return ctx.ir.execution.get(node).config


def memtiles(ctx) -> set:
    """Tensors buffered through a memory tile."""
    return {b.get('tensor', b.get('name')) for b in ctx.ir.physical.plan['buffers'] if b.get('tensor', b.get('name'))}


def direct_edges(ctx) -> set:
    """(producer, consumer) pairs handed over without a memtile."""
    return {(e['source'].split('.')[0], e['target'].split('.')[0]) for e in ctx.ir.physical.plan['direct_edges']}


def output_staging(ctx, node: str, port: int = 0) -> dict:
    """The descriptor a node writes with. Equality across an edge is what makes it direct."""
    inst = ctx.ir.execution.get(node)
    return inst.variant.describe_output_staging(inst.node, inst.config, inst.node.outputs[0].name, port)


# --------------------------------------------------------------------------- #
# directives  (combine with `|`)
# --------------------------------------------------------------------------- #


def parallelism(cas_num: int, *, cas_length: int = 1, contract: str | None = None) -> dict:
    cfg: Dict[str, Any] = {'cas_num': cas_num, 'cas_length': cas_length}
    if contract is not None:
        cfg['contract'] = contract
    return {'parallelism': cfg}


def microtiling(m: int, k: int, n: int) -> dict:
    """Pin a matmul's aie::mmul shape; its output is stored as (m, n)."""
    return {'microtiling': {'microtile_m': m, 'microtile_k': k, 'microtile_n': n}}


# --------------------------------------------------------------------------- #
# numerical validation (ops/) -- needs Vitis
# --------------------------------------------------------------------------- #


def assert_aie_matches_onnx(
    model,
    feeds,
    directives,
    tmp_path,
    *,
    project='proj',
    batch,
    frac=4,
    max_code_diff=5,
    part=PART,
    iterations=2,
    per_iteration=False,
):
    """The same check through aiesim rather than x86sim, for what x86 cannot see."""
    return _assert_matches_onnx(
        model,
        feeds,
        directives,
        tmp_path,
        project=project,
        batch=batch,
        frac=frac,
        max_code_diff=max_code_diff,
        part=part,
        iterations=iterations,
        simulator='aie',
        per_iteration=per_iteration,
    )


def assert_x86_matches_onnx(
    model,
    feeds,
    directives,
    tmp_path,
    *,
    project='proj',
    batch,
    frac=4,
    max_code_diff=5,
    part=PART,
    iterations=1,
    per_iteration=False,
):
    """Compile a model for x86, simulate it, and check every output against onnxruntime.

    The reference runs the float ONNX graph; each AIE output is compared in the quantized int8
    code space, tolerating a small rounding difference. One compile+sim covers every output the
    model exposes, which is why an ops/ test packs several configurations into one graph.

    Every iteration is fed the same input and checked against the same reference, so asking for
    more than one catches a kernel that carries state from one inference into the next. With
    `per_iteration`, each feed instead carries a leading axis of `iterations` distinct inputs, each
    checked against its own reference -- which is what catches framing drifting across iterations,
    since repeating one input hides it.
    """
    return _assert_matches_onnx(
        model,
        feeds,
        directives,
        tmp_path,
        project=project,
        batch=batch,
        frac=frac,
        max_code_diff=max_code_diff,
        part=part,
        iterations=iterations,
        simulator='x86',
        per_iteration=per_iteration,
    )


def _assert_matches_onnx(
    model,
    feeds,
    directives,
    tmp_path,
    *,
    project,
    batch,
    frac,
    max_code_diff,
    part,
    iterations,
    simulator,
    per_iteration,
):
    import onnxruntime as ort

    aie_model = from_onnx(
        model,
        {
            'Part': part,
            'AIEConfig': {'BatchSize': batch, 'Iterations': iterations},
            'LayerDirectives': dict(directives),
        },
        output_dir=Path(tmp_path) / project,
        project_name=project,
    )
    if simulator == 'aie':
        aie_model.build('all')
    else:
        aie_model.compile()

    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    names = [o.name for o in sess.get_outputs()]
    if per_iteration:
        runs = [sess.run(None, {k: v[i] for k, v in feeds.items()}) for i in range(iterations)]
    else:
        runs = [sess.run(None, feeds)] * iterations
    ref = {name: [run[j] for run in runs] for j, name in enumerate(names)}

    # Outputs only: per-kernel profiling slows aiesim by ~40% and no check reads it.
    got = aie_model.predict(feeds, simulator=simulator, quantize_in=False, dequantize_out=False, aie_profile=False)
    if not isinstance(got, dict):
        got = {next(iter(ref)): got}

    scale = float(2.0**-frac)
    for name, wants in ref.items():
        produced = np.asarray(got[name]).astype(np.int8)
        for iteration in range(iterations):
            want = np.clip(np.rint(np.asarray(wants[iteration], np.float32) / scale), -128, 127).astype(np.int8)
            have = produced[iteration * batch : (iteration + 1) * batch]
            diff = np.abs(have.astype(np.int16) - want.astype(np.int16))
            assert (
                int(diff.max()) <= max_code_diff
            ), f'{name}: iteration {iteration} max code diff {int(diff.max())} > {max_code_diff}'
