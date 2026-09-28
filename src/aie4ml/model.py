# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

from .ir import get_backend_context
from .ir.context import CONTEXT_ATTR, AIEBackendContext
from .passes import run_aie_passes
from .pipeline import DEFAULT_PIPELINE
from .simulation import (
    build_io_layout,
    collect_outputs,
    dequantize_outputs,
    prepare_inputs,
    run_simulation_target,
    write_input_files,
)
from .writer import AIEProjectEmitter

log = logging.getLogger(__name__)


class AIEModel:
    """Backend-owned runtime/project object shared by all frontends."""

    def __init__(self, ctx: AIEBackendContext, source_model: Optional[Any] = None):
        setattr(self, CONTEXT_ATTR, ctx)
        self.source_model = source_model

    @classmethod
    def from_context(cls, ctx: AIEBackendContext, source_model: Optional[Any] = None) -> 'AIEModel':
        return cls(ctx, source_model=source_model)

    @property
    def context(self) -> AIEBackendContext:
        return get_backend_context(self)

    def run_pipeline(self) -> 'AIEModel':
        self.context.emitted = None
        run_aie_passes(self.context, [cls() for cls in DEFAULT_PIPELINE])
        return self

    def write(self) -> 'AIEModel':
        self.run_pipeline()
        AIEProjectEmitter().emit(self.context)
        return self

    def build(self, make_target: str = 'all', env=None, log_to_stdout: bool = True, jobs: Optional[int] = None) -> int:
        """Run a target of the generated Makefile, writing the project first if needed.

        make_target: 'all' compiles for the hardware target; others as in the Makefile ('x86com', ...).
        env: environment for make (default: this process's).
        log_to_stdout: stream the tools' output; otherwise it is captured into the log.
        jobs: kernels compiled in parallel (make -jN); default: the AIE compiler's, 8.
        """
        ctx = self.context
        output_dir = ctx.project_config.output_dir
        if ctx.emitted != ctx.configuration_fingerprint():  # never written, or configured differently since
            self.write()

        cmd = ['make', make_target] + ([f'-j{int(jobs)}'] if jobs else [])  # -jN: kernels compiled at once
        log.debug('Running %s in %s', ' '.join(cmd), output_dir)

        stdout = None if log_to_stdout else subprocess.PIPE
        stderr = None if log_to_stdout else subprocess.STDOUT
        result = subprocess.run(cmd, cwd=output_dir, env=env, stdout=stdout, stderr=stderr, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Make target "{make_target}" failed for project "{ctx.project_config.project_name}"')
        if not log_to_stdout and result.stdout:
            log.info(result.stdout)
        return result.returncode

    def compile(self) -> int:
        return self.build(make_target='x86com')

    def predict(
        self,
        X,
        simulator: str = 'x86',
        *,
        quantize_in: bool = True,
        dequantize_out: bool = True,
        aie_profile: bool = True,
    ):
        """Simulate the compiled project on `X` and return its outputs.

        X: input samples, one array or a dict by input tensor; one sample or `Iterations` of them.
        simulator: 'x86' (functional) or 'aie' (cycle-accurate aiesim).
        quantize_in / dequantize_out: convert inputs to and outputs from the ports' fixed-point types.
        aie_profile: aiesim with per-kernel profiling, for report(); slower.
        """
        sim_key = simulator.lower()
        if sim_key == 'x86':
            make_target = 'x86sim'
        elif sim_key == 'aie':
            make_target = 'profile' if aie_profile else 'aiesim'
        else:
            raise ValueError(f'Unknown simulator "{simulator}". Expected one of: x86, aie.')

        ctx = self.context
        output_dir = ctx.project_config.output_dir
        layout = self.write_inputs(X, quantize_in=quantize_in)

        log.info('Running %s simulation using make %s', ctx.project_config.project_name, make_target)
        run_simulation_target(output_dir, make_target)
        sim_out = collect_outputs(output_dir, sim_key, layout)
        final_out = dequantize_outputs(layout, sim_out) if dequantize_out else sim_out

        def _flatten_iters(arr):
            if getattr(arr, 'ndim', 0) >= 2:
                return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
            return arr

        if len(final_out) == 1:
            return _flatten_iters(next(iter(final_out.values())))
        return {k: _flatten_iters(v) for k, v in final_out.items()}

    def write_inputs(self, X, *, quantize_in: bool = True):
        """Write `X` into the written project's simulator input files, as `predict` does; returns the I/O layout."""
        ctx = self.context
        output_dir = ctx.project_config.output_dir
        if ctx.emitted != ctx.configuration_fingerprint():
            raise RuntimeError(
                f'"{output_dir}" does not hold this model\'s project as configured now. Run write() or build() first.'
            )
        layout = build_io_layout(self)
        prepared = prepare_inputs(layout, X, iterations=int(ctx.aie_config['Iterations']), quantize=quantize_in)
        write_input_files(output_dir, layout, prepared, plio_width_bits=ctx.device.plio_width_bits)
        return layout

    def report(self):
        """Resource, latency and per-kernel cycle report for this project."""
        from .report import report

        return report(self)


def from_hls4ml(model) -> AIEModel:
    if not hasattr(model, CONTEXT_ATTR):
        from .frontends.hls4ml.lower import LowerToAieIr

        LowerToAieIr().transform(model)
    ctx = get_backend_context(model)
    return AIEModel.from_context(ctx, source_model=model)
