# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Base class for AIE passes."""

from __future__ import annotations

from typing import Iterable

from ..ir import get_backend_context
from ..ir.context import AIEBackendContext


class AIEPass:
    """Base class for all AIE compiler passes.

    Subclasses set `self.name` and implement `transform(model_or_ctx)`.
    When driven by the hls4ml optimizer, `model_or_ctx` is the hls4ml model.
    When driven by `run_aie_passes`, `model_or_ctx` is a bare `AIEBackendContext`.
    In both cases, call `get_backend_context(model_or_ctx)` to obtain the context.
    Returns True if the IR was modified, False otherwise.
    """

    name: str

    def transform(self, model_or_ctx) -> bool:
        raise NotImplementedError


def run_pass(p: AIEPass, model_or_ctx) -> bool:
    """Run one pass, then verify every level of the IR it leaves."""
    changed = p.transform(model_or_ctx)
    get_backend_context(model_or_ctx).ir.verify()
    return changed


def run_aie_passes(ctx: AIEBackendContext, passes: Iterable[AIEPass]) -> None:
    """Run a sequence of AIE passes once in the declared pipeline order."""
    ctx.ir.logical.verify()
    for p in passes:
        run_pass(p, ctx)
