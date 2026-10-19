# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Pass to pack kernel artifacts into tiled layouts for AIE mmul kernels."""

from ..ir import get_backend_context
from .base import AIEPass


class PackKernelArtifacts(AIEPass):
    """
    Packs kernel-resident tensors into variant-specific tiled layouts
    required by AIE mmul-based kernels.
    """

    def __init__(self):
        self.name = 'pack_kernel_artifacts'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        changed = False

        for inst in ctx.ir.execution:
            packed = inst.variant.pack(inst)

            if packed:
                inst.artifacts.update(packed)
                changed = True

        return changed
