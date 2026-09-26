"""Pool fusion on aie4ml logical IR."""

from ..ir import TraitInstance, get_backend_context
from ..op_impls import get_family_resolver_registry
from .base import AIEPass


class FusePool(AIEPass):
    """Fuse each pool2d into its producer (the fused_pool trait). No standalone pool kernel exists, so an
    unfusable pool is refused here."""

    def __init__(self):
        self.name = 'fuse_pool'

    def transform(self, model_or_ctx):
        ctx = get_backend_context(model_or_ctx)
        graph = ctx.ir.logical
        changed = False
        for pool in list(graph.nodes):
            if pool.op_type != 'pool2d':
                continue
            source, pooled = pool.inputs[0], pool.outputs[0]
            producer = source.producer
            kind = pool.metadata['kind']
            resolver = get_family_resolver_registry().find(producer.op_type) if producer is not None else None
            if resolver is None or f'{kind}_pool' not in resolver.supported_fusions:
                raise NotImplementedError(
                    f'{pool.name}: a {kind} pool runs fused into the op producing its input, but '
                    f'{producer.op_type if producer else "the graph input"} does not fuse it.'
                )
            if len(producer.outputs) != 1 or len(source.consumers) != 1 or source.name in graph.output_tensor_names:
                raise NotImplementedError(
                    f'{pool.name}: {producer.name} writes only the pooled tensor, but {source.name!r} is also '
                    f'read elsewhere ({len(source.consumers)} consumers, or a graph output).'
                )
            if pool.directives:
                raise ValueError(
                    f'{pool.name}: a fused pool runs inside {producer.name}; direct that layer instead, not the pool.'
                )
            if source.precision != pooled.precision:
                raise NotImplementedError(
                    f'{pool.name}: a fused pool keeps its input quantization, but its output is quantized '
                    f'{pooled.precision}, not {source.precision}.'
                )
            window = {k: pool.metadata[k] for k in ('kernel_shape', 'strides', 'dilations', 'pads')}
            producer.add_trait(TraitInstance('fused_pool', {'kind': kind, **window}))
            graph.remove_node(pool, mode='contract')
            changed = True
        return changed
