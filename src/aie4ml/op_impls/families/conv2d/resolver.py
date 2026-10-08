"""Conv2D's semantic contract: what the operation means, independent of any kernel."""

from __future__ import annotations

from ....ir.graph import VIEW_FLATTEN_2D, input_tensor_for_role
from ...family_registry import FamilyResolver, family_resolver
from ...utils import SpatialAccess2D
from ...utils.math import align_up
from .common import CHANNEL_BLOCK, fused_pool_of, spatial_access_of


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@family_resolver('conv2d')
class Conv2dFamilyResolver(FamilyResolver):
    """NHWC activations, compact `[kh, kw, Cin/groups, Cout]` weights; kernel limits live in variants."""

    op_type = 'conv2d'
    supported_fusions = frozenset({'bias', 'relu', 'max_pool'})
    supported_output_views = frozenset({VIEW_FLATTEN_2D})

    def parallelism_candidates(self, node, _device):
        """The partitions the core cuts: divisors of the input and output channel blocks, and of the output rows."""
        _, h, w, cin = (int(d) for d in input_tensor_for_role(node, 'lhs').shape)
        in_blocks = align_up(cin, CHANNEL_BLOCK) // CHANNEL_BLOCK
        out_blocks = align_up(int(input_tensor_for_role(node, 'rhs').shape[-1]), CHANNEL_BLOCK) // CHANNEL_BLOCK
        out_rows = spatial_access_of(node).output_extent(h, w)[0]
        return tuple(
            {'contract': contract, 'cas_num': cas_num, 'cas_length': cas_length}
            for contract, extent in (('inner', out_blocks), ('outer', out_rows))
            for cas_num in _divisors(extent)
            for cas_length in _divisors(in_blocks)
        )

    def fallback_kernels(self, node, _device):
        """A conv reading the graph input with few channels folds it, at a tile per split: unfolded is its fallback."""
        lhs = input_tensor_for_role(node, 'lhs')
        few = lhs.producer is None and int(lhs.shape[-1]) <= CHANNEL_BLOCK // 2
        return ({'input_fold': False},) if few and spatial_access_of(node).kernel != (1, 1) else ()

    def spatial_access(self, node) -> SpatialAccess2D:
        return spatial_access_of(node)

    def validate_structure(self, node, _device) -> None:
        lhs = input_tensor_for_role(node, 'lhs')
        rhs = input_tensor_for_role(node, 'rhs')
        out = node.outputs[0]
        if len(lhs.shape) != 4:
            raise ValueError(f'{node.name}: conv2d input must be a rank-4 NHWC activation, got {tuple(lhs.shape)}.')
        if not rhs.is_parameter or len(rhs.shape) != 4:
            raise ValueError(f'{node.name}: conv2d weights must be a constant [kh, kw, Cin/groups, Cout] tensor.')
        spatial = spatial_access_of(node)  # validates the window attributes
        groups = int(node.metadata['groups'])
        kh, kw, cin_g, cout = (int(d) for d in rhs.shape)
        batch, h, w, cin = (int(d) for d in lhs.shape)
        if spatial.kernel != (kh, kw):
            raise ValueError(f'{node.name}: kernel_shape {spatial.kernel} does not match the weights {(kh, kw)}.')
        if groups <= 0 or cin_g * groups != cin or cout % groups:
            raise ValueError(f'{node.name}: groups={groups} does not divide Cin={cin} / Cout={cout}.')
        out_h, out_w = spatial.output_extent(h, w)
        if min(out_h, out_w) < 1:
            raise ValueError(f'{node.name}: conv2d window {spatial} leaves no output for a {h}x{w} input.')
        pool = fused_pool_of(node)
        if pool is not None:
            if f'{pool.kind}_pool' not in self.supported_fusions:
                raise ValueError(f'{node.name}: conv2d cannot fuse a {pool.kind} pool.')
            out_h, out_w = pool.window.output_extent(out_h, out_w)
            if min(out_h, out_w) < 1:
                raise ValueError(f'{node.name}: the fused pool {pool.window} leaves no output.')
        view = node.traits.get('output_view')
        flatten = view is not None and view.data['kind'] == VIEW_FLATTEN_2D
        expected = (batch, out_h * out_w * cout) if flatten else (batch, out_h, out_w, cout)
        if tuple(int(d) for d in out.shape) != expected:
            raise ValueError(f'{node.name}: conv2d output {tuple(out.shape)} does not match {expected}.')
