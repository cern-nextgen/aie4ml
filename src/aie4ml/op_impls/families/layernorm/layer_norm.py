from __future__ import annotations

import math
from typing import Any, ClassVar, Dict

from ....aie_types import AIEDataType, FloatIntent
from ....ir.graph import ExecutionInstance, OpNode, input_tensor_for_role
from ...base import rom_artifact
from ...registry import register_variant
from ...row_wise import RowWiseOpImplVariant, TiledRowWiseMixin
from ...utils import (
    ParallelismConfig,
    build_io_views,
    extract_inner_outer,
    find_tile_split,
    layout_variant_matches,
    parse_directives,
    require_power_of_two,
)
from ...utils.io import view_shape
from ...utils.precision import (
    aie_rounding_token,
    infer_accumulator_tag,
    resolve_exact_storage_dtype,
    storage_bytes_for_spec,
    to_quant_intent,
)
from .common import (
    BETA_FRAC_BITS,
    GAMMA_FRAC_BITS,
    TILED_MEAN_FRAC,
    layernorm_vec_size,
    pack_layernorm_param,
    tiled_fixed_point,
)
from .config import LayerNormConfig


def _resolve_eps_q0(node_name: str, metadata: dict, input_frac: int) -> int:
    if 'epsilon' in metadata:
        epsilon = float(metadata['epsilon'])
        if epsilon < 0.0:
            raise ValueError(f'{node_name}: LayerNormalization epsilon must be non-negative, got {epsilon}.')
        eps_q0_f = epsilon * float(1 << (2 * int(input_frac)))
        eps_q0 = int(round(eps_q0_f))
        if not math.isclose(eps_q0_f, float(eps_q0), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f'{node_name}: epsilon={epsilon} is not exactly representable as integer EPS_Q0 '
                f'for input frac={input_frac} (epsilon / input_scale^2 = {eps_q0_f}).'
            )
        if eps_q0 < 1:
            raise ValueError(
                f'{node_name}: epsilon={epsilon} maps to EPS_Q0={eps_q0}; '
                'integer LayerNorm requires a positive variance floor.'
            )
        return eps_q0
    eps_q0 = int(metadata.get('eps_q0', 1))
    if eps_q0 < 1:
        raise ValueError(f'{node_name}: eps_q0 must be positive, got {eps_q0}.')
    return eps_q0


class _LayerNormVariantBase(RowWiseOpImplVariant):
    """Everything the LayerNorm variants share, independent of the layout they normalise in."""

    #: The `layout:` directive name this variant answers to -- the sole selector between the
    #: two. Only a selection key: the layout the op actually writes is its staging descriptor,
    #: and the microtile it publishes for inheritance comes from output_microtile.
    layout_name: ClassVar[str]

    op_type = 'layer_norm'
    graph_header = 'layer_norm_graph.h'
    graph_name = 'layer_norm_graph'
    param_template = 'layer_norm'
    plevel = 10
    supported_directives: ClassVar[frozenset] = frozenset({'layout', 'parallelism'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not layout_variant_matches(node, self.layout_name):
            return False
        in_tensor = input_tensor_for_role(node, 'lhs')
        if isinstance(in_tensor.precision, FloatIntent):
            return False
        in_prec = resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name)
        out_prec = resolve_exact_storage_dtype(node.outputs[0].precision, namespace='output', layer_name=node.name)
        return int(in_prec.width) == 8 and bool(in_prec.signed) and int(out_prec.width) == 8 and bool(out_prec.signed)

    def resolve_microtile(self, _node: OpNode, _directives, _input_contracts):
        """The microtile this variant normalises in, or None when it works on whole rows."""
        return None

    def fixed_point(self, _node: OpNode, _eps_q0: int):
        """Fraction bits the kernel computes gamma, beta, gamma / sigma and x - mu in."""
        return GAMMA_FRAC_BITS, BETA_FRAC_BITS, 15, 0

    def resolve(self, node: OpNode, device, directives, input_contracts) -> LayerNormConfig:
        io_route, parallel_cfg = parse_directives(directives)

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]

        lhs_precision = resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name)
        eps_q0 = _resolve_eps_q0(node.name, node.metadata, int(lhs_precision.frac))
        gamma_frac, beta_frac, scale_frac, mean_frac = self.fixed_point(node, eps_q0)
        precision = {
            'lhs': lhs_precision,
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'gamma': AIEDataType(format='int16', frac=gamma_frac),
            'beta': AIEDataType(format='int16', frac=beta_frac),
        }

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        full_inner, outer_prefix, last_outer = extract_inner_outer(in_shape)

        vec_size = layernorm_vec_size(precision['lhs'], device)
        if full_inner % vec_size != 0:
            raise ValueError(
                f'{node.name}: full_inner={full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the inner dimension before LayerNorm.'
            )

        in_bpp = storage_bytes_for_spec(precision['lhs'])
        out_bpp = storage_bytes_for_spec(precision['output'])
        gamma_bpp = storage_bytes_for_spec(precision['gamma'])
        beta_bpp = storage_bytes_for_spec(precision['beta'])

        cas_num, tile_outer = find_tile_split(
            partition_size=last_outer,
            max_rows=max(1, int(device.rows) - int(device.row_start)),
            bank_bytes=int(device.bank_mem_bytes),
            tile_bytes_fn=lambda to: max(
                outer_prefix * to * full_inner * in_bpp,
                outer_prefix * to * full_inner * out_bpp,
                full_inner * gamma_bpp,
                full_inner * beta_bpp,
            ),
            parallel_cfg=parallel_cfg,
            input_contracts=input_contracts,
            primary_tensor_name=in_tensor.name,
            contract='outer',
            descending=True,
        )

        microtile = self.resolve_microtile(node, directives, input_contracts)
        io_views = build_io_views(
            node,
            [in_tensor],
            [out_tensor],
            full_inner=full_inner,
            full_outer=last_outer,
            tile_inner=full_inner,
            tile_outer=tile_outer,
            tile_inner_raw=full_inner,
            tile_outer_raw=tile_outer,
            microtile=microtile,
        )

        return LayerNormConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num), contract='outer'),
            rows=int(outer_prefix * tile_outer),
            cols=int(full_inner),
            vec_size=int(vec_size),
            accumulator_tag=infer_accumulator_tag(device, precision['lhs'], precision['lhs']),
            gamma_shift=int(gamma_frac),
            scale_frac=int(scale_frac),
            mean_frac=int(mean_frac),
            out_shift=int(to_quant_intent(out_tensor.precision).frac),
            eps_q0=eps_q0,
            rounding_mode=aie_rounding_token(precision['output']),
            alternating_horizontal=device.alternating_horizontal,
            io_views=io_views,
            io_route=io_route,
            layout=self.layout_name,
            microtile=microtile,
            transpose_input=io_views[in_tensor.name].is_transposed,
        )

    def validate_config(self, node: OpNode, config: LayerNormConfig, _device) -> None:
        require_power_of_two(f'{node.name}: cols', config.cols)
        top = config.scale_frac + config.mean_frac
        if not 0 <= config.out_shift <= top:
            raise ValueError(
                f'{node.name}: out_shift={config.out_shift} must be in [0, {top}], the fraction bits of '
                '(x - mu) * gamma / sigma; integer LayerNorm cannot left-shift its result.'
            )

    def kernel_params(self, node: OpNode, config: LayerNormConfig):
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        return params

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        p = inst.config
        gamma_tensor = input_tensor_for_role(inst.node, 'gamma')
        beta_tensor = input_tensor_for_role(inst.node, 'beta')
        cas_num = int(p.parallelism.cas_num)
        return {
            'packed_gamma': pack_layernorm_param(
                gamma_tensor.data,
                name='gamma',
                full_inner=int(p.cols),
                frac=int(p.precision['gamma'].frac),
                cas_num=cas_num,
                microtile=p.microtile,
            ),
            'packed_beta': pack_layernorm_param(
                beta_tensor.data,
                name='beta',
                full_inner=int(p.cols),
                frac=int(p.precision['beta'].frac),
                cas_num=cas_num,
                microtile=p.microtile,
            ),
        }

    def get_artifacts(self, inst: ExecutionInstance):
        precision = inst.config.precision
        return [
            rom_artifact(inst, 'gamma', '1d', inst.artifacts['packed_gamma'], precision['gamma'], 'gamma'),
            rom_artifact(inst, 'beta', '1d', inst.artifacts['packed_beta'], precision['beta'], 'beta'),
        ]


@register_variant
class LayerNormLinearOpImplVariant(_LayerNormVariantBase):
    """LayerNorm over row-contiguous data -- the layout every non-matmul op reads and writes."""

    variant_id = 'layer_norm.i8.v1'
    layout_name = 'linear'

    def work(self, node: OpNode, config) -> int:
        """Multiply-accumulates' time one tile takes per call: a cost per call, per row and per element, fitted to this
        kernel on an AIE-ML tile and counted at its 256 int8 multiply-accumulates a cycle."""
        *rows, cols = config.io_views[input_tensor_for_role(node, 'lhs').name].tile
        rows = math.prod(rows)
        return round((172 + 58.5 * rows + 0.39 * rows * cols) * 256)


@register_variant
class LayerNormTiledOpImplVariant(TiledRowWiseMixin, _LayerNormVariantBase):
    """LayerNorm over microtiles, so a matmul on either side needs no memtile to reorder.

    A block holds one lane group per row, which also lets the reduction be segmented (once
    per row band) instead of a full cross-lane reduce per row.
    """

    variant_id = 'layer_norm.i8.tiled.v1'
    layout_name = 'tiled'
    supported_directives = _LayerNormVariantBase.supported_directives | {'microtiling'}
    kernel_transposes_microtile = True

    def work(self, node: OpNode, config) -> int:
        """Multiply-accumulates' time one tile takes per call: a cost per call, per row and per element, fitted to this
        kernel on an AIE-ML tile and counted at its 256 int8 multiply-accumulates a cycle."""
        *rows, cols = config.io_views[input_tensor_for_role(node, 'lhs').name].tile
        rows = math.prod(rows)
        return round((187 + 9.5 * rows + 0.142 * rows * cols) * 256)

    def fixed_point(self, node: OpNode, eps_q0: int):
        gamma = input_tensor_for_role(node, 'gamma').data
        beta = input_tensor_for_role(node, 'beta').data
        return (*tiled_fixed_point(gamma, beta, eps_q0), TILED_MEAN_FRAC)

    def validate_config(self, node: OpNode, config: LayerNormConfig, device) -> None:
        super().validate_config(node, config, device)
        mt = config.microtile
        self.check_whole_microtiles(node, mt, config.rows, config.cols)
        block_bytes = mt.outer * mt.inner * storage_bytes_for_spec(config.precision['lhs'])
        if block_bytes < 16:
            raise ValueError(
                f'{node.name}: a {mt.outer}x{mt.inner} microtile of '
                f'{config.precision["lhs"].c_type} is {block_bytes} bytes, under the 16-byte minimum '
                f'vector width. This layout needs a wider microtile or a wider dtype.'
            )
        if mt.inner not in (8, 16, 32) or mt.outer not in (2, 4, 8) or mt.outer * mt.inner > 64:
            raise ValueError(
                f'{node.name}: the tiled LayerNorm cannot normalise a {mt.outer}x{mt.inner} microtile. '
                f'It needs 2, 4 or 8 rows of 8, 16 or 32 features, 64 elements at most. The producer '
                f"chose this tiling; retile it, or run this LayerNorm as layout='linear'."
            )
