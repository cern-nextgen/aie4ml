from __future__ import annotations

import math
from typing import Any, ClassVar, Dict

from ....aie_types import AIEDataType, FloatIntent
from ....errors import ConfigRefused
from ....ir.graph import ExecutionInstance, OpNode, input_tensor_for_role
from ....passes.utils import sanitize_identifier
from ...base import BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PortBinding, PortMap, kernel_endpoints
from ...registry import register_variant
from ...utils import (
    MicrotileShape,
    ParallelismConfig,
    build_io_views,
    describe_partition_staging,
    extract_inner_outer,
    find_tile_split,
    inherited_microtile,
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
    layernorm_vec_size,
    pack_layernorm_param,
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


class _LayerNormVariantBase(OpImplVariant):
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

    def work(self, node: OpNode, config) -> int:
        """Multiply-accumulates' time one tile takes per call: int8 LayerNorm on an AIE-ML tile took about 300 cc per
        statistics call, one per 16 rows, beside 34 cc a row and 0.22 cc an element (4x64: 468 cc, 16x64: 1062,
        16x128: 1285), counted at its 256 int8 multiply-accumulates a cycle."""
        *rows, cols = config.io_views[input_tensor_for_role(node, 'lhs').name].tile
        rows = math.prod(rows)
        return round((300 * math.ceil(rows / 16) + rows * (34 + 0.22 * cols)) * 256)

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not layout_variant_matches(node, self.layout_name):
            return False
        if device.generation not in ('AIE', 'AIE-ML', 'AIE-MLV2'):
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

    def resolve(self, node: OpNode, device, directives, input_contracts) -> LayerNormConfig:
        io_route, parallel_cfg = parse_directives(directives)

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]

        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            'gamma': AIEDataType(format='int16', frac=GAMMA_FRAC_BITS),
            'beta': AIEDataType(format='int16', frac=BETA_FRAC_BITS),
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
            accumulator_tag=infer_accumulator_tag(device, precision['lhs'], precision['lhs'], None),
            gamma_shift=int(GAMMA_FRAC_BITS),
            out_shift=int(to_quant_intent(out_tensor.precision).frac),
            eps_q0=_resolve_eps_q0(node.name, node.metadata, int(precision['lhs'].frac)),
            rounding_mode=aie_rounding_token(precision['output']),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            io_views=io_views,
            io_route=io_route,
            layout=self.layout_name,
            microtile=microtile,
            transpose_input=io_views[in_tensor.name].is_transposed,
        )

    def validate_config(self, node: OpNode, config: LayerNormConfig, _device) -> None:
        require_power_of_two(f'{node.name}: cols', config.cols)
        if not (0 <= config.out_shift <= 15):
            raise ValueError(
                f'{node.name}: out_shift={config.out_shift} must be in [0, 15]; '
                'integer LayerNorm cannot left-shift or exceed NORM_SHIFT=15.'
            )

    def kernel_params(self, node: OpNode, config: LayerNormConfig):
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        return params

    def describe_input_staging(self, _node, config, tensor_name, port, _producer=None):
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'read',
            'outer',
        )

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_partition_staging(config.io_views[tensor_name], port, 'write', 'outer')

    def output_staging_contract(self, _node, config: LayerNormConfig, _tensor_name: str):
        return str(config.parallelism.contract)

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
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        return [
            {
                'name': 'gamma',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_gamma'],
                'dtype': p.precision['gamma'].c_type,
                'storage_dtype': p.precision['gamma'].storage_dtype,
                'filename': f'gamma_{inst_name}.h',
                'port': 'gamma',
            },
            {
                'name': 'beta',
                'kind': '1d',
                'storage': 'rom',
                'array': inst.artifacts['packed_beta'],
                'dtype': p.precision['beta'].c_type,
                'storage_dtype': p.precision['beta'].storage_dtype,
                'filename': f'beta_{inst_name}.h',
                'port': 'beta',
            },
        ]

    def footprint(self, node: OpNode, config: LayerNormConfig) -> OpImplFootprint:
        return OpImplFootprint(
            width=1,
            height=int(config.parallelism.cas_num),
        )

    def buffer_locations(self, _node: OpNode, config: LayerNormConfig, anchor_row: int):
        locations = []
        for row in range(int(config.parallelism.cas_num)):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + row, 1)
            locations.append(BufferLocation('in1', row, flow.input_col, row, (0, 3)))
            locations.append(BufferLocation('out1', row, flow.output_col, row, (0, 3)))
        return tuple(locations)

    def build_ports(self, node: OpNode, config: LayerNormConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={in_tensor.name: PortBinding('in1', n, endpoints=kernel_endpoints(n, 'in[0]'))},
            outputs={node.outputs[0].name: PortBinding('out1', n, endpoints=kernel_endpoints(n, 'out[0]'))},
        )


@register_variant
class LayerNormLinearOpImplVariant(_LayerNormVariantBase):
    """LayerNorm over row-contiguous data -- the layout every non-matmul op reads and writes."""

    variant_id = 'layer_norm.i8.v1'
    layout_name = 'linear'


@register_variant
class LayerNormTiledOpImplVariant(_LayerNormVariantBase):
    """LayerNorm over microtiles, so a matmul on either side needs no memtile to reorder.

    A block holds one lane group per row, which also lets the reduction be segmented (once
    per row band) instead of a full cross-lane reduce per row.
    """

    variant_id = 'layer_norm.i8.tiled.v1'
    layout_name = 'tiled'
    supported_directives = _LayerNormVariantBase.supported_directives | {'microtiling'}
    plevel = 11
    kernel_transposes_microtile = True

    def resolve_microtile(self, node: OpNode, directives, input_contracts):
        """Match the producer's microtile so the edge is direct; else choose our own."""
        return inherited_microtile(node, input_contracts) or self.preferred_microtile(node, directives)

    def preferred_microtile(self, node: OpNode, directives) -> MicrotileShape:
        """This kernel's own microtile when nothing upstream constrains it (e.g., a graph boundary).

        A `microtiling` directive pins it (microtile_m -> row band, microtile_n -> feature block);
        otherwise 4x8.
        """
        mt = directives.get('microtiling')
        if mt is None:
            return MicrotileShape(outer=4, inner=8)
        if set(mt) != {'microtile_m', 'microtile_n'}:
            raise ValueError(f'{node.name}: microtiling takes microtile_m and microtile_n here, got {sorted(mt)}.')
        return MicrotileShape(outer=int(mt['microtile_m']), inner=int(mt['microtile_n']))

    def validate_config(self, node: OpNode, config: LayerNormConfig, device) -> None:
        super().validate_config(node, config, device)
        mt = config.microtile
        if config.rows % mt.outer:
            raise ConfigRefused(
                f'{node.name}: rows={config.rows} must be a whole number of {mt.outer}-row microtile '
                f'bands. Partitioning across kernels chose this tile height, so adjust cas_num.'
            )
        if config.cols % mt.inner:
            raise ValueError(f'{node.name}: cols={config.cols} must be a whole number of {mt.inner}-wide microtiles.')
        block_bytes = mt.outer * mt.inner * storage_bytes_for_spec(config.precision['lhs'])
        if block_bytes < 16:
            raise ValueError(
                f'{node.name}: a {mt.outer}x{mt.inner} microtile of '
                f'{config.precision["lhs"].c_type} is {block_bytes} bytes, under the 16-byte minimum '
                f'vector width. This layout needs a wider microtile or a wider dtype.'
            )
        if mt.inner < 8 or mt.outer < 2 or mt.outer > 8 or mt.outer * mt.inner > 64:
            raise ValueError(
                f'{node.name}: the tiled LayerNorm cannot normalise a {mt.outer}x{mt.inner} microtile. '
                f'It needs 2 <= outer <= 8, inner >= 8, and outer*inner <= 64, because the gamma/beta '
                f'lane group and the widened block each have to be a legal vector width. The producer '
                f"chose this tiling; retile it, or run this LayerNorm as layout='linear'."
            )
