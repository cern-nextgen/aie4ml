from __future__ import annotations

import math
from typing import Any, ClassVar, Dict

from ....aie_types import AIEDataType, FloatIntent
from ....errors import ConfigRefused
from ....ir.graph import ExecutionInstance, OpNode, input_tensor_for_role
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
)
from ...utils.io import view_shape
from ...utils.precision import infer_accumulator_tag, resolve_exact_storage_dtype, storage_bytes_for_spec
from .common import DEFAULT_INV_SHIFT, infer_hccs_param_sets, pack_hccs_params, softmax_vec_size, validate_hccs_params
from .config import SoftmaxConfig

#: Fractional bits of the base-2 exponent in the integer exp kernel (must match EXP_ZF in the
#: parameters template).
EXP_ZF = 8
#: z = (max-x) * EXP_KQ must fit int16; the largest int8 logit gap is 255, so EXP_KQ <= 127.
EXP_KQ_MAX = 32767 // 255


def _input_scale(node: OpNode) -> float:
    """The runtime scale folded into this Softmax (`fold_scale`); 1.0 when none was."""
    trait = node.traits.get('input_scale')
    return 1.0 if trait is None else float(trait.data['scale'])


def _requested_approximation(node: OpNode) -> str:
    """The approximation this node asks for. Default 'exp': a plain ONNX Softmax with no directive
    lowers to the accurate integer exp, which needs no calibration."""
    directives = node.directives or {}
    approx = directives.get('approximation')
    return 'exp' if approx is None else str(approx).lower()


def _parse_hccs_directives(node_name: str, directives) -> dict:
    directives = directives or {}
    hccs = dict(directives.get('hccs', {}) or {})
    missing = [name for name in ('B', 'S', 'Dmax') if name not in hccs]
    if missing:
        raise ValueError(f'{node_name}: HCCS Softmax directives missing {", ".join(missing)}.')
    return hccs


class _SoftmaxVariantBase(OpImplVariant):
    """Everything Softmax variants share, independent of approximation and layout.

    The kernel is picked on two axes: `approximation` ('hccs' calibrated clipped-linear surrogate,
    or 'exp' accurate integer exponential) and `layout` ('linear' whole rows, or 'tiled' microtiles).
    """

    approximation: ClassVar[str]
    layout_name: ClassVar[str]

    op_type = 'softmax'
    graph_header = 'softmax_graph.h'
    graph_name = 'softmax_hccs_graph'
    param_template = 'softmax'
    plevel = 10
    supported_directives: ClassVar[frozenset] = frozenset({'approximation', 'layout', 'parallelism'})

    def work(self, node: OpNode, config) -> int:
        """Multiply-accumulates' time one tile takes per call, mostly per row: int8 HCCS on an AIE-ML tile took about
        22 cc a row and 0.18 cc an element with the CLB reciprocal (16x64: 545 cc, 64x128: 2929), about 235 and 0.3
        with the exact one (16x64: 4064), counted at its 256 int8 multiply-accumulates a cycle. The exp Softmax is
        slower still; it is costed as exact HCCS."""
        *rows, cols = config.io_views[input_tensor_for_role(node, 'lhs').name].tile
        per_row, per_element = (22, 0.18) if config.use_clb else (235, 0.3)
        return round(math.prod(rows) * (per_row + per_element * cols) * 256)

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not layout_variant_matches(node, self.layout_name):
            return False
        if _requested_approximation(node) != self.approximation:
            return False
        if device.generation not in ('AIE', 'AIE-ML', 'AIE-MLV2'):
            return False
        in_tensor = input_tensor_for_role(node, 'lhs')
        if isinstance(in_tensor.precision, FloatIntent):
            return False
        in_prec = resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name)
        return in_prec.format == 'int8'

    def resolve_microtile(self, _node: OpNode, _directives, _input_contracts):
        """The microtile this variant reads/writes, or None when it works on whole rows."""
        return None

    def _extra_precision(self) -> Dict[str, Any]:
        """Approximation-specific parameter tensors (e.g. HCCS B/S/Dmax)."""
        return {}

    def _resolve_params(self, node: OpNode, directives, full_inner: int, cas_num: int) -> Dict[str, Any]:
        """Approximation-specific config fields (score parameters)."""
        raise NotImplementedError

    def resolve(self, node: OpNode, device, directives, input_contracts) -> SoftmaxConfig:
        io_route, parallel_cfg = parse_directives(directives)

        in_tensor = input_tensor_for_role(node, 'lhs')
        out_tensor = node.outputs[0]
        precision = {
            'lhs': resolve_exact_storage_dtype(in_tensor.precision, namespace='lhs', layer_name=node.name),
            'output': resolve_exact_storage_dtype(out_tensor.precision, namespace='output', layer_name=node.name),
            **self._extra_precision(),
        }

        in_shape = tuple(int(x) for x in view_shape(node, in_tensor, 'inputs'))
        full_inner, outer_prefix, last_outer = extract_inner_outer(in_shape)
        vec_size = softmax_vec_size(precision['lhs'], device)
        microtile = self.resolve_microtile(node, directives, input_contracts)
        if microtile is None and full_inner % vec_size != 0:  # the tiled kernels check their microtile instead
            raise ValueError(
                f'{node.name}: softmax axis length {full_inner} must be a multiple of vec_size={vec_size}; '
                'pad the softmax dimension before lowering.'
            )

        cas_length = int(parallel_cfg.get('cas_length', 1))
        if cas_length != 1:
            raise ValueError(f'{node.name}: Softmax requires cas_length=1, got {cas_length}.')

        in_bpp = storage_bytes_for_spec(precision['lhs'])
        out_bpp = storage_bytes_for_spec(precision['output'])
        scratch_rows = microtile.outer if microtile else 1  # the int16 scores kept between passes
        cas_num, tile_outer = find_tile_split(
            partition_size=last_outer,
            max_rows=max(1, int(device.rows) - int(device.row_start)),
            bank_bytes=int(device.bank_mem_bytes),
            tile_bytes_fn=lambda to: max(
                outer_prefix * to * full_inner * in_bpp,
                outer_prefix * to * full_inner * out_bpp,
                scratch_rows * full_inner * 2,
            ),
            parallel_cfg=parallel_cfg,
            input_contracts=input_contracts,
            primary_tensor_name=in_tensor.name,
            contract='outer',
        )

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

        params = self._resolve_params(node, directives, int(full_inner), int(cas_num))
        return SoftmaxConfig(
            precision=precision,
            parallelism=ParallelismConfig(cas_num=int(cas_num), contract='outer'),
            vec_size=int(vec_size),
            accumulator_tag=infer_accumulator_tag(device, precision['lhs'], precision['lhs'], None),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            io_views=io_views,
            io_route=io_route,
            layout=self.layout_name,
            microtile=microtile,
            transpose_input=io_views[in_tensor.name].is_transposed,
            approximation=self.approximation,
            **params,
        )

    def validate_config(self, node: OpNode, config: SoftmaxConfig, _device) -> None:
        out_format = config.precision['output'].format
        if out_format not in ('uint8', 'int16'):
            raise ValueError(f'{node.name}: {self.variant_id} requires uint8 or int16 output, got {out_format!r}.')
        expected_frac = 8 if out_format == 'uint8' else 15
        if int(config.precision['output'].frac) != expected_frac:
            raise ValueError(
                f'{node.name}: {self.variant_id} emits {out_format} Q{expected_frac} probabilities, '
                f'got output frac={config.precision["output"].frac}.'
            )

    def kernel_params(self, node: OpNode, config: SoftmaxConfig):
        in_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(rows=int(in_view.compacted_tile_outer), cols=int(in_view.full_inner))
        params['packed_hccs'] = self._packed_hccs(config, int(in_view.full_inner))
        return params

    def _packed_hccs(self, config: SoftmaxConfig, cols: int) -> Dict[str, Any]:
        raise NotImplementedError

    def describe_input_staging(self, _node, config, tensor_name, port, _producer=None):
        return describe_partition_staging(
            config.io_views[tensor_name],
            port,
            'read',
            'outer',
        )

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_partition_staging(config.io_views[tensor_name], port, 'write', 'outer')

    def output_staging_contract(self, _node, config: SoftmaxConfig, _tensor_name: str):
        return str(config.parallelism.contract)

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        return {}

    def get_artifacts(self, inst: ExecutionInstance):
        return []

    def footprint(self, node: OpNode, config: SoftmaxConfig) -> OpImplFootprint:
        return OpImplFootprint(
            width=1,
            height=int(config.parallelism.cas_num),
        )

    def buffer_locations(self, _node: OpNode, config: SoftmaxConfig, anchor_row: int):
        locations = []
        for row in range(int(config.parallelism.cas_num)):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + row, 1)
            locations.append(BufferLocation('in1', row, flow.input_col, row, (0, 3)))
            locations.append(BufferLocation('out1', row, flow.output_col, row, (0, 3)))
        return tuple(locations)

    def build_ports(self, node: OpNode, config: SoftmaxConfig):
        in_tensor = input_tensor_for_role(node, 'lhs')
        n = int(config.parallelism.cas_num)
        return PortMap(
            inputs={in_tensor.name: PortBinding('in1', n, endpoints=kernel_endpoints(n, 'in[0]'))},
            outputs={node.outputs[0].name: PortBinding('out1', n, endpoints=kernel_endpoints(n, 'out[0]'))},
        )


class _SoftmaxTiledMixin:
    """Microtile inheritance and the tiled envelope, shared by the HCCS and exp tiled variants."""

    # Prefer microtiled execution when layout is unconstrained; explicit layout wins in matches().
    plevel = 11

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

    def validate_config(self, node: OpNode, config: SoftmaxConfig, device) -> None:
        super().validate_config(node, config, device)
        mt = config.microtile
        in_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        rows = int(in_view.compacted_tile_outer)
        cols = int(in_view.full_inner)
        if rows % mt.outer:
            raise ConfigRefused(
                f'{node.name}: rows={rows} must be a whole number of {mt.outer}-row microtile bands. '
                f'Partitioning across kernels chose this tile height, so adjust cas_num.'
            )
        if cols % mt.inner:
            raise ValueError(f'{node.name}: cols={cols} must be a whole number of {mt.inner}-wide microtiles.')
        block_bytes = mt.outer * mt.inner * storage_bytes_for_spec(config.precision['lhs'])
        if block_bytes < 16:
            raise ValueError(
                f'{node.name}: a {mt.outer}x{mt.inner} microtile of {config.precision["lhs"].c_type} is '
                f'{block_bytes} bytes, under the 16-byte minimum vector width. This layout needs a wider '
                f'microtile or a wider dtype.'
            )
        if mt.inner < 8 or mt.outer < 2 or mt.outer > 8 or mt.outer * mt.inner > 64:
            raise ValueError(
                f'{node.name}: the tiled Softmax cannot reduce a {mt.outer}x{mt.inner} microtile. '
                f'It needs 2 <= outer <= 8, inner >= 8, and outer*inner <= 64, matching the aie::mmul '
                f'microtiles a producer can emit. The producer chose this tiling; retile it, or run '
                f"this Softmax as layout='linear'."
            )


class _SoftmaxHccsBase(_SoftmaxVariantBase):
    """Head-Calibrated Clipped-Linear Softmax -- an integer surrogate using B/S/Dmax instead of an
    exponential. Not a drop-in for float Softmax. See https://arxiv.org/pdf/2604.02292v1"""

    approximation = 'hccs'
    supported_directives = _SoftmaxVariantBase.supported_directives | {'hccs'}

    def _extra_precision(self) -> Dict[str, Any]:
        return {'B': AIEDataType(format='int16'), 'S': AIEDataType(format='int8'), 'Dmax': AIEDataType(format='uint8')}

    def _resolve_params(self, node: OpNode, directives, full_inner: int, cas_num: int) -> Dict[str, Any]:
        hccs = _parse_hccs_directives(node.name, directives)
        input_scale = _input_scale(node)
        if abs(input_scale - 1.0) > 1e-9:
            raise ValueError(
                f'{node.name}: {self.variant_id} cannot apply a runtime input_scale={input_scale}; bake the '
                'softmax temperature into the HCCS B/S/Dmax calibration, or drop the approximation directive '
                'to use the accurate exp Softmax, which applies input_scale directly.'
            )
        return {
            'param_sets': int(infer_hccs_param_sets(hccs)),
            'inv_shift': int(hccs.get('inv_shift', DEFAULT_INV_SHIFT)),
            'use_clb': bool(hccs.get('use_clb', False)),
            'hccs': hccs,
            'exp_kq': 0,
        }

    def validate_config(self, node: OpNode, config: SoftmaxConfig, device) -> None:
        super().validate_config(node, config, device)
        if config.param_sets != 1:
            raise ValueError(
                f'{node.name}: {self.variant_id} does not support multi-head param_sets={config.param_sets}; '
                'lower attention to one Softmax op per head.'
            )
        if not (1 <= int(config.inv_shift) <= 30):
            raise ValueError(f'{node.name}: HCCS Softmax inv_shift must be in [1, 30], got {config.inv_shift}.')
        in_view = config.io_views[input_tensor_for_role(node, 'lhs').name]
        validate_hccs_params(config.hccs, cols=int(in_view.full_inner), param_sets=int(config.param_sets))

    def _packed_hccs(self, config: SoftmaxConfig, cols: int) -> Dict[str, Any]:
        return pack_hccs_params(
            config.hccs, param_sets=int(config.param_sets), cols=cols, cas_num=int(config.parallelism.cas_num)
        )


@register_variant
class SoftmaxHccsLinearOpImplVariant(_SoftmaxHccsBase):
    """HCCS Softmax over row-contiguous data."""

    variant_id = 'softmax.hccs.i8.v1'
    layout_name = 'linear'


@register_variant
class SoftmaxHccsTiledOpImplVariant(_SoftmaxTiledMixin, _SoftmaxHccsBase):
    """HCCS Softmax over microtiles."""

    variant_id = 'softmax.hccs.i8.tiled.v1'
    layout_name = 'tiled'
    supported_directives = _SoftmaxHccsBase.supported_directives | {'microtiling'}
    kernel_transposes_microtile = True


def _exp_kq(input_scale: float) -> int:
    """log2(e) * input_scale, in Q(EXP_ZF) -- the per-unit-gap step in the exp kernel."""
    return int(round(math.log2(math.e) * float(input_scale) * (1 << EXP_ZF)))


class _SoftmaxExpBase(_SoftmaxVariantBase):
    """Accurate integer Softmax: a real 2nd-order integer exp which a plain Softmax lowers here by default.

    TODO(perf): these exp kernels are much slower than the HCCS surrogate and are not yet
    optimised -- accurate and working, but slow; the poly + per-lane 2^(-z_int) shift need work.
    """

    approximation = 'exp'

    def _resolve_params(self, node: OpNode, directives, full_inner: int, cas_num: int) -> Dict[str, Any]:
        # The exp needs the real logit scale: the input's fixed-point scale (2^-frac) times any
        # runtime temperature trait. HCCS instead bakes this into its B/S/Dmax calibration.
        in_prec = resolve_exact_storage_dtype(
            input_tensor_for_role(node, 'lhs').precision, namespace='lhs', layer_name=node.name
        )
        temperature = _input_scale(node)
        input_scale = (2.0 ** -int(in_prec.frac)) * temperature
        return {
            'param_sets': 1,
            'inv_shift': DEFAULT_INV_SHIFT,
            'use_clb': False,
            'hccs': {},
            'exp_kq': _exp_kq(input_scale),
        }

    def validate_config(self, node: OpNode, config: SoftmaxConfig, device) -> None:
        super().validate_config(node, config, device)
        kq = int(config.exp_kq)
        if not (1 <= kq <= EXP_KQ_MAX):
            input_scale = _input_scale(node)
            raise ValueError(
                f'{node.name}: exp Softmax needs 1 <= EXP_KQ <= {EXP_KQ_MAX}, got {kq} for input_scale='
                f'{input_scale}. The input fixed-point scale is out of range (roughly frac in [2, 9]): too '
                'fine and every logit gap rounds to zero (uniform output); too coarse and (max-x)*KQ overflows '
                'int16. Requantise the softmax input, or use an int16-input exp variant.'
            )

    def _packed_hccs(self, config: SoftmaxConfig, cols: int) -> Dict[str, Any]:
        import numpy as np

        n = int(config.parallelism.cas_num)
        return {'B': np.zeros(n, np.int16), 'S': np.zeros(n, np.int8), 'Dmax': np.zeros(n, np.uint8)}


@register_variant
class SoftmaxExpLinearOpImplVariant(_SoftmaxExpBase):
    """Accurate exp Softmax over row-contiguous data -- the default for a plain ONNX Softmax."""

    variant_id = 'softmax.exp.i8.v1'
    layout_name = 'linear'


@register_variant
class SoftmaxExpTiledOpImplVariant(_SoftmaxTiledMixin, _SoftmaxExpBase):
    """Accurate exp Softmax over microtiles."""

    variant_id = 'softmax.exp.i8.tiled.v1'
    layout_name = 'tiled'
    supported_directives = _SoftmaxExpBase.supported_directives | {'microtiling'}
    kernel_transposes_microtile = True
