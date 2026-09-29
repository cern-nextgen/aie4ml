from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np

from ....aie_types import FloatIntent
from ....errors import ConfigRefused
from ....ir.graph import ExecutionInstance, OpNode, has_input_role, input_tensor_for_role, shard_major_order
from ....passes.utils import sanitize_identifier
from ...base import BufferLocation, OpImplFootprint, OpImplVariant, row_flow
from ...common_types import PORT_KIND_STREAM, PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, inherited_microtile, parse_directives
from ...utils.precision import (
    aie_rounding_token,
    element_bytes,
    resolve_accumulator_output_shift,
    resolve_bias_dtype,
    resolve_operand_precision,
    resolve_output_scale_shift,
)
from .common import (
    bitwidths_supported,
    describe_inner_lhs_staging,
    describe_inner_output_staging,
    describe_outer_lhs_staging,
    describe_outer_output_staging,
    describe_stream_staging,
    np_bias_dtype_for_spec,
    np_dtype_for_spec,
    pack_as_float,
    pack_mmul_rhs_matrix,
    pack_vector_by_n_slice,
    quantize_to_int,
    requested_contract,
)
from .config import DenseConfig, DenseFlags
from .resolver import _build_matmul_io_views, _resolve_parallelism, _resolve_tile_cfg


class _BaseDenseMatmulVariant(OpImplVariant):
    """Unregistered shared base for Dense and Matmul variants."""

    contract: ClassVar[str]
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism', 'microtiling'})

    def work(self, node, config) -> int:
        lhs = config.io_views[input_tensor_for_role(node, 'lhs').name].tile
        return int(np.prod(lhs)) * int(config.io_views[node.outputs[0].name].tile[-1])

    def kernel_params(self, node, config):
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        lhs_view = config.io_views[lhs_tensor.name]
        output_view = config.io_views[node.outputs[0].name]
        params = {f: getattr(config, f) for f in config.__dataclass_fields__}
        params.update(
            full_outer=self.kernel_outer_extent(lhs_view),
            full_inner_lhs=lhs_view.full_inner,
            full_inner_rhs=output_view.full_inner,
            tile_inner_lhs=lhs_view.tile_inner,
            tile_inner_rhs=output_view.tile_inner,
            tile_inner_lhs_raw=lhs_view.tile_raw_inner,
            tile_inner_rhs_raw=output_view.tile_raw_inner,
        )
        params['stream_io'] = self.port_kind == PORT_KIND_STREAM
        return params

    def kernel_outer_extent(self, lhs_view):
        """Rows this kernel loops over. Contract-specific: declared by each variant."""
        raise NotImplementedError

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return str(config.parallelism.contract)


class _DenseVariantBase(_BaseDenseMatmulVariant):
    """Everything the dense variants share, independent of which contract they implement."""

    op_type = 'dense'
    kernel_transposes_microtile = True
    graph_header = 'dense_bias_relu_graph.h'
    graph_name = 'dense_bias_relu_graph'
    param_template = 'dense_bias_relu'
    plevel = 10

    def matches(self, node: OpNode, device, directives) -> bool:
        return requested_contract(node, directives) == self.contract and bitwidths_supported(node, device)

    def resolve(self, node: OpNode, device, directives, input_contracts) -> DenseConfig:
        io_route, parallel_cfg = parse_directives(directives)
        precision, accumulator_tag = resolve_operand_precision(node, device)
        precision['bias'] = resolve_bias_dtype(node, precision)
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        required_microtile = None
        producer_contract = input_contracts.get(lhs_tensor.name)
        # A hand-off with no memory tile to re-shard it (AIE1, or any stream port) takes the
        # producer's port count as its cas_length and, for buffers, inherits its microtile.
        direct_only = not device.has_memtile or self.port_kind == PORT_KIND_STREAM
        if direct_only and self.contract == 'inner' and producer_contract is not None:
            if producer_contract.contract != 'inner':
                raise ConfigRefused(
                    f'{node.name}: dense inner contract cannot directly consume producer '
                    f'{producer_contract.contract!r} staging.'
                )
            required_cas_length = len(producer_contract.port_staging)
            requested_cas_length = parallel_cfg.get('cas_length')
            if requested_cas_length is not None and int(requested_cas_length) != required_cas_length:
                raise ConfigRefused(
                    f'{node.name}: cas_length={requested_cas_length} conflicts with the producer port count '
                    f'{required_cas_length} required for direct AIE1 transport.'
                )
            parallel_cfg['cas_length'] = required_cas_length
            required_microtile = inherited_microtile(node, input_contracts)

        microtiling = _resolve_tile_cfg(
            node,
            device,
            precision['lhs'],
            precision['rhs'],
            required_lhs_microtile=required_microtile,
        )
        tiling = _resolve_parallelism(
            node,
            device,
            microtiling,
            precision,
            self.contract,
            parallel_cfg=parallel_cfg,
        )
        io_views = _build_matmul_io_views(node, microtiling, tiling)
        # K stored shard by shard is still K: the kernel reduces over it whatever its order, so the weight rows
        # follow it (see pack) and every tiling of it stays legal.
        lhs_inner_shards = producer_contract.inner_shards if producer_contract is not None else None
        if lhs_inner_shards is not None and io_views[lhs_tensor.name].is_transposed:
            raise ConfigRefused(
                f'{node.name}: a transposed view of {lhs_tensor.name!r} reduces over another axis than the one its '
                'producer stores shard by shard.'
            )

        rhs_tensor = input_tensor_for_role(node, 'rhs')
        is_float = isinstance(lhs_tensor.precision, FloatIntent)

        shift = (
            0
            if is_float
            else resolve_accumulator_output_shift(lhs_tensor.precision, node.outputs[0].precision, rhs_tensor.precision)
        )
        shift += resolve_output_scale_shift(node, is_float=is_float)

        fused_act = node.traits.get('fused_activation')
        use_relu = fused_act is not None and fused_act.data['activation'] == 'relu'

        return DenseConfig(
            precision=precision,
            parallelism=ParallelismConfig(
                cas_length=tiling.cas_length, cas_num=tiling.cas_num, contract=tiling.contract
            ),
            microtiling=microtiling,
            io_views=io_views,
            io_route=io_route,
            shift=shift,
            accumulator_tag=accumulator_tag,
            rounding_mode='conv_even' if is_float else aie_rounding_token(precision['output']),
            bank_mem_bytes=int(device.bank_mem_bytes),
            alternating_horizontal=device.cascade_layout == 'alternating_horizontal',
            flags=DenseFlags(
                use_relu=use_relu,
                transpose_lhs=io_views[lhs_tensor.name].is_transposed,
                use_bias=has_input_role(node, 'bias'),
            ),
            lhs_inner_shards=lhs_inner_shards,
        )

    def input_inner_shards(self, node, config, tensor_name):
        return config.lhs_inner_shards if tensor_name == input_tensor_for_role(node, 'lhs').name else None

    def _quantize_weight_bias(self, inst: ExecutionInstance):
        """Quantize the weight matrix and bias vector; shared by every dense contract."""
        input_tensor = inst.node.inputs[0]
        weight_tensor = inst.node.inputs[1]
        bias_tensor = inst.node.inputs[2] if len(inst.node.inputs) > 2 else None

        wi = weight_tensor.precision
        if isinstance(wi, FloatIntent):
            W = pack_as_float(weight_tensor.data, wi.format)
            b = np.asarray(bias_tensor.data, dtype=np.float32) if bias_tensor is not None else None
        else:
            W = quantize_to_int(
                weight_tensor.data,
                wi.frac,
                wi.width,
                signed=wi.signed,
                rounding_mode=wi.rounding,
                saturation_mode=wi.saturation,
            )
            if bias_tensor is not None:
                bi = bias_tensor.precision
                accum_frac = input_tensor.precision.frac + wi.frac
                b = quantize_to_int(
                    bias_tensor.data,
                    accum_frac,
                    32,
                    signed=bi.signed,
                    rounding_mode=bi.rounding,
                    saturation_mode=bi.saturation,
                )
            else:
                b = None

        W = np.asarray(W)
        if W.ndim < 2:
            raise ValueError(f'{inst.name}: weight matrix must have at least 2 dimensions, got {W.ndim}.')
        shards = inst.config.lhs_inner_shards
        if shards is not None:
            W = np.take(W, shard_major_order(W.shape[-2], *shards), axis=-2)
        return W, b, int(W.shape[-2]), int(W.shape[-1])

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(
            width=int(config.parallelism.cas_length),
            height=int(config.parallelism.cas_num),
        )

    def buffer_locations(self, _node, config, anchor_row):
        locations = []
        cas_num = int(config.parallelism.cas_num)
        cas_length = int(config.parallelism.cas_length)
        outer = config.parallelism.contract == 'outer'
        for chain in range(cas_num):
            flow = row_flow(config.alternating_horizontal, int(anchor_row) + chain, cas_length)
            for pos in range(cas_length):
                idx = chain * cas_length + pos
                tile_col = cas_length - 1 - pos if flow.reversed else pos
                port = idx if outer else pos
                locations.append(BufferLocation('in1', port, tile_col + flow.input_col, chain, (0, 3)))
            last = 0 if flow.reversed else cas_length - 1
            locations.append(BufferLocation('out1', chain, last + flow.output_col, chain, (0, 3)))
        return tuple(locations)

    def get_artifacts(self, inst: ExecutionInstance):
        inst_name = sanitize_identifier(inst.name)
        p = inst.config
        output_view = p.io_views[inst.node.outputs[0].name]
        artifacts = [
            {
                'name': 'weights',
                'kind': '2d',
                'storage': 'rom',
                'array': inst.artifacts['packed_weights'],
                'dtype': p.precision['rhs'].c_type,
                'storage_dtype': p.precision['rhs'].storage_dtype,
                'filename': f'weights_{inst_name}.h',
                'port': 'wts',
            }
        ]
        packed_bias = inst.artifacts.get('packed_bias')
        if packed_bias is None:
            # Dense graph always exposes a bias RTP port; feed explicit zeros for biasless layers.
            packed_bias = np.zeros(
                (int(p.parallelism.cas_num), output_view.tile_inner),
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
        artifacts.append(
            {
                'name': 'bias',
                'kind': '1d',
                'storage': 'rom',
                'array': packed_bias,
                'dtype': p.precision['bias'].c_type,
                'storage_dtype': p.precision['bias'].storage_dtype,
                'filename': f'bias_{inst_name}.h',
                'port': 'bias',
            }
        )
        return artifacts

    def validate_config(self, node: OpNode, config: DenseConfig, _device) -> None:
        if config.shift < 0:
            raise ValueError(f'{node.name}: dense accumulator output shift must be non-negative, got {config.shift}.')

    def build_ports(self, node: OpNode, config: DenseConfig):
        cas_length = int(config.parallelism.cas_length)
        cas_num = int(config.parallelism.cas_num)
        if config.parallelism.contract == 'outer':
            lhs_endpoints = tuple((f'kk[{port}].in[0]',) for port in range(cas_length * cas_num))
        else:
            lhs_endpoints = tuple(
                tuple(f'kk[{chain * cas_length + port}].in[0]' for chain in range(cas_num))
                for port in range(cas_length)
            )
        out_endpoints = tuple((f'kk[{chain * cas_length + cas_length - 1}].out[0]',) for chain in range(cas_num))
        lhs_tensor = input_tensor_for_role(node, 'lhs')
        return PortMap(
            inputs={lhs_tensor.name: PortBinding('in1', len(lhs_endpoints), self.port_kind, lhs_endpoints)},
            outputs={node.outputs[0].name: PortBinding('out1', cas_num, self.port_kind, out_endpoints)},
        )


@register_variant
class DenseOpImplVariant(_DenseVariantBase):
    """Dense with the output features partitioned across cascade chains ('inner' contract)."""

    variant_id = 'dense.b.r.v1'
    contract = 'inner'

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_full_outer

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_inner_lhs_staging(config.io_views[tensor_name], port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_inner_output_staging(config.io_views[tensor_name], port, buf_dims)

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        # 'inner': cas_num slices the columns, so chain c owns weight/bias columns
        # [c*N_slice, (c+1)*N_slice) -- which is exactly what the packers lay out.
        p = inst.config
        W, b, n_in, n_out = self._quantize_weight_bias(inst)
        lhs_view = p.io_views[inst.node.inputs[0].name]
        output_view = p.io_views[inst.node.outputs[0].name]

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=lhs_view.tile_inner,
            N_slice=output_view.tile_inner,
            microtile_k=p.microtiling.microtile_k,
            microtile_n=p.microtiling.microtile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=p.parallelism.cas_num,
            dtype=np_dtype_for_spec(p.precision['rhs']),
        )
        packed_B = (
            pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=output_view.tile_inner,
                cas_num=p.parallelism.cas_num,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            if b is not None
            else None
        )
        return {'packed_weights': packed_W, 'packed_bias': packed_B}


@register_variant
class DenseRowWiseOpImplVariant(_DenseVariantBase):
    """Dense with the rows partitioned across cascade chains ('outer' contract)."""

    variant_id = 'dense.b.r.row.v1'
    contract = 'outer'
    plevel = 10

    def kernel_outer_extent(self, lhs_view):
        return lhs_view.compacted_tile_outer

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_outer_lhs_staging(config.io_views[tensor_name], config.parallelism, port, buf_dims)

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_outer_output_staging(config.io_views[tensor_name], port, buf_dims)

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        # The packers slice columns as chain*N_slice; here N_slice is the whole N, so pack a
        # single chain (offset 0, full width) and give every row-group that same copy.
        p = inst.config
        W, b, n_in, n_out = self._quantize_weight_bias(inst)
        lhs_view = p.io_views[inst.node.inputs[0].name]
        output_view = p.io_views[inst.node.outputs[0].name]
        cas_num = int(p.parallelism.cas_num)

        packed_W = pack_mmul_rhs_matrix(
            W,
            K=n_in,
            N=n_out,
            K_slice=lhs_view.tile_inner,
            N_slice=output_view.tile_inner,
            microtile_k=p.microtiling.microtile_k,
            microtile_n=p.microtiling.microtile_n,
            cas_length=p.parallelism.cas_length,
            cas_num=1,
            dtype=np_dtype_for_spec(p.precision['rhs']),
        )
        packed_W = np.repeat(packed_W, cas_num, axis=0)

        packed_B = None
        if b is not None:
            packed_B = pack_vector_by_n_slice(
                b,
                N=n_out,
                N_slice=output_view.tile_inner,
                cas_num=1,
                dtype=np_bias_dtype_for_spec(p.precision['bias']),
            )
            packed_B = np.repeat(packed_B, cas_num, axis=0)
        return {'packed_weights': packed_W, 'packed_bias': packed_B}


class _StreamDenseMixin:
    """Stream-port flavour of a dense variant."""

    port_kind: ClassVar[str] = PORT_KIND_STREAM

    def buffer_locations(self, _node, _config, _anchor_row):
        return ()

    def footprint(self, _node, config) -> OpImplFootprint:
        return OpImplFootprint(width=int(config.parallelism.cas_length), height=int(config.parallelism.cas_num))

    def describe_input_staging(self, _node, config, tensor_name, port, buf_dims=None, _producer=None):
        return describe_stream_staging(
            config.io_views[tensor_name], port, 'read', self.contract, config.parallelism.cas_length, buf_dims
        )

    def describe_output_staging(self, _node, config, tensor_name, port, buf_dims=None):
        return describe_stream_staging(config.io_views[tensor_name], port, 'write', self.contract, buf_dims=buf_dims)

    def validate_config(self, node: OpNode, config: DenseConfig, device) -> None:
        super().validate_config(node, config, device)
        if config.flags.transpose_lhs:
            raise ValueError(f'{node.name}: a stream port carries the tensor in linear order; it cannot transpose it.')
        # The kernel reads and writes the stream in 128-bit chunks and re-tiles a row band in
        # registers: a microtile row must be one or more whole chunks, or half a chunk zipped
        # from an even number of rows.
        m = int(config.microtiling.microtile_m)
        for axis, extent, spec in (
            ('K', int(config.microtiling.microtile_k), config.precision['lhs']),
            ('N', int(config.microtiling.microtile_n), config.precision['output']),
        ):
            row_bytes = extent * element_bytes(spec)
            if row_bytes == 8 and m % 2 == 0:
                continue
            if row_bytes >= 16 and row_bytes % 16 == 0:
                continue
            raise ValueError(
                f'{node.name}: microtile {axis}={extent} of {element_bytes(spec)}-byte elements ({row_bytes} B '
                f'per row, M={m}) cannot be staged from a stream; a row must be a multiple of 16 B, '
                'or 8 B with an even M.'
            )


@register_variant
class DenseStreamOpImplVariant(_StreamDenseMixin, DenseOpImplVariant):
    variant_id = 'dense.b.r.stream.v1'


@register_variant
class DenseRowWiseStreamOpImplVariant(_StreamDenseMixin, DenseRowWiseOpImplVariant):
    variant_id = 'dense.b.r.row.stream.v1'
