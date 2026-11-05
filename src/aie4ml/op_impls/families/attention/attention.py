from __future__ import annotations

from typing import ClassVar

from ....ir.graph import OpNode, has_input_role, input_tensor_for_role
from ...base import PING_PONG_BANKS, BufferLocation, OpImplVariant
from ...common_types import PortBinding, PortMap
from ...registry import register_variant
from ...utils import ParallelismConfig, build_tensor_view
from ..matmul.common import describe_outer_lhs_staging, describe_outer_output_staging
from .common import (
    BIAS_BLOCK_CYCLES,
    ROW_BANDS,
    attends,
    attention_params,
    bias_locations,
    bias_shift,
    bias_view,
    chain_tiles,
    chains,
    check_bank,
    softmax_arithmetic,
    spill_bytes,
    stack_bytes,
    storage_precision,
    stored_microtile,
    tokens_of,
)
from .config import AttentionConfig


@register_variant
class AttentionOpImplVariant(OpImplVariant):
    """One head's softmax(Q Kᵀ) V, the probabilities unnormalised (exp(0) = 255) and each output divided once: within
    a code of the QDQ graph. Chains split the queries: a tile each, or a scores tile handing the probabilities down
    the cascade to a values tile that divides. exp and the reciprocal are table lookups."""

    variant_id = 'attention.i8.v1'
    op_type = 'attention'
    graph_header = 'attention_graph.h'
    graph_name = 'attention_graph'
    param_template = 'attention'
    supported_directives: ClassVar[frozenset] = frozenset({'parallelism'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        return has_input_role(node, 'q') and attends(node, device)

    def resolve(self, node: OpNode, device, directives, input_contracts) -> AttentionConfig:
        cas_num, cas_length = chains(node, device, directives)
        q, k, v = (input_tensor_for_role(node, role) for role in ('q', 'k', 'v'))
        out = node.outputs[0]
        tokens, head_dim = (int(x) for x in q.shape)
        # the head dimension as the producers store it: a Dense pads its columns to whole steps with zeros, which the
        # kernel reads as more of the head dimension (zero scores, zero outputs), so the hand-overs stay direct
        contracts = [input_contracts.get(t.name) for t in (q, k, v)]
        columns = max([head_dim, *(int(c.view.full_inner) for c in contracts if c is not None and c.view is not None)])
        # in each ping-pong bank: every key on a chain's first tile, every value and the chain's output on its last (on
        # one tile, all three)
        check_bank(node, device, (3 - cas_length) * tokens * columns + tokens // cas_num * columns, 'buffers')
        fracs = [int(t.precision.frac) for t in (q, k, v, out)]

        def view(tensor, direction, rows, microtile):
            return build_tensor_view(
                node, tensor, direction, full_inner=columns, tile_inner=columns, tile_inner_raw=head_dim,
                full_outer=tokens, tile_outer=rows, tile_outer_raw=rows, microtile=microtile,
            )  # fmt: skip

        # each buffer in its producer's microtile; O in Q's, its rows the same
        mq, mk, mv = (stored_microtile(c) for c in contracts)
        io_views = {
            q.name: view(q, 'inputs', tokens // cas_num, mq),
            k.name: view(k, 'inputs', tokens, mk),
            v.name: view(v, 'inputs', tokens, mv),
            out.name: view(out, 'outputs', tokens // cas_num, mq),
        }
        if has_input_role(node, 'bias'):
            io_views[input_tensor_for_role(node, 'bias').name] = bias_view(node, cas_num)
        return AttentionConfig(
            precision=storage_precision(node, [('q', q), ('k', k), ('v', v), ('output', out)]),
            parallelism=ParallelismConfig(cas_num=cas_num, cas_length=cas_length, contract='outer'),
            io_views=io_views,
            alternating_horizontal=device.alternating_horizontal,
            columns=columns,
            **softmax_arithmetic(node, *fracs),
            table_bank_bits=int(device.lookup_bank_bits),
            # one copy of each table a tile looks up in: exp and the reciprocal, or one of them in a chain of two
            stack_bytes=stack_bytes(
                node, device, 2 if cas_length == 1 else 1, spill_bytes(device, 'plain', cas_length, tokens, columns)
            ),
            int8_macs_per_cycle=int(device.int8_macs_per_cycle),
        )

    def kernel_params(self, node: OpNode, config: AttentionConfig):
        roles = {role: input_tensor_for_role(node, role).name for role in ('q', 'k', 'v')}
        roles['output'] = node.outputs[0].name
        return {
            **attention_params(node, config),
            'microtile_rows': {role: int(config.io_views[name].microtile.outer) for role, name in roles.items()},
        }

    def build_ports(self, node: OpNode, config: AttentionConfig) -> PortMap:
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        q, k, v = (input_tensor_for_role(node, role) for role in ('q', 'k', 'v'))
        return PortMap(
            inputs={
                q.name: PortBinding('in1', n, endpoints=tuple((f'kk[{i * c}].in[0]',) for i in range(n))),
                k.name: PortBinding('in2', 1, endpoints=(tuple(f'kk[{i * c}].in[1]' for i in range(n)),)),
                v.name: PortBinding(
                    'in3', 1, endpoints=(tuple(f'kk[{i * c + c - 1}].in[{2 if c == 1 else 1}]' for i in range(n)),)
                ),
            },
            outputs={
                node.outputs[0].name: PortBinding(
                    'out1', n, endpoints=tuple((f'kk[{i * c + c - 1}].out[0]',) for i in range(n))
                )
            },
        )

    def describe_input_staging(self, _node, config, tensor_name, port):
        """A row band at the stored width, laid out as a Dense writes one (Q a chain's band, K and V one of every
        row), so a Dense hands it over directly."""
        return describe_outer_lhs_staging(config.io_views[tensor_name], ROW_BANDS, port)

    def describe_output_staging(self, _node, config, tensor_name, port):
        return describe_outer_output_staging(config.io_views[tensor_name], port)

    def output_staging_contract(self, _node, config, _tensor_name: str):
        return 'outer'

    def buffer_locations(self, _node: OpNode, config: AttentionConfig, anchor_row: int):
        """A chain's queries where its first tile reads its producer's (as a row-wise op's input), every key in that
        tile's own memory and every value in its last tile's, its output after its last tile (as a row-wise op's)."""
        locations = []
        for chain, flow, cols in chain_tiles(config, anchor_row):
            locations.append(
                BufferLocation('in1', chain, cols[0] + flow.input_col, chain, PING_PONG_BANKS, reader=(cols[0], chain))
            )
            locations.append(BufferLocation('in2', 0, cols[0], chain, PING_PONG_BANKS, reader=(cols[0], chain)))
            locations.append(BufferLocation('in3', 0, cols[-1], chain, PING_PONG_BANKS, reader=(cols[-1], chain)))
            locations.append(BufferLocation('out1', chain, cols[-1] + flow.output_col, chain, PING_PONG_BANKS))
        return tuple(locations)

    def work(self, node: OpNode, config: AttentionConfig) -> int:
        """A chain's slower tile's cycles per call (`attention_cycles`), as its core's multiply-accumulates."""
        return attention_cycles(node, config) * config.int8_macs_per_cycle


@register_variant
class AttentionBiasOpImplVariant(AttentionOpImplVariant):
    """AttentionOpImplVariant with an additive bias on the logits (attention_bias.h): a chain's first tile reads its
    queries' rows of it, which start the scores' accumulator, so the logits and the bias round once."""

    variant_id = 'attention.bias.i8.v1'
    graph_header = 'attention_bias_graph.h'
    graph_name = 'attention_bias_graph'
    supported_fusions: ClassVar[frozenset] = frozenset({'bias'})

    def matches(self, node: OpNode, device, _directives) -> bool:
        if not (has_input_role(node, 'q') and has_input_role(node, 'bias') and attends(node, device)):
            return False
        products = sum(int(input_tensor_for_role(node, role).precision.frac) for role in ('q', 'k'))
        return bias_shift(node, products) is not None

    def build_ports(self, node: OpNode, config: AttentionConfig) -> PortMap:
        ports = super().build_ports(node, config)
        n, c = int(config.parallelism.cas_num), int(config.parallelism.cas_length)
        u = PortBinding('in4', n, endpoints=tuple((f'kk[{i * c}].in[{3 if c == 1 else 2}]',) for i in range(n)))
        return PortMap(inputs={**ports.inputs, input_tensor_for_role(node, 'bias').name: u}, outputs=ports.outputs)

    def buffer_locations(self, node: OpNode, config: AttentionConfig, anchor_row: int):
        """As AttentionOpImplVariant's, and a chain's rows of the bias as its queries (`bias_locations`)."""
        return (*super().buffer_locations(node, config, anchor_row), *bias_locations(config, anchor_row, 'in4'))


def attention_cycles(node: OpNode, config: AttentionConfig) -> int:
    """Cycles of a chain's slower tile per call: per query block, fitted to the kernels' AIE-ML profiles at 32 and 64
    tokens, head dimensions 16 to 64."""
    tokens = tokens_of(node)
    par = config.parallelism
    qb, kb, nj = tokens // par.cas_num // 8, tokens // 8, config.columns // 8
    bias = kb * BIAS_BLOCK_CYCLES[config.bias_bits] if config.bias_bits else 0
    if par.cas_length == 1:
        block = 38 * kb + 12.3 * kb * nj + 21 + bias
    else:  # the probabilities' tile or the values'
        block = max(48 * kb + 6 * kb * nj - 50 + bias, 14 * kb * nj + 4 * nj + 10)
    return round(25 + qb * block)
