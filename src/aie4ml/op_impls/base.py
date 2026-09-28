from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..ir.graph import ExecutionInstance, ExecutionValue, OpNode
from .common_types import PORT_KIND_BUFFER, PORT_KIND_STREAM, PortMap


@dataclass(frozen=True)
class BufferLocation:
    """A port buffer's footprint-relative location; the op's graph pins it there, ping and pong one
    per bank. Weights, stacks and cascade resources are not listed."""

    port_group: str
    port: int
    rel_col: int
    rel_row: int
    banks: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.port_group or self.port < 0:
            raise ValueError('Buffer locations require a port group and non-negative port index.')
        if (
            not self.banks
            or len(self.banks) > 2
            or len(set(self.banks)) != len(self.banks)
            or any(bank not in range(4) for bank in self.banks)
        ):
            raise ValueError(f'Invalid ADF bank set {self.banks}.')


@dataclass(frozen=True)
class RowFlow:
    """Hand-over geometry of one row. `input_col`/`output_col` offset the tile holding a kernel's input
    (a chain's output) from that kernel's (the chain's last kernel's) column; `reversed` marks a cascade
    running right to left (odd rows on AIE)."""

    reversed: bool
    input_col: int
    output_col: int


def row_flow(alternating_horizontal: bool, row: int, cas_length: int) -> RowFlow:
    """On AIE odd-row cores reach their east neighbour's memory, even rows the west's; on AIE-ML all west."""
    reaches_east = bool(alternating_horizontal and int(row) % 2)
    if reaches_east and int(cas_length) > 1:
        return RowFlow(reversed=True, input_col=1, output_col=0)
    if reaches_east:
        return RowFlow(reversed=False, input_col=0, output_col=1)
    return RowFlow(reversed=False, input_col=-1, output_col=0)


@dataclass(frozen=True)
class LayoutConversion:
    """A kernel graph re-laying `source` into `target`, an execution-only value the op reads instead.
    `shared_memory` makes the hand-over to the op a hard no-DMA requirement."""

    name: str
    source: str
    target: str
    variant: 'OpImplVariant'
    config: Any
    shared_memory: bool


@dataclass(frozen=True)
class OpImplFootprint:
    """Rectangular tile footprint required by an op implementation."""

    width: int
    height: int
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f'Footprint dimensions must be positive, got {self.width}x{self.height}.')


class OpImplVariant:
    """Self-contained compilation unit for one op variant.

    Each subclass owns the full lifecycle: selection (matches + plevel),
    configuration (resolve), verification (validate_config), and code
    generation (build_template_params, build_ports, footprint, pack, get_artifacts).
    """

    variant_id: ClassVar[str] = ''
    op_type: ClassVar[str] = ''
    graph_header: ClassVar[str] = ''
    graph_name: ClassVar[str] = ''
    param_template: ClassVar[str] = ''
    plevel: ClassVar[int] = 10  # higher value = higher selection priority
    kernel_transposes_microtile: ClassVar[bool] = False
    port_kind: ClassVar[str] = PORT_KIND_BUFFER  # what the `ports` directive selects on
    # Directives read beyond placement, io_route and ports, which every variant honours; others are refused.
    supported_directives: ClassVar[frozenset] = frozenset()

    def matches(self, _node: OpNode, _device: Any, _directives: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def resolve(
        self, _node: OpNode, _device: Any, _directives: Dict[str, Any], _input_contracts: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError

    def validate_config(self, _node: OpNode, _config: Any, _device: Any) -> None:
        """Post-lowering attribute verifier. Override to enforce kernel ABI rules."""

    def kernel_params(self, _node: OpNode, _config: Any) -> Dict[str, Any]:
        """What the kernel's configuration struct is generated from, placement aside: everything the compiler
        schedules the kernel from."""
        raise NotImplementedError(f'{self.variant_id} does not separate its kernel parameters from its placement.')

    def build_template_params(self, node: OpNode, config: Any, placement: Dict[str, int]) -> Dict[str, Any]:
        """The kernel parameters, then where placement puts its buffers."""
        locations = self.buffer_locations(node, config, int(placement['row']))
        return {**self.kernel_params(node, config), 'buffer_locations': locations}

    def input_conversions(
        self, _node: OpNode, _config: Any, _sources: Dict[str, ExecutionValue]
    ) -> Tuple['LayoutConversion', ...]:
        """Conversions for inputs whose `sources` (execution values) arrive in a layout the op cannot read."""
        return ()

    def buffer_locations(self, _node: OpNode, _config: Any, _anchor_row: int) -> Tuple[BufferLocation, ...]:
        """Return transport-visible buffers relative to an op anchor.

        Repeated group/port pairs describe multicast graph ports. A shared-memory edge lists both of its
        ports at the same place.
        """
        return ()

    def output_staging_contract(self, _node: OpNode, _config: Any, _tensor_name: str) -> Optional[str]:
        return None

    def output_port_count(self, _node: OpNode, config: Any) -> Optional[int]:
        return int(config.parallelism.cas_num)

    def pack(self, inst: ExecutionInstance) -> Dict[str, Any]:
        raise NotImplementedError

    def get_artifacts(self, inst: ExecutionInstance) -> List[Dict[str, Any]]:
        return []

    def input_precision(self, config: Any, role: str) -> Any:
        return config.precision[role]

    def output_precision(self, config: Any) -> Any:
        return config.precision['output']

    def describe_output_staging(
        self, _node: OpNode, _config: Any, _tensor_name: str, _port: int, _buf_dims: Any = None
    ) -> Any:
        return None

    def describe_input_staging(
        self,
        _consumer: OpNode,
        _config: Any,
        _tensor_name: str,
        _port: int,
        _buf_dims: Any = None,
        _producer: Optional[OpNode] = None,
    ) -> Any:
        return None

    def footprint(self, node: OpNode, config: Any) -> OpImplFootprint:
        raise NotImplementedError

    def work(self, node: OpNode, config: Any) -> int:
        """Multiply-accumulates one tile computes per call, padding included: the size of this stage by which a
        performance search compares designs. A proxy for time, not a measure of it."""
        raise NotImplementedError(f'{node.name}: {self.variant_id} gives no work estimate to compare designs by.')

    def build_ports(self, _node: OpNode, _config: Any) -> PortMap:
        """Assemble the PortMap for this variant.  Must be overridden."""
        raise NotImplementedError

    def validate_ports(self, node: OpNode, ports: PortMap, device: Any) -> None:
        """Every kernel of an op sees at most one port per group, so the stream groups must fit
        the core's stream ports (two in/out on AIE, one in/out on AIE-ML)."""
        for direction, bindings, budget in (
            ('input', ports.inputs, int(device.core_stream_inputs)),
            ('output', ports.outputs, int(device.core_stream_outputs)),
        ):
            streams = [name for name, binding in bindings.items() if binding.kind == PORT_KIND_STREAM]
            if len(streams) > budget:
                raise ValueError(
                    f'{node.name}: {self.variant_id} needs {len(streams)} {direction} stream ports per kernel '
                    f'({", ".join(streams)}) but {device.platform} cores have {budget}.'
                )
