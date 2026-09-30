from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ...ir import OpNode


@dataclass(frozen=True)
class Endpoint:
    """One tensor endpoint participating in a transport leg."""

    node: Optional[OpNode]
    tensor: str
    group: str
    ports: Optional[Tuple[int, ...]] = None
    offset_base: Tuple[int, ...] = ()  # where the endpoint's part of the tensor starts, for a view's part

    def __post_init__(self) -> None:
        if self.ports is not None:
            ports = tuple(int(port) for port in self.ports)
            if not ports or len(set(ports)) != len(ports) or any(port < 0 for port in ports):
                raise ValueError(f'{self.tensor}: endpoint ports must be unique non-negative indices.')
            object.__setattr__(self, 'ports', ports)

        object.__setattr__(self, 'offset_base', tuple(int(value) for value in self.offset_base))

    def selected_ports(self, total: int) -> Tuple[int, ...]:
        total = int(total)
        if total <= 0:
            raise ValueError(f'{self.tensor}: endpoint port count must be positive, got {total}.')
        ports = self.ports if self.ports is not None else tuple(range(total))
        if any(port >= total for port in ports):
            raise ValueError(f'{self.tensor}: endpoint port selection {ports} exceeds port count {total}.')
        return ports


@dataclass(frozen=True)
class Connection:
    """One semantic producer-to-consumer or graph-boundary transport leg."""

    logical_tensor: str
    producer: Endpoint
    consumer: Optional[Endpoint]

    def __post_init__(self) -> None:
        if self.producer.node is None and self.consumer is None:
            raise ValueError(f'{self.logical_tensor}: transport leg cannot have two graph-boundary endpoints.')

    def endpoints(self) -> Tuple[Endpoint, ...]:
        return (self.producer,) if self.consumer is None else (self.producer, self.consumer)


@dataclass(frozen=True)
class TransportDecision:
    realization: str
    staging_compatible: Optional[bool]

    def __post_init__(self) -> None:
        if self.realization not in ('direct', 'memtile'):
            raise ValueError(f'Unsupported transport realization {self.realization!r}.')
        if self.realization == 'direct' and not self.staging_compatible:
            raise ValueError('Direct realization requires staging-compatible transport.')


@dataclass(frozen=True)
class TransportUnit:
    """The ports one realization of a leg joins: the whole leg when direct, or one memory-tile buffer of `count`."""

    producer_ports: Tuple[int, ...]
    consumer_ports: Tuple[int, ...]
    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if not self.producer_ports:
            raise ValueError('Transport unit requires at least one producer port.')
        if self.count <= 0 or self.index < 0 or self.index >= self.count:
            raise ValueError(f'Invalid transport unit index/count ({self.index}/{self.count}).')


@dataclass(frozen=True)
class GraphInputSpec:
    port_descriptors: Dict[int, Dict[str, Any]]
    writer_descriptors: Dict[int, Dict[str, Any]]


@dataclass
class EdgeEntry:
    """Transport leg state progressively resolved by transport passes."""

    logical_tensor: str
    producer: Endpoint
    producer_port_count: int
    consumers: list[Connection] = field(default_factory=list)
    graph_output: bool = False
    decision: Optional[TransportDecision] = None
    unit: Optional[TransportUnit] = None
    graph_input: Optional[GraphInputSpec] = None

    def endpoints(self) -> Tuple[Endpoint, ...]:
        return (self.producer, *(conn.consumer for conn in self.consumers if conn.consumer is not None))

    def single_consumer(self) -> Endpoint:
        if len(self.consumers) != 1 or self.consumers[0].consumer is None:
            raise RuntimeError(f'{self.logical_tensor}: transport entry requires exactly one consumer.')
        return self.consumers[0].consumer
