from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

from .utils.tensor_view import StagingDescriptor

PORT_KIND_BUFFER = 'buffer'
PORT_KIND_STREAM = 'stream'
PORT_KINDS = frozenset({PORT_KIND_BUFFER, PORT_KIND_STREAM})


@dataclass(frozen=True)
class PortBinding:
    """Binds one tensor to an ADF port group and replication count.

    `group` is the ADF port array name (for example `in1` or `out2`).
    `count` is the number of physical ports in that group.
    `kind` is the ADF transport kind: a `buffer` port is a DMA-fed tile buffer whose
    layout the staging descriptor describes; a `stream` port is a core stream that
    carries that descriptor's tile in its linear element order.
    `endpoints[i]` names the kernel ports (`kk[k].in[j]` / `kk[k].out[j]`) behind
    hierarchical port `i`; DMA access constraints bind there, not on the group.
    """

    group: str
    count: int
    kind: str = PORT_KIND_BUFFER
    endpoints: Tuple[Tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PORT_KINDS:
            raise ValueError(f'{self.group}: unknown port kind {self.kind!r}; expected one of {sorted(PORT_KINDS)}.')
        endpoints = tuple(tuple(str(name) for name in names) for names in self.endpoints)
        if len(endpoints) != self.count or any(not names for names in endpoints):
            raise ValueError(
                f'{self.group}: expected kernel endpoints for each of {self.count} ports, got {endpoints}.'
            )
        object.__setattr__(self, 'endpoints', endpoints)


def kernel_endpoints(count: int, port: str) -> Tuple[Tuple[str, ...], ...]:
    """Endpoints of a group whose hierarchical port `i` feeds kernel `kk[i]` alone."""
    return tuple((f'kk[{index}].{port}',) for index in range(int(count)))


@dataclass(frozen=True)
class PortMap:
    """Port contract for one op implementation instance."""

    inputs: Dict[str, PortBinding]
    outputs: Dict[str, PortBinding]


def to_plain(value: Any):
    if isinstance(value, StagingDescriptor):
        return to_plain(value.to_dict())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, '__dataclass_fields__'):
        return {k: to_plain(getattr(value, k)) for k in value.__dataclass_fields__}
    return value
