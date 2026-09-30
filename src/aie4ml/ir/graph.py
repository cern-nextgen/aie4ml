"""Intermediate representation for the aie4ml backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..aie_types import PrecisionIntent

if TYPE_CHECKING:  # pragma: no cover - runtime circular import guard
    from ..op_impls import OpImplVariant


@dataclass
class TensorVar:
    """Logical tensor value with producer/consumer connectivity.

    Parameter tensors (weights, biases) have data set and producer=None.
    Activation tensors have data=None and a producer node.
    """

    name: str
    shape: Tuple[int, ...]
    precision: Optional[PrecisionIntent] = None
    data: Optional[np.ndarray] = None
    producer: Optional['OpNode'] = None
    consumers: List['OpNode'] = field(default_factory=list)

    @property
    def is_parameter(self) -> bool:
        return self.data is not None


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


@dataclass
class TraitInstance:
    """Instance data for a trait that augments an IR node."""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpNode:
    """
    Logical IR node capturing semantic computation only.
    """

    name: str
    op_type: str

    inputs: List[TensorVar] = field(default_factory=list)
    outputs: List[TensorVar] = field(default_factory=list)

    traits: Dict[str, TraitInstance] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    directives: Dict[str, Any] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)

    @property
    def is_folded_view(self) -> bool:
        """A slice, split or concat folded into its readers: no kernel implements it."""
        return 'slice_view' in self.traits or 'concat_view' in self.traits

    def add_trait(self, trait: TraitInstance) -> None:
        self.traits[trait.name] = trait


def set_input_roles(node: OpNode, tensors: Sequence[TensorVar], role_names: Sequence[str]) -> None:
    if len(tensors) != len(role_names):
        raise ValueError(f'{node.name}: set_input_roles got {len(tensors)} tensors but {len(role_names)} role names.')
    _refuse_repeated_operands(node, [tensor.name for tensor in tensors])
    node.roles = {tensor.name: str(role) for tensor, role in zip(tensors, role_names)}


def _refuse_repeated_operands(node: OpNode, names: Sequence[str]) -> None:
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise NotImplementedError(f'{node.name}: reads {repeated} as more than one operand, which is not supported.')


def input_role(node: OpNode, tensor_name: str) -> Optional[str]:
    return node.roles.get(tensor_name)


def has_input_role(node: OpNode, role: str) -> bool:
    return role in node.roles.values()


def input_tensor_for_role(node: OpNode, role: str) -> TensorVar:
    for tensor in node.inputs:
        if node.roles.get(tensor.name) == role:
            return tensor
    raise ValueError(f'{node.name}: missing {role} tensor.')


@dataclass
class LogicalIR:
    """
    Tensor-centric logical IR graph.

    - tensors: all TensorVars in the graph
    - nodes:   ordered list of logical nodes
    """

    tensors: Dict[str, TensorVar] = field(default_factory=dict)
    nodes: List[OpNode] = field(default_factory=list)
    input_tensor_names: List[str] = field(default_factory=list)
    output_tensor_names: List[str] = field(default_factory=list)

    def add_tensor(self, tensor: TensorVar) -> None:
        if tensor.name in self.tensors:
            raise ValueError(f"Tensor '{tensor.name}' already exists.")
        self.tensors[tensor.name] = tensor

    def add_node(self, node: OpNode) -> None:
        self.nodes.append(node)

    def mark_graph_input(self, tensor_name: str) -> None:
        if tensor_name not in self.tensors:
            raise ValueError(f"Graph input tensor '{tensor_name}' does not exist.")
        if tensor_name not in self.input_tensor_names:
            self.input_tensor_names.append(tensor_name)

    def mark_graph_output(self, tensor_name: str) -> None:
        if tensor_name not in self.tensors:
            raise ValueError(f"Graph output tensor '{tensor_name}' does not exist.")
        if tensor_name not in self.output_tensor_names:
            self.output_tensor_names.append(tensor_name)

    def remove_node(self, node: OpNode, mode: str) -> None:
        """Remove a one-input, one-output node: `bypass` keeps its input, `contract` keeps its output."""
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise ValueError(
                f'Cannot remove node {node.name}: it has {len(node.inputs)} inputs and {len(node.outputs)} outputs.'
            )
        if mode == 'bypass':
            self._bypass_node(node)
        elif mode == 'contract':
            in_tv = node.inputs[0]
            if len(in_tv.consumers) != 1:
                raise ValueError(
                    f'Cannot contract node {node.name}: input tensor {in_tv.name} has {len(in_tv.consumers)} consumers.'
                )
            self._contract_node(node)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        if node in self.nodes:
            self.nodes.remove(node)
        node.inputs.clear()
        node.outputs.clear()

    def _bypass_node(self, node: OpNode):
        in_tv, out_tv = node.inputs[0], node.outputs[0]

        for consumer in list(out_tv.consumers):
            for i, inp in enumerate(consumer.inputs):
                if inp is out_tv:
                    consumer.inputs[i] = in_tv
            if consumer not in in_tv.consumers:
                in_tv.consumers.append(consumer)
            if out_tv.name in consumer.roles:
                consumer.roles[in_tv.name] = consumer.roles.pop(out_tv.name)

        if node in in_tv.consumers:
            in_tv.consumers.remove(node)

        out_tv.producer = None
        out_tv.consumers.clear()
        self.tensors.pop(out_tv.name, None)

    def _retarget_boundary(self, old: 'TensorVar', new: 'TensorVar') -> None:
        for names in (self.input_tensor_names, self.output_tensor_names):
            if old.name in names:
                names[names.index(old.name)] = new.name

    def _contract_node(self, node: OpNode):
        in_tv, out_tv = node.inputs[0], node.outputs[0]
        producer = in_tv.producer

        if producer:
            for i, outp in enumerate(producer.outputs):
                if outp is in_tv:
                    producer.outputs[i] = out_tv
            out_tv.producer = producer
        else:
            out_tv.producer = None

        if node in in_tv.consumers:
            in_tv.consumers.remove(node)

        in_tv.producer = None
        if not in_tv.consumers:
            self._retarget_boundary(in_tv, out_tv)
            self.tensors.pop(in_tv.name, None)

    def graph_inputs(self) -> List[TensorVar]:
        if not self.input_tensor_names:
            raise ValueError('LogicalIR.graph_inputs requires explicit input_tensor_names.')
        return [self.tensors[name] for name in self.input_tensor_names]

    def graph_outputs(self) -> List[TensorVar]:
        if not self.output_tensor_names:
            raise ValueError('LogicalIR.graph_outputs requires explicit output_tensor_names.')
        return [self.tensors[name] for name in self.output_tensor_names]

    def verify(self) -> None:
        """Assert structural invariants."""
        self._verify_unique_node_names()
        for node in self.nodes:
            _refuse_repeated_operands(node, [tensor.name for tensor in node.inputs])
        self._verify_topological_order()
        self._verify_tensor_producers()
        self._verify_connectivity()
        self._verify_graph_boundaries()

    def _verify_unique_node_names(self) -> None:
        seen: set = set()
        for node in self.nodes:
            if node.name in seen:
                raise RuntimeError(f'LogicalIR has duplicate node name {node.name!r}.')
            seen.add(node.name)

    def _verify_topological_order(self) -> None:
        seen: set = set()
        for node in self.nodes:
            for tensor in node.inputs:
                if tensor.producer is not None and tensor.producer.name not in seen:
                    raise RuntimeError(
                        f'LogicalIR is not in topological order: {node.name!r} consumes {tensor.name!r} '
                        f'whose producer {tensor.producer.name!r} has not yet been visited.'
                    )
            seen.add(node.name)

    def _verify_tensor_producers(self) -> None:
        graph_input_names = set(self.input_tensor_names)
        for node in self.nodes:
            for tensor in node.inputs:
                if tensor.is_parameter:
                    continue
                if tensor.producer is None and tensor.name not in graph_input_names:
                    raise RuntimeError(
                        f'{node.name}: activation input {tensor.name!r} has no producer '
                        'and is not a declared graph input.'
                    )

    def _verify_connectivity(self) -> None:
        nodes = {id(node) for node in self.nodes}
        for node in self.nodes:
            for tensor in (*node.inputs, *node.outputs):
                if self.tensors.get(tensor.name) is not tensor:
                    raise RuntimeError(f'{node.name}: {tensor.name!r} is not the tensor registered under that name.')
            if not set(node.roles) <= {tensor.name for tensor in node.inputs}:
                raise RuntimeError(f'{node.name}: roles {node.roles} name tensors it does not read.')
            for tensor in node.inputs:
                if node not in tensor.consumers:
                    raise RuntimeError(f'{node.name}: reads {tensor.name!r}, which does not list it as a consumer.')
            for tensor in node.outputs:
                if tensor.producer is not node:
                    raise RuntimeError(
                        f'{node.name}: writes {tensor.name!r}, whose producer is '
                        f'{tensor.producer.name if tensor.producer else None!r}.'
                    )
        for tensor in self.tensors.values():
            if tensor.producer is not None and (tensor.is_parameter or id(tensor.producer) not in nodes):
                raise RuntimeError(f'{tensor.name}: its producer {tensor.producer.name!r} is not a node of this graph.')
            if len({id(consumer) for consumer in tensor.consumers}) != len(tensor.consumers):
                raise RuntimeError(f'{tensor.name}: lists a consumer more than once.')
            for consumer in tensor.consumers:
                if tensor not in consumer.inputs:
                    raise RuntimeError(f'{tensor.name}: lists {consumer.name!r} as a consumer, which does not read it.')
            if any(int(extent) <= 0 for extent in tensor.shape):
                raise RuntimeError(f'{tensor.name}: has a non-positive extent in {tuple(tensor.shape)}.')

    def _verify_graph_boundaries(self) -> None:
        for names in (self.input_tensor_names, self.output_tensor_names):
            if len(set(names)) != len(names):
                raise RuntimeError(f'graph boundary {names} names a tensor more than once.')
        for name in self.input_tensor_names:
            if name not in self.tensors:
                raise RuntimeError(f'graph input {name!r} is not a tensor of this graph.')
        for name in self.output_tensor_names:
            tensor = self.tensors.get(name)
            if tensor is None:
                raise RuntimeError(f'graph output {name!r} is not a tensor of this graph.')
            if tensor.producer is None and name not in self.input_tensor_names:
                raise RuntimeError(f'graph output {name!r} has no producer and is not a graph input.')

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


STAGING_CONTRACTS: frozenset = frozenset({'outer', 'inner'})
"""Compiler-wide vocabulary of valid 2D execution partition-axis contracts."""

ROUTE_MODES: frozenset = frozenset({'direct', 'memtile', 'auto'})
"""Valid `io_route` modes for an edge: a direct connection, a memory-tile stage, or the planner's choice."""


VIEW_FLATTEN_2D = 'flatten_2d'
"""Output view: the producer writes its result as one `[1, K]` row instead of its natural shape."""

TENSOR_LAYOUTS: frozenset = frozenset({'linear', 'tiled'})
"""Valid `layout:` directive names. Only a variant selector -- the layout itself is the
staging descriptor, not a name."""


@dataclass(frozen=True)
class TensorContract:
    """Resolved execution contract for a single tensor edge.

    `port_staging` is the layout: the staging each producer port writes, which a consumer may adopt
    port for port. `view` is the producer's view of the tensor, the geometry those stagings lay out.

    `contract` is the one scalar a consumer inherits directly -- the partition axis is not
    recoverable from a single descriptor. Everything else it reads out of `port_staging`.

    `inner_shards` is (shards, width) when the producer stores the inner axis shard by shard (see
    `shard_major_order`), and the descriptors address that stored order; None when they address the
    logical order. Only a consumer that adopts it may read the tensor.
    """

    contract: str  # one of STAGING_CONTRACTS: which axis cas_num partitions
    port_staging: Tuple[Dict[str, Any], ...] = ()
    inner_shards: Optional[Tuple[int, int]] = None
    view: Any = None  # TensorView


def shard_major_order(extent: int, shards: int, width: int) -> np.ndarray:
    """The logical index of each stored element of an inner axis stored shard by shard: logically runs of `width`
    elements dealt to `shards` shards in turn, stored with every run of shard 0 first, then shard 1, and so on."""
    if int(extent) % (int(shards) * int(width)):
        raise ValueError(f'An inner axis of {extent} elements does not hold whole rounds of {shards} x {width}.')
    return np.arange(int(extent)).reshape(-1, int(shards), int(width)).transpose(1, 0, 2).reshape(-1)


@dataclass(frozen=True)
class ExecutionInput:
    """One activation an execution instance reads. `shared_memory` requires the edge's no-DMA
    realisation; without it placement still draws the kernels together and the plan records its choice."""

    tensor: str
    role: Optional[str]
    shared_memory: bool = False


@dataclass(frozen=True)
class ViewPart:
    """`extent` elements along the view's axis of `source`, starting at `start`."""

    source: str
    start: int
    extent: int


@dataclass(frozen=True)
class ExecutionView:
    """A value no kernel writes, folded from a slice, split or concat (`kind`, the op `node`); transport maps
    its readers onto its sources. A slice or split output is one part, a window of its source starting at
    `start`; a concat places each whole source at its `start` in the value, in order along `axis`."""

    kind: str
    node: str
    axis: int
    parts: Tuple[ViewPart, ...]

    def __post_init__(self) -> None:
        if self.kind not in ('slice', 'split', 'concat') or not self.parts:
            raise ValueError(f'{self.node}: a {self.kind!r} view with parts {self.parts}.')
        if self.kind != 'concat' and len(self.parts) != 1:
            raise ValueError(f'{self.node}: a {self.kind} output is one window of its source, not {self.parts}.')
        if self.axis < 0 or any(part.start < 0 or part.extent <= 0 for part in self.parts):
            raise ValueError(f'{self.node}: view axis {self.axis} or parts {self.parts} out of range.')
        if self.kind == 'concat':
            ends = [0] + [part.start + part.extent for part in self.parts]
            if any(part.start != end for part, end in zip(self.parts, ends)):
                raise ValueError(f'{self.node}: concat parts {self.parts} do not tile the value in order.')

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(part.source for part in self.parts)


@dataclass(frozen=True)
class ExecutionValue:
    """One activation: written by `producer`, or else a folded `view`, or else a graph input."""

    name: str
    producer: Optional[str] = None
    view: Optional[ExecutionView] = None

    def __post_init__(self) -> None:
        if self.producer is not None and self.view is not None:
            raise ValueError(f'{self.name}: a value is written by a kernel or is a view, not both.')


@dataclass
class ExecutionInstance:
    """One kernel graph to build. `node` is read only: the logical node it implements, or an
    execution-only node for a kernel a lowering pass inserted. Connectivity is `inputs`/`outputs`."""

    node: OpNode
    variant: 'OpImplVariant'
    ports: Any
    io_route: Dict[str, Any]
    port_views: Dict[str, Any]  # the view of each value it reads or writes; config.io_views is the kernel's own
    config: Any
    graph_header: str
    graph_name: str
    param_template: str
    inputs: Tuple[ExecutionInput, ...] = ()
    outputs: Tuple[str, ...] = ()
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def op_type(self) -> str:
        return self.node.op_type

    def input(self, tensor: str) -> ExecutionInput:
        for item in self.inputs:
            if item.tensor == tensor:
                return item
        raise KeyError(f'{self.name}: reads no execution value {tensor!r}.')


@dataclass
class ExecutionIR:
    """The kernel graphs to build, in producer-before-consumer order. After resolution it is the only
    source of executable connectivity; the logical graph keeps the semantics."""

    instances: Dict[str, ExecutionInstance] = field(default_factory=dict)
    tensor_contracts: Dict[str, TensorContract] = field(default_factory=dict)
    values: Dict[str, ExecutionValue] = field(default_factory=dict)
    graph_inputs: Tuple[str, ...] = ()
    graph_outputs: Tuple[str, ...] = ()

    def add(self, inst: ExecutionInstance) -> None:
        if inst.name in self.instances:
            raise ValueError(f'{inst.name}: an execution instance of that name already exists.')
        self.instances[inst.name] = inst

    def insert_before(self, name: str, inst: ExecutionInstance) -> None:
        if inst.name in self.instances:
            raise ValueError(f'{inst.name}: an execution instance of that name already exists.')
        if name not in self.instances:
            raise KeyError(f'{name}: no execution instance to insert {inst.name} before.')
        ordered = {}
        for key, value in self.instances.items():
            if key == name:
                ordered[inst.name] = inst
            ordered[key] = value
        self.instances = ordered

    def add_value(self, value: ExecutionValue) -> None:
        if value.name in self.values:
            raise ValueError(f'{value.name}: the execution graph already has a value of that name.')
        self.values[value.name] = value

    def get(self, name: str) -> Optional[ExecutionInstance]:
        return self.instances.get(name)

    def clear(self) -> None:
        self.instances.clear()
        self.tensor_contracts.clear()
        self.values.clear()
        self.graph_inputs = ()
        self.graph_outputs = ()

    def verify(self) -> None:
        visited = set()
        for inst in self.instances.values():
            for item in inst.inputs:
                value = self.values.get(item.tensor)
                if value is None:
                    raise RuntimeError(f'{inst.name}: reads {item.tensor!r}, which is not an execution value.')
                if item.tensor not in inst.ports.inputs:
                    raise RuntimeError(f'{inst.name}: reads {item.tensor!r} but binds no port to it.')
                if value.producer is not None and value.producer not in visited:
                    raise RuntimeError(f'{inst.name}: reads {item.tensor!r} before {value.producer} writes it.')
                if value.producer is None and value.view is None and item.tensor not in self.graph_inputs:
                    raise RuntimeError(f'{inst.name}: reads {item.tensor!r}, which nothing produces.')
                if value.view is not None:
                    for name in value.view.sources:
                        source = self.values.get(name)
                        if source is None:
                            raise RuntimeError(
                                f'{inst.name}: reads the view {item.tensor!r} over {name!r}, which is not an '
                                'execution value.'
                            )
                        if source.producer is not None and source.producer not in visited:
                            raise RuntimeError(
                                f'{inst.name}: reads the view {item.tensor!r} over {name!r} before '
                                f'{source.producer} writes it.'
                            )
                if item.shared_memory and value.producer is None:
                    raise RuntimeError(
                        f'{inst.name}: requires {item.tensor!r} through shared memory, but no kernel writes it.'
                    )
            bound = {item.tensor for item in inst.inputs} | set(inst.outputs)
            if set(inst.port_views) != bound:
                raise RuntimeError(f'{inst.name}: has views for {sorted(inst.port_views)}, but binds {sorted(bound)}.')
            for name in inst.outputs:
                value = self.values.get(name)
                if value is None or value.producer != inst.name:
                    raise RuntimeError(f'{inst.name}: writes {name!r}, but the value names another producer.')
                if name not in inst.ports.outputs:
                    raise RuntimeError(f'{inst.name}: writes {name!r} but binds no port to it.')
            visited.add(inst.name)
        for name in self.graph_outputs:
            if name not in self.values:
                raise RuntimeError(f'graph output {name!r} is not an execution value.')

    def __iter__(self):
        return iter(self.instances.values())


@dataclass
class PhysicalIR:
    """Physical layer of the IR capturing placement and routing."""

    placements: Dict[str, Dict[str, int]] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.placements.clear()
        self.plan.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'placements': _deep_copy(self.placements),
            'plan': _deep_copy(self.plan),
        }


@dataclass
class AIEPipelineIR:
    """Three-level IR bundle shared across backend passes."""

    logical: LogicalIR = field(default_factory=LogicalIR)
    execution: ExecutionIR = field(default_factory=ExecutionIR)
    physical: PhysicalIR = field(default_factory=PhysicalIR)
    optimizer: Dict[str, Any] = field(default_factory=dict)  # the design search's choice, when it ran

    def reset(self) -> None:
        self.logical = LogicalIR()
        self.execution = ExecutionIR()
        self.physical = PhysicalIR()
        self.optimizer = {}
