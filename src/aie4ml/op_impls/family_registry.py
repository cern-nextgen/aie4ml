from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

from ..errors import ConfigRefused
from ..ir.graph import VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS, has_input_role
from .utils import requested_port_kinds

if TYPE_CHECKING:
    from .base import OpImplVariant

COMMON_DIRECTIVES = frozenset({'placement', 'io_route', 'ports'})
# What a kernel may fuse, as the passes ask (`fuses`) and `fusions_of` names a node's: a variant declares from these.
FUSIONS = frozenset(
    {'bias', 'relu', 'max_pool', 'average_pool', 'sum_pool', 'output_scale', 'input_scale', 'apply_alpha'}
)
OUTPUT_VIEWS = frozenset({VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS})


def fusions_of(node: Any) -> frozenset:
    """What `node` carries that its kernel must implement, named as variants declare it (`supported_fusions`)."""
    fused = {scale for scale in ('output_scale', 'input_scale') if node.traits.get(scale) is not None}
    if has_input_role(node, 'bias'):
        fused.add('bias')
    activation = node.traits.get('fused_activation')
    if activation is not None:
        fused.add(activation.data['activation'])
    pool = node.traits.get('fused_pool')
    if pool is not None:
        fused.add(f"{pool.data['kind']}_pool")
    return frozenset(fused)


class FamilyResolver(ABC):
    """Structural validator, variant dispatcher and capability record for one op type; passes ask
    capabilities instead of naming op types. Its capabilities are those some variant declares."""

    op_type: ClassVar[str] = ''

    @property
    def supported_fusions(self) -> frozenset:
        """The epilogues and absorbed scales some variant's kernels fuse."""
        return frozenset().union(*(variant.supported_fusions for variant in self._variants()))

    @property
    def supported_output_views(self) -> frozenset:
        """The output views (VIEW_FLATTEN_2D, VIEW_REGROUP_ROWS) some variant writes directly."""
        return frozenset().union(*(variant.supported_output_views for variant in self._variants()))

    def _variants(self):
        from .registry import get_op_impl_registry

        return get_op_impl_registry().candidates(self.op_type)

    def parallelism_candidates(self, _node: Any, _device: Any) -> Tuple[Dict[str, Any], ...]:
        """The `parallelism` directives a design search may resolve the node under; () leaves it to resolution,
        which follows its producer. Whether a candidate is legal is resolution's call, not this list's."""
        return ()

    def microtiling_candidates(self, _node: Any, _device: Any) -> Tuple[Dict[str, Any], ...]:
        """The `microtiling` directives a design search may resolve the node under, which differ only in the rows of
        their blocks; () leaves it to resolution. Legality is resolution's call."""
        return ()

    def fallback_kernels(self, _node: Any, _device: Any) -> Tuple[Dict[str, Any], ...]:
        """Directives resolving the node with a kernel its resolution prefers not to, which takes fewer tiles: a
        design search falls back on it where the preferred kernel's design does not fit the budget, or, optimizing
        speed, is slower."""
        return ()

    def macs(self, _node: Any) -> int:
        """Multiply-accumulates per inference as the model states the op, padding aside: what its throughput is
        rated by. 0 for an op that multiplies nothing."""
        return 0

    def spatial_access(self, _node: Any):
        """The 2-D window read around each output pixel (sizes its producer's frame), or None."""
        return None

    def reorder_reduction_rows(self, node: Any, tensor: Any, _order) -> None:
        """Adopt a folded view's permutation of the rows this op reduces over into its constants."""
        raise NotImplementedError(
            f'{node.name}: {self.op_type} cannot adopt a reordered {tensor.name!r}; its constants '
            'assume the original row order.'
        )

    @abstractmethod
    def validate_structure(self, node: Any, device: Any) -> None:
        """Refuse a node whose inputs, shapes or attributes no variant of the family implements."""

    def _check_output_view(self, node: Any) -> None:
        """A folded output view must be one this family writes; its shape is the output tensor's."""
        view = node.traits.get('output_view')
        if view is None:
            return
        if set(view.data) != {'kind'}:
            raise ValueError(f'{node.name}: an output_view holds exactly its kind, got {sorted(view.data)}.')
        if view.data['kind'] not in self.supported_output_views:
            raise ValueError(f"{node.name}: {self.op_type} does not write the output view {view.data['kind']!r}.")

    def _check_fused_activation(self, node: Any) -> None:
        fused = node.traits.get('fused_activation')
        if fused is None:
            return
        if set(fused.data) != {'activation'} or fused.data['activation'] not in self.supported_fusions:
            raise ValueError(f'{node.name}: {self.op_type} cannot fuse the activation {fused.data}.')

    def resolve(
        self, node: Any, device: Any, directives: Dict[str, Any], input_contracts: Dict[str, Any]
    ) -> Tuple[Any, OpImplVariant]:
        """The config and variant for `node` under `directives` (the user's intents and a search's choices), given
        the contracts its inputs arrive in."""
        from .registry import get_op_impl_registry

        self._check_output_view(node)
        self._check_fused_activation(node)
        self.validate_structure(node, device)
        ports = requested_port_kinds(node)
        on_ports = [
            variant
            for variant in get_op_impl_registry().candidates(self.op_type)
            if (variant.input_port_kind, variant.output_port_kind) == ports
        ]
        view = node.traits.get('output_view')
        needed = fusions_of(node) | ({view.data['kind']} if view is not None else set())
        capable = [v for v in on_ports if needed <= v.supported_fusions | v.supported_output_views]
        if on_ports and not capable:
            offered = frozenset().union(*(v.supported_fusions | v.supported_output_views for v in on_ports))
            raise ConfigRefused(
                f'{node.name}: no {self.op_type} variant on {ports[0]} input and {ports[1]} output ports implements '
                f'{sorted(needed - offered)}.'
            )
        matching = [variant for variant in capable if variant.matches(node, device, directives)]
        if not matching:
            raise ConfigRefused(
                f'{node.name}: no {self.op_type} variant matches (generation={device.generation!r}, ports={ports!r}).'
            )
        variant = matching[0]
        if len(matching) > 1 and matching[1].plevel == variant.plevel:
            raise RuntimeError(
                f'{node.name}: {variant.variant_id} and {matching[1].variant_id} both match at priority '
                f'{variant.plevel}; the choice would depend on import order.'
            )
        unsupported = sorted(set(directives) - COMMON_DIRECTIVES - variant.supported_directives)
        if unsupported:
            raise NotImplementedError(
                f'{node.name}: {variant.variant_id} does not implement the directive(s) {unsupported}; it '
                f'supports {sorted(COMMON_DIRECTIVES | variant.supported_directives)}.'
            )
        return variant.resolve(node, device, directives, input_contracts), variant


class FamilyResolverRegistry:
    def __init__(self):
        self._resolvers: dict[str, FamilyResolver] = {}

    def register(self, op_type: str, resolver: FamilyResolver) -> None:
        if op_type in self._resolvers:
            raise ValueError(f'a family resolver for op_type={op_type!r} is already registered.')
        self._resolvers[op_type] = resolver

    def get(self, op_type: str) -> FamilyResolver:
        resolver = self._resolvers.get(op_type)
        if resolver is None:
            raise NotImplementedError(f'No family resolver registered for op_type={op_type!r}.')
        return resolver

    def find(self, op_type: str) -> Optional[FamilyResolver]:
        """Like `get`, but None for an op no family implements (a view)."""
        return self._resolvers.get(op_type)


_GLOBAL_FAMILY_RESOLVER_REGISTRY = FamilyResolverRegistry()


def get_family_resolver_registry() -> FamilyResolverRegistry:
    return _GLOBAL_FAMILY_RESOLVER_REGISTRY


def fuses(node: Any, capability: str) -> bool:
    """Whether some variant of `node`'s family fuses `capability` or writes it as an output view; never for a graph
    input (no node) or an op no family implements."""
    family = _GLOBAL_FAMILY_RESOLVER_REGISTRY.find(node.op_type) if node is not None else None
    return family is not None and capability in family.supported_fusions | family.supported_output_views


def family_resolver(*op_types: str):
    def decorator(cls):
        instance = cls()
        for op_type in op_types:
            _GLOBAL_FAMILY_RESOLVER_REGISTRY.register(str(op_type), instance)
        return cls

    return decorator
