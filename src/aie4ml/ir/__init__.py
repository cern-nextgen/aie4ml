"""Public exports for the AIE intermediate representation."""

from .context import (
    AIEBackendContext,
    DeviceSpec,
    ensure_backend_context,
    get_backend_context,
)
from .graph import (
    AIEPipelineIR,
    ExecutionInput,
    ExecutionInstance,
    ExecutionIR,
    ExecutionValue,
    ExecutionView,
    LogicalIR,
    OpNode,
    PhysicalIR,
    TensorVar,
    ViewPart,
    input_role,
    input_tensor_for_role,
    set_input_roles,
)

__all__ = [
    'AIEBackendContext',
    'AIEPipelineIR',
    'ExecutionInstance',
    'ExecutionInput',
    'ExecutionValue',
    'ExecutionView',
    'ViewPart',
    'ExecutionIR',
    'input_role',
    'input_tensor_for_role',
    'LogicalIR',
    'PhysicalIR',
    'OpNode',
    'set_input_roles',
    'TensorVar',
    'DeviceSpec',
    'ensure_backend_context',
    'get_backend_context',
]
