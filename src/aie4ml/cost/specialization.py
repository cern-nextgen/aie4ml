"""The identity of a compiled kernel: `key` (sources, config struct, role, part, compiler) names exactly the kernels
that compile alike. Coarser, for the cost model: `coordinate`, the kernel parameters its op type's descriptor names
(aie4ml.cost.variants); `group`, those of them that set its code; `features`, the others' values."""

from __future__ import annotations

import dataclasses
import enum
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..aie_types import AIEDataType
from .variants import DESCRIPTORS, NOT_CODE

# Bump when what a measurement records, or how a key is formed, changes.
SCHEMA_VERSION = 1

_TEMPLATES = Path(__file__).parent.parent / 'templates'


_COMPILER_DIRS = {'AIE': 'target', 'AIE-ML': 'target_aie_ml', 'AIE-MLV2': 'target_aie2ps'}


def installed_compiler(generation: str) -> Optional[str]:
    """The exact AIE compiler the installed Vitis ships for a generation; None without Vitis."""
    if 'XILINX_VITIS' not in os.environ:
        return None
    root = Path(os.environ['XILINX_VITIS']) / 'aietools' / 'tps' / 'lnx64' / _COMPILER_DIRS[generation]
    return (root / 'chessdir' / 'release_version').read_text().split()[1]


class CostUnavailable(RuntimeError):
    """No measurement or validated schedule case covers this kernel."""


def kernel_templates(kernel: str, parameters: str | None = None) -> str:
    """Fingerprint of what a kernel is compiled from: the sources in nnet_utils/`kernel`, its parameter
    template in firmware/variants/`parameters` (default: the same name), the shared headers and the compiler
    options (aie.cfg and the Makefile's flags)."""
    roots = [_TEMPLATES / 'nnet_utils' / kernel, _TEMPLATES / 'firmware' / 'variants' / (parameters or kernel)]
    if not all(root.is_dir() for root in roots):
        raise ValueError(f'{kernel!r}/{parameters!r} names no kernel with sources under {_TEMPLATES}.')
    files = [p for root in roots for p in sorted(root.rglob('*')) if p.is_file() and p.suffix != '.pyc']
    files += sorted((_TEMPLATES / 'nnet_utils').glob('*.h'))
    files += [_TEMPLATES / 'firmware' / 'aie.cfg.jinja', _TEMPLATES / 'firmware' / 'Makefile.jinja']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(_TEMPLATES).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class KernelSpecialization:
    """The kernels of one kernel graph as the compiler sees them: `chains` x `length` kernels compiled from
    nnet_utils/`kernel` against the `config` struct rendered at a fixed anchor, so placement stays out of it."""

    variant_id: str
    kernel: str
    parameters: str
    config: str
    chains: int
    length: int
    coordinate: Optional[str]  # canonical JSON; None when the variant declares no schedule parameters
    group: Optional[str]  # canonical JSON
    features: Optional[Dict[str, int]]

    @property
    def roles(self) -> Tuple[str, ...]:
        """Each kernel's cascade role along one chain."""
        if self.length == 1:
            return ('single',)
        return ('first',) + ('middle',) * (self.length - 2) + ('last',)

    def key(self, role: str, part: str, compiler: str) -> str:
        if role not in self.roles:
            raise ValueError(f'{self.variant_id}: a chain of {self.length} has no {role!r} kernel.')
        identity = [
            SCHEMA_VERSION,
            part,
            compiler,
            self.variant_id,
            role,
            kernel_templates(self.kernel, self.parameters),
            self.config,
        ]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]


def kernel_specialization(inst) -> KernelSpecialization:
    from ..writer import TEMPLATE_ROOT, kernel_config

    header = next((TEMPLATE_ROOT / 'nnet_utils').rglob(inst.graph_header), None)
    if header is None:
        raise ValueError(f'{inst.name}: no kernel sources hold its graph header {inst.graph_header!r}.')
    return KernelSpecialization(
        variant_id=inst.variant.variant_id,
        kernel=header.parent.name,
        parameters=inst.param_template,
        config=kernel_config(inst),
        chains=int(inst.config.parallelism.cas_num),
        length=int(inst.config.parallelism.cas_length),
        **_schedule(inst),
    )


def _schedule(inst) -> Dict[str, Any]:
    """The instance's cost features, the named workload fields of its complete kernel parameters, and its group:
    every other field, but those its op type's spec names as not setting the kernel's code."""
    spec = DESCRIPTORS.get(inst.variant.op_type)
    if spec is None:
        return {'coordinate': None, 'group': None, 'features': None}
    params = _flat(_canonical(inst.variant.kernel_params(inst.node, inst.config)))
    features = {path: int(params[path]) for path in spec.features}
    group = {
        path: value
        for path, value in params.items()
        if path not in features and not any(_under(path, pattern) for pattern in (*NOT_CODE, *spec.not_code))
    }
    return {
        'coordinate': json.dumps({**group, **features}, sort_keys=True),
        'group': json.dumps(group, sort_keys=True),
        'features': features,
    }


def _flat(value: Any, path: str = '') -> Dict[str, Any]:
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _flat(item, f'{path}.{key}' if path else key).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in _flat(item, f'{path}.{i}').items()}
    return {path: value}


def _under(path: str, pattern: str) -> bool:
    """Whether `path` is `pattern` or inside it, a `*` standing for one name."""
    parts, wanted = path.split('.'), pattern.split('.')
    return len(parts) >= len(wanted) and all(fnmatch.fnmatchcase(p, w) for p, w in zip(parts, wanted))


def _at(value: Any, path: str) -> Any:
    for name in path.split('.'):
        value = value[name] if isinstance(value, dict) else getattr(value, name)
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, AIEDataType):  # its storage width is what the kernel is compiled for
        return {**{f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}, 'width': value.width}
    if dataclasses.is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f'no canonical form for a schedule parameter of type {type(value).__name__}.')
