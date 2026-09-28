# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""Backend context shared across AIE passes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .graph import AIEPipelineIR

CONTEXT_ATTR = '_aie_backend_context'


@dataclass
class BackendPolicies:
    """Policies steering graph lowering and transformation stages."""

    fusion: Dict[str, Any] = field(default_factory=dict)
    decomposition: Dict[str, Any] = field(default_factory=dict)
    pack: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    tensors_have_batch: bool = False


@dataclass(frozen=True)
class RamPool:
    """One PL on-chip RAM pool: its blocks, and one block's Depth x WidthBits."""

    blocks: int
    depth: int
    width_bits: int


@dataclass(frozen=True)
class DmaSpec:
    """One kind of DMA, as its architecture manual documents it: its buffer descriptors, split into pools by
    channel parity when `bd_pools` is 2, the axes one BD walks, and whether it fills zeros past the data."""

    bds: int
    bd_pools: int
    dimensions: int
    zero_padding: bool


@dataclass
class DeviceSpec:
    """Model-level device specification published to passes: the facts aie_devices.json gives for a platform
    or part (aie4ml.device_catalog), with the user's AIEConfig choices over them. Every fact is required."""

    platform: str
    part: str  # the device part the platform, or the part named, resolves to
    generation: str
    aie_clock_mhz: float  # the AIE array clock the compiler builds the part at
    pl_clock_mhz: float
    columns: int
    rows: int
    column_start: int  # the placement region's first column
    preferred_column_start: int  # where placement starts when the design fits from there
    row_start: int
    plio_width_bits: int
    core_stream_inputs: int
    core_stream_outputs: int
    stream_switch_width_bits: int
    cascade_width_bits: int
    bank_mem_bytes: int
    max_mem_in_ports: int
    max_mem_out_ports: int
    vector_bytes: int
    has_memtile: bool
    bank_count: int
    tile_mem_bytes: int
    cascade_layout: str
    # Words in flight between cascaded cores: the sender's output FIFO and the receiver's input FIFO; None where
    # the architecture manual does not document them.
    cascade_words_in_flight: Optional[int]
    aie_compiler_target: str
    # The PL on-chip pools data movers stage in, where the catalog gives them; system planning picks the one
    # PLMemory selects.
    uram: Optional[RamPool]
    bram: Optional[RamPool]
    tile_dma: DmaSpec
    memtile_dma: Optional[DmaSpec]  # None without memory tiles

    @classmethod
    def from_config(cls, platform: str, cfg: Dict[str, Any]) -> 'DeviceSpec':
        def require(source: Dict[str, Any], key: str) -> Any:
            if key not in source:
                raise KeyError(f'Device {platform!r} is missing "{key}".')
            return source[key]

        compiler_target = str(require(cfg, 'AIECompilerTarget')).lower()
        if compiler_target not in ('part', 'platform'):
            raise ValueError(f'Unsupported AIECompilerTarget {compiler_target!r}; expected "part" or "platform".')
        generation = str(require(cfg, 'Generation'))
        if generation not in ('AIE', 'AIE-ML', 'AIE-MLV2'):
            raise ValueError(f'Unknown AIE generation {generation!r}; expected "AIE", "AIE-ML" or "AIE-MLV2".')
        fifos = [key in cfg for key in ('CascadeOutputFifoDepth', 'CascadeInputFifoDepth')]
        if any(fifos) and not all(fifos):
            raise KeyError(f'Device {platform!r} gives one cascade FIFO depth without the other.')

        def dma(entry: Dict[str, Any]) -> DmaSpec:
            return DmaSpec(
                bds=int(require(entry, 'Bds')),
                bd_pools=int(require(entry, 'BdPools')),
                dimensions=int(require(entry, 'Dimensions')),
                zero_padding=bool(require(entry, 'ZeroPadding')),
            )

        def pool(key: str) -> Optional[RamPool]:
            if key not in cfg:
                return None
            entry = cfg[key]
            return RamPool(*(int(require(entry, name)) for name in ('Blocks', 'Depth', 'WidthBits')))

        return cls(
            platform=platform,
            part=str(require(cfg, 'Part')),
            generation=generation,
            aie_clock_mhz=float(require(cfg, 'AIEClockFreqMHz')),
            pl_clock_mhz=float(require(cfg, 'PLClockFreqMHz')),
            columns=int(require(cfg, 'Columns')),
            rows=int(require(cfg, 'Rows')),
            # A user's ColumnStart bounds the region. Otherwise it spans every column but 0 -- a kernel keeps
            # buffers in its west neighbour -- and placement prefers to start at the first shim column with a PL
            # interface, where the PLIOs sit next to the kernels.
            column_start=int(cfg.get('ColumnStart', 1)),
            preferred_column_start=int(cfg['ColumnStart'] if 'ColumnStart' in cfg else require(cfg, 'PLColumnStart')),
            row_start=int(require(cfg, 'RowStart')),
            plio_width_bits=int(require(cfg, 'PLIOWidthBits')),
            core_stream_inputs=int(require(cfg, 'CoreStreamInputs')),
            core_stream_outputs=int(require(cfg, 'CoreStreamOutputs')),
            stream_switch_width_bits=int(require(cfg, 'StreamSwitchWidthBits')),
            cascade_width_bits=int(require(cfg, 'CascadeWidthBits')),
            bank_mem_bytes=int(require(require(cfg, 'Memory'), 'BankMemBytes')),
            max_mem_in_ports=int(require(cfg, 'MaxMemTileInPorts')),
            max_mem_out_ports=int(require(cfg, 'MaxMemTileOutPorts')),
            vector_bytes=int(require(cfg, 'VectorBytes')),
            has_memtile=bool(require(cfg, 'HasMemTile')),
            bank_count=int(require(cfg, 'BankCount')),
            tile_mem_bytes=int(require(cfg, 'TileMemBytes')),
            cascade_layout=str(require(cfg, 'CascadeLayout')),
            cascade_words_in_flight=(
                int(cfg['CascadeOutputFifoDepth']) + int(cfg['CascadeInputFifoDepth']) if all(fifos) else None
            ),
            aie_compiler_target=compiler_target,
            uram=pool('UltraRAM'),
            bram=pool('BlockRAM'),
            tile_dma=dma(require(require(cfg, 'Dma'), 'Tile')),
            memtile_dma=dma(cfg['Dma']['MemTile']) if 'MemTile' in cfg['Dma'] else None,
        )


@dataclass
class ProjectConfig:
    """Project-level config populated during lowering; consumed by writer, simulation, and build."""

    output_dir: Path
    project_name: str
    stamp: Optional[str]
    custom_sources: Dict[str, str]


@dataclass
class AIEBackendContext:
    """Container carrying IR graph, device spec and policies."""

    device: DeviceSpec
    policies: BackendPolicies
    project_config: ProjectConfig
    aie_config: Dict[str, Any] = field(default_factory=dict)
    ir: AIEPipelineIR = field(default_factory=AIEPipelineIR)
    emitted: Optional[str] = None  # configuration_fingerprint() of the project last written

    def configuration_fingerprint(self) -> str:
        """A hash of what a project is generated from: the device, the AIE configuration, every node's directives
        and the design search's choice. Not of the whole project: the model's graph is fixed at conversion."""
        return hashlib.sha256(
            json.dumps(
                {
                    'device': [self.device.platform, self.device.part],
                    'aie_config': self.aie_config,
                    'directives': {node.name: node.directives for node in self.ir.logical},
                    'optimizer': self.ir.optimizer,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def reset_ir(self) -> None:
        self.ir.reset()
        self.emitted = None


def ensure_backend_context(model, factory: Callable[[], AIEBackendContext]) -> AIEBackendContext:
    """Return the shared backend context, creating it if needed."""
    ctx = getattr(model, CONTEXT_ATTR, None)
    if ctx is None:
        ctx = factory()
        setattr(model, CONTEXT_ATTR, ctx)
    return ctx


def get_backend_context(model_or_ctx: Union[Any, AIEBackendContext]) -> AIEBackendContext:
    if isinstance(model_or_ctx, AIEBackendContext):
        return model_or_ctx
    ctx = getattr(model_or_ctx, CONTEXT_ATTR, None)
    if ctx is None:
        raise RuntimeError('AIE backend context missing. Run lowering before invoking downstream passes.')
    return ctx
