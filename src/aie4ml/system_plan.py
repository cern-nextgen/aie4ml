# Copyright 2025 D. Danopoulos, aie4ml
# SPDX-License-Identifier: Apache-2.0

"""System-level (PL data mover + host) emission planning.

The AIE array project is always emitted. When the user selects ``target='hardware'`` on the
frontend, the project additionally emits the PL data mover, the v++ linker connectivity, and
the XRT host program so the design can run on a board (not just simulate the array).
``emits_system`` is the single gate consulted by the writer and ``AIEModel`` for that opt-in.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .ir import get_backend_context
from .passes.utils import sanitize_identifier

# Bytes in one 512-bit DDR/AXI word -- the unit the PL data mover transfers.
# This is a transport constant (matches the kernel's ap_uint<512> m_axi word); it is
# independent of the element dtype (int8/int16/int32 just change how many elements fit
# per word, not the word size).
_DDR_WORD_BYTES = 64

# The data mover runs iterations in fixed-size groups, so the usable iteration count must
# be a whole multiple of this
_ITERATIONS_PER_GROUP = 8

# Fraction of the on-chip RAM pool the planner is allowed to spend on preload/stream buffers.
# The model assumes perfect packing, but HLS's adds some overhead
# Budgeting only 80% of the pool leaves headroom for the overhead
_PL_USABLE_FRACTION = 0.8


def _max_preloadable_iterations(
    ctx, pl_memory: str, n_ifm: int, n_ofm: int, ifm_per_stream: int, ofm_per_stream: int
) -> int:
    """Largest n_iter whose benchmark preload buffers fit the PL on-chip pool (URAM or BRAM,
    whichever PLMemory selects), rounded down to a whole group of _ITERATIONS_PER_GROUP.

    Counts BLOCKS, not bytes (see _onchip_blocks): the preload buffers store 512-bit words, so
    each per-stream bank is width-pinned and its depth (per_stream * n_iter) rounds up to the
    block depth. A byte budget would over-count and hand back an n_iter that then over-utilizes
    at v++ link. Benchmark-only: that mover stages every iteration on-chip (one buffer per
    stream, no ping-pong -> copies=1); the memory_stream movers stream instead.
    """
    avail_blocks, depth, width, _ = _pl_pool(ctx, pl_memory)

    def _blocks(n_iter: int) -> int:
        return _onchip_blocks(512, ifm_per_stream * n_iter, n_ifm, depth, width, copies=1) + _onchip_blocks(
            512, ofm_per_stream * n_iter, n_ofm, depth, width, copies=1
        )

    # Grow by whole groups while they still fit (block count is monotonic in n_iter).
    possible_iters = 0
    while _blocks(possible_iters + _ITERATIONS_PER_GROUP) <= avail_blocks:
        possible_iters += _ITERATIONS_PER_GROUP
    # Error if not even one group fits -- per-iteration footprint too large for this device.
    if possible_iters < _ITERATIONS_PER_GROUP:
        raise RuntimeError(
            f'PL on-chip pool ({avail_blocks} blocks) cannot hold one group of '
            f'{_ITERATIONS_PER_GROUP} preloaded iterations (needs {_blocks(_ITERATIONS_PER_GROUP)} '
            f'blocks); the per-iteration IO footprint is too large for this device.'
        )
    return possible_iters


# A 512-bit data-mover word is WIDTH-PINNED to ceil(512/WidthBits)=8 blocks no matter how shallow it is
# which is why a shallow ping-pong buffer can exhaust the pool by block count even when its byte footprint
# looks tiny. So the budgets below count blocks, not bytes.


def _pl_pool(ctx, pl_memory: str):
    """(usable_blocks, depth, width_bits, label) for the on-chip RAM pool PLMemory selects, read
    from the device catalog (aie_devices.json -> ctx.device)."""
    label = 'BRAM' if pl_memory == 'bram' else 'URAM'
    pool = ctx.device.bram if pl_memory == 'bram' else ctx.device.uram
    if pool is None:
        raise RuntimeError(f'aie_devices.json gives no {label} pool for {ctx.device.platform}: no PL buffers to plan.')
    return int(pool.blocks * _PL_USABLE_FRACTION), pool.depth, pool.width_bits, label


def _onchip_blocks(
    word_bits: int, rows: int, n_banks: int, block_depth: int, block_width_bits: int, copies: int = 2
) -> int:
    """RAM blocks for `copies` ping-pong buffers, each `n_banks` independent banks of `rows`
    deep x `word_bits` wide. Width-pinned: ceil(word_bits/block_width_bits) blocks per bank,
    depth rounded up to block_depth."""
    width_blocks = math.ceil(int(word_bits) / int(block_width_bits))
    depth_blocks = math.ceil(int(rows) / int(block_depth))
    return int(copies) * int(n_banks) * width_blocks * depth_blocks


def _stream_buffer_blocks(ctx, pl_memory: str, n_ifm: int, n_ofm: int, ifm_per_stream: int, ofm_per_stream: int) -> int:
    """Total URAM/BRAM blocks the memory_stream ping-pong buffers occupy (mm2s + s2mm) in the
    pool PLMemory selects. mm2s = n_ifm banks x ifm_per_stream rows; s2mm = n_ofm x ofm_per_stream."""
    _, depth, width, _ = _pl_pool(ctx, pl_memory)
    return _onchip_blocks(512, ifm_per_stream, n_ifm, depth, width) + _onchip_blocks(
        512, ofm_per_stream, n_ofm, depth, width
    )


def _check_memory_stream_fits(
    ctx, pl_memory: str, n_ifm: int, n_ofm: int, ifm_per_stream: int, ofm_per_stream: int
) -> int:
    """If the memory_stream ping-pong buffers exceed
    the on-chip pool, counted in BLOCKS. Suggests the other pool when it would fit. Returns blocks."""
    avail, _, width, label = _pl_pool(ctx, pl_memory)
    needed = _stream_buffer_blocks(ctx, pl_memory, n_ifm, n_ofm, ifm_per_stream, ofm_per_stream)
    if needed > avail:
        other = 'bram' if pl_memory == 'uram' else 'uram'
        hint = ' Reduce the PLIO count (coarser slice) or target a larger device.'
        if (ctx.device.bram if other == 'bram' else ctx.device.uram) is not None:
            o_avail, _, _, o_label = _pl_pool(ctx, other)
            o_needed = _stream_buffer_blocks(ctx, other, n_ifm, n_ofm, ifm_per_stream, ofm_per_stream)
            if o_needed <= o_avail:
                hint = f" PLMemory='{other}' would fit ({o_needed}/{o_avail} {o_label}); try that."
        raise RuntimeError(
            f'memory_stream buffers need {needed} {label} blocks but only {avail} are '
            f'available (2 ping-pong x [{n_ifm} ifm + {n_ofm} ofm] banks; each 512-bit word '
            f'pins {math.ceil(512 / width)} blocks/bank).' + hint
        )
    return needed


# PL data-mover template locations. The benchmark mover lives under pl/benchmark/; the
# deployment movers (memory_stream / external_stream) live under pl/deployment/.
_BENCHMARK_KERNEL = 'ddr_pl_aie_datamover'
_BENCHMARK_TEMPLATE_DIR = 'pl/benchmark'
_DEPLOYMENT_TEMPLATE_DIR = 'pl/deployment'


# Hardware-emission vocabulary — the single source of truth both frontends validate against.
PL_TARGETS = ('aie', 'hardware')
PL_MEMORIES = ('uram', 'bram')
PL_DATA_MOVER_MODES = ('benchmark', 'memory_stream', 'external_stream')


def _one_of(cfg: Dict[str, Any], key: str, valid, default: str) -> None:
    value = str(cfg.get(key, default)).lower()
    if value not in valid:
        raise ValueError(f'AIEConfig.{key} must be one of {valid}, got {cfg.get(key)!r}.')
    cfg[key] = value


def normalize_pl_config(aie_config: Dict[str, Any]) -> Dict[str, Any]:
    _one_of(aie_config, 'Target', PL_TARGETS, 'aie')
    _one_of(aie_config, 'PLMemory', PL_MEMORIES, 'uram')
    _one_of(aie_config, 'PLDataMoverMode', PL_DATA_MOVER_MODES, 'benchmark')
    aie_config['EnablePLTiming'] = bool(aie_config.get('EnablePLTiming', False))
    return aie_config


def emits_system(model_or_ctx) -> bool:
    """Return True when PL + host code should be emitted (``target='hardware'``).

    Defaults to AIE-only emission (``target='aie'``) when the key is absent.
    """
    ctx = get_backend_context(model_or_ctx)
    return str(ctx.aie_config.get('Target', 'aie')).lower() == 'hardware'


def _single_io_feat(ports_map: Dict[str, Any], direction: str, batch: int) -> int:
    """Per-sample feature count of a single graph IO tensor.

    Supports exactly one graph input and one graph output (multiple graph I/O tensors are
    not yet supported).
    """
    tensors = list(ports_map)
    port0 = ports_map[tensors[0]][0]
    total = int(math.prod(port0.numpy_boundary_shape))
    if total % int(batch) != 0:
        raise RuntimeError(
            f'graph {direction} {tensors[0]!r}: element count {total} is not divisible by batch {batch}.'
        )
    return total // int(batch)


def port_transfer_bytes(port) -> int:
    """Bytes one port moves per iteration: its transfer, whatever the layout behind it."""
    return port.transfer_bytes


def _stream_words_512(port, direction: str) -> int:
    """512-bit words one PLIO stream carries per iteration, which must be word-aligned."""
    stream_bytes = port_transfer_bytes(port)
    if stream_bytes % _DDR_WORD_BYTES != 0:
        raise NotImplementedError(
            f'graph {direction}: {stream_bytes} bytes per stream is not a multiple of '
            f'{_DDR_WORD_BYTES} B/word; 512-bit/per-stream padding is not yet supported.'
        )
    return stream_bytes // _DDR_WORD_BYTES


def _kernel_entry(name: str, template_dir: str) -> Dict[str, str]:
    """One PL kernel for the writer (which templates to render) and the Makefile (.xo)."""
    return {
        'name': name,
        'cpp_template': f'{template_dir}/{name}.cpp.jinja',
        'cfg_template': f'{template_dir}/{name}.cfg.jinja',
    }


def build_pl_plan(mode: str, n_ifm: int, n_ofm: int, enable_pl_timing: bool) -> Dict[str, Any]:
    """Describe the PL data path for a given ``PLDataMoverMode``: which kernels to emit,
    the v++ linker connectivity (``nk=`` / ``sc=``), and the host's timing wiring.

    This is the single source of truth consumed by:
      * the writer            -> ``plan['kernels']`` (which .cpp/.cfg templates to render),
      * ``Makefile.jinja``    -> kernel ``.xo`` list + per-kernel build,
      * ``system.cfg.jinja``  -> ``plan['nk']`` / ``plan['sc']`` connectivity,
      * ``host.cpp.jinja``    -> ``plan['host']`` (timing CU + ``cycles_*`` register names).

    Modes:
      ``benchmark``     single combined ``ddr_pl_aie_datamover`` CU (today's design);
      ``memory_stream`` split ``mm2s`` + ``s2mm`` double-buffered CUs, plus a shared
                        ``tick_gen`` timer CU when PL timing is on (separate CUs have
                        independent ``ap_start``, so one timer gives a common time base);
      ``external_stream`` on-chip HLS ``traffic_gen`` source -> AIE -> ``s2mm`` (a
                        synthesizable stand-in for an external AXI producer); PL timing off.
    """
    mode = str(mode).lower()

    if mode == 'benchmark':
        # Benchmark is a cycle-accurate measurement harness -- the PL timers are its
        # whole point, so force them ON regardless of EnablePLTiming.
        if not enable_pl_timing:
            print("[aie4ml] PLDataMoverMode='benchmark' forces PL timers ON " '(overriding EnablePLTiming=False).')
            enable_pl_timing = True
        name = _BENCHMARK_KERNEL
        sc = [f'{name}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{name}.s_in_{s}' for s in range(n_ofm)]
        # The combined CU owns tick_gen internally; the host reads its 7 cycles_* regs.
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': [_kernel_entry(name, _BENCHMARK_TEMPLATE_DIR)],
            'nk': [f'{name}:1:{name}'],
            'sc': sc,
            'host': {
                'timing_kernel': name,
                'cycles': [
                    'preload_done',
                    'first_send',
                    'last_send',
                    'first_recv',
                    'last_recv',
                    'compute_done',
                    'total',
                ],
            },
        }

    if mode == 'memory_stream':
        ifm_k, ofm_k = 'mm2s', 's2mm'
        kernels = [_kernel_entry(ifm_k, _DEPLOYMENT_TEMPLATE_DIR), _kernel_entry(ofm_k, _DEPLOYMENT_TEMPLATE_DIR)]
        nk = [f'{ifm_k}:1:{ifm_k}', f'{ofm_k}:1:{ofm_k}']
        sc = [f'{ifm_k}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{ofm_k}.s_in_{s}' for s in range(n_ofm)]
        host = {'timing_kernel': None, 'cycles': []}
        if enable_pl_timing:
            timer = 'tick_gen'
            kernels.append(_kernel_entry(timer, _DEPLOYMENT_TEMPLATE_DIR))
            nk.append(f'{timer}:1:{timer}')
            # PL-to-PL event pulses give tick_gen a single time base across the two CUs.
            sc += [
                f'{ifm_k}.ev_first_send:{timer}.ev_first_send',
                f'{ifm_k}.ev_last_send:{timer}.ev_last_send',
                f'{ofm_k}.ev_first_recv:{timer}.ev_first_recv',
                f'{ofm_k}.ev_last_recv:{timer}.ev_last_recv',
                f'{ofm_k}.ev_done:{timer}.ev_done',
            ]
            host = {'timing_kernel': timer, 'cycles': ['first_send', 'last_send', 'first_recv', 'last_recv', 'total']}
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': kernels,
            'nk': nk,
            'sc': sc,
            'host': host,
        }

    if mode == 'external_stream':
        # The DDR->PLIO input mover (mm2s) is replaced by an ON-CHIP HLS traffic_gen
        # The output side is the unchanged s2mm mover so the host still reads the
        # result back / golden-checks it. On a deployed board,
        # On read deployment the traffic_gen is swapped for the user's producer
        # IP wired the same way (sc=<producer>.M_AXIS:ai_engine_0.PLIO_ifm_*).
        #
        # traffic_gen emits no timing event pulses, so there is no shared time base -- PL timing
        # is unavailable in this mode; the host measures wall-clock latency instead.
        if enable_pl_timing:
            print(
                "[aie4ml] PLDataMoverMode='external_stream' has no PL timers "
                '(the traffic_gen source emits no events); using host-side timing '
                '(overriding EnablePLTiming=True).'
            )
            enable_pl_timing = False
        src_k, ofm_k = 'traffic_gen', 's2mm'
        kernels = [_kernel_entry(src_k, _DEPLOYMENT_TEMPLATE_DIR), _kernel_entry(ofm_k, _DEPLOYMENT_TEMPLATE_DIR)]
        nk = [f'{src_k}:1:{src_k}', f'{ofm_k}:1:{ofm_k}']
        sc = [f'{src_k}.s_out_{s}:ai_engine_0.PLIO_ifm_{s}' for s in range(n_ifm)]
        sc += [f'ai_engine_0.PLIO_ofm_{s}:{ofm_k}.s_in_{s}' for s in range(n_ofm)]
        return {
            'mode': mode,
            'enable_pl_timing': enable_pl_timing,
            'kernels': kernels,
            'nk': nk,
            'sc': sc,
            'host': {'timing_kernel': None, 'cycles': []},
        }
    raise ValueError(f'unknown PLDataMoverMode {mode!r}.')


def build_system_io(model_or_ctx) -> Dict[str, Any]:
    """Project the resolved IR + physical plan into the flat variable bag the PL/host
    templates in ``templates/system/`` consume.

    Reuses :func:`aie4ml.simulation.build_io_layout` for per-PLIO-port boundary/dtype data and
    the dense/matmul execution instances for per-layer RTP (weight/bias) loading. Scope:
    single graph input + single graph output, 512-bit-aligned sizes.
    """
    from .simulation import build_io_layout

    ctx = get_backend_context(model_or_ctx)
    plan = ctx.ir.physical.plan
    if 'graph_input_count' not in plan or 'buffers' not in plan:
        raise RuntimeError('build_system_io requires a materialized physical plan; run the pipeline first.')

    layout = build_io_layout(ctx)
    n_ifm = int(plan['graph_input_count'])
    n_ofm = int(plan['graph_output_count'])
    if len(layout.inputs) != 1 or len(layout.outputs) != 1:
        raise RuntimeError(
            f'system I/O plan supports a single graph tensor; '
            f'got {len(layout.inputs)} and ({layout.outputs}). Multiple graph are not yet supported.'
        )
    batch = int(ctx.ir.logical.tensors[next(iter(layout.inputs))].shape[0])  # the input's leading axis

    in_feat = _single_io_feat(layout.inputs, 'input', batch)
    out_feat = _single_io_feat(layout.outputs, 'output', batch)
    # Per-PLIO-tile feature slices come from the GRAPH BOUNDARY ports -- the single graph
    # input and single graph output (already validated to be 1 each above).
    gin_port = next(iter(layout.inputs.values()))[0]
    gout_port = next(iter(layout.outputs.values()))[0]
    in_feat_slice = gin_port.tiling_dimension[gin_port.slice_dimension]
    out_feat_slice = gout_port.tiling_dimension[gout_port.slice_dimension]

    # Per-layer RTP artifacts (weights/bias/...). layer_index counts every executed node
    # (incl. param-less activations) so the RTP port suffix matches the app.cpp graph index.
    layers = []
    layer_index = 0
    for inst in ctx.ir.execution:
        node = inst.node
        layer_index += 1
        artifacts = inst.variant.get_artifacts(inst)
        if not artifacts:
            continue  # param-less op (e.g. activation) -- no kernel-resident RTP
        parallelism = getattr(inst.config, 'parallelism', None)
        if parallelism is None:
            raise RuntimeError(f'{node.name}: RTP-bearing layer has no parallelism config.')
        inst_name = sanitize_identifier(node.name)
        layers.append(
            {
                'inst_name': inst_name,
                'cas_num': int(parallelism.cas_num),
                'cas_length': int(parallelism.cas_length),
                'artifacts': [
                    {
                        'name': a['name'],
                        'kind': a['kind'],  # '1d' | '2d' -> RTP loop shape
                        'header': a['filename'],  # generated header to #include
                        'prefix': f"{a['name']}_{inst_name}",  # C symbol base (matches header)
                        'port': f"{a['port']}{layer_index}",  # ADF RTP port (name + layer idx, == app.cpp Lidx)
                    }
                    for a in artifacts
                ],
            }
        )
    # Top-level cas_* describe the graph-output-producing (last weight) layer.
    cas_num = layers[-1]['cas_num'] if layers else 1
    cas_length = layers[-1]['cas_length'] if layers else 1
    if in_feat % n_ifm != 0:
        raise NotImplementedError(
            f'in_feat {in_feat} not divisible by n_ifm {n_ifm}; uneven input shard is not yet supported.'
        )
    if out_feat % cas_num != 0:
        raise NotImplementedError(
            f'out_feat {out_feat} not divisible by cas_num {cas_num}; uneven output shard is not yet supported.'
        )
    ifm_per_stream = _stream_words_512(gin_port, 'input')
    ofm_per_stream = _stream_words_512(gout_port, 'output')
    iterations = int(ctx.aie_config['Iterations'])

    # On-chip pool selection -- common to every mover (all bind_storage to URAM or BRAM).
    pl_memory = str(ctx.aie_config.get('PLMemory', 'uram')).lower()
    pl_mem_impl = 'BRAM' if pl_memory == 'bram' else 'URAM'

    # PL data-path style (benchmark single CU vs split deployment movers).
    # Resolve the mode FIRST -- the on-chip budget below is mode-specific.
    pl_data_mover_mode = str(ctx.aie_config.get('PLDataMoverMode', 'benchmark')).lower()
    pl_plan = build_pl_plan(pl_data_mover_mode, n_ifm, n_ofm, bool(ctx.aie_config.get('EnablePLTiming', True)))
    enable_pl_timing = pl_plan['enable_pl_timing']

    # On-chip budget is mode-specific, but BOTH count in BLOCKS -- the buffers are 512-bit
    # words, so each per-stream bank is width-pinned to 8 URAM/BRAM blocks regardless of depth
    # (a byte budget badly under-counts):
    #   benchmark        preloads ALL n_iter on-chip -> cap n_iter to what the pool holds.
    #   memory_stream    fixed 2-deep ping-pong buffers (n_iter unbounded) -> just verify they fit.
    #   external_stream  like memory_stream but the input side is the on-chip traffic_gen, which
    #                    holds NO ping-pong buffer -> only the s2mm (ofm) buffers count.
    if pl_data_mover_mode == 'benchmark':
        max_n_iter = _max_preloadable_iterations(ctx, pl_memory, n_ifm, n_ofm, ifm_per_stream, ofm_per_stream)
    elif pl_data_mover_mode == 'external_stream':
        # traffic_gen streams on the fly (no input buffer): zero input banks AND rows so only
        # the s2mm (ofm) ping-pong is charged.
        _check_memory_stream_fits(ctx, pl_memory, 0, n_ofm, 0, ofm_per_stream)
        max_n_iter = iterations
    else:
        # memory_stream: not preload-capped; fail early if the ping-pong won't fit the pool.
        _check_memory_stream_fits(ctx, pl_memory, n_ifm, n_ofm, ifm_per_stream, ofm_per_stream)
        max_n_iter = iterations
    return {
        'project_name': ctx.project_config.project_name,
        'platform': ctx.device.platform,
        'graph_name': 'dut',
        'pl_freq_hz': float(ctx.aie_config['PLClockFreqMHz']) * 1e6,
        'plio_width_bits': int(ctx.device.plio_width_bits),
        'pl_mem_impl': pl_mem_impl,
        'enable_pl_timing': enable_pl_timing,
        'n_ifm': n_ifm,
        'n_ofm': n_ofm,
        'batch': batch,
        'iterations': iterations,
        'max_n_iter': max_n_iter,
        'in_feat': in_feat,
        'out_feat': out_feat,
        'in_feat_slice': in_feat_slice,
        'out_feat_slice': out_feat_slice,
        'cas_num': cas_num,
        'cas_length': cas_length,
        'ifm_per_stream': ifm_per_stream,
        'ofm_per_stream': ofm_per_stream,
        'layers': layers,
        'pl_data_mover_mode': pl_data_mover_mode,
        'pl_plan': pl_plan,
    }


# ---------------------------------------------------------------------------
# Host data.h generation (DDR-packed input) — target='hardware'
#
# The PL data movers (templates/firmware/pl/benchmark/ or pl/deployment/) move whole 512-bit
# DDR words and round-robin stripe them across the N per-direction PLIO streams
# (word i -> stream i % N). The host (host.cpp) replays this packed input each iteration.
# This produces `host/data.h` with the packed input and the IO word sizes.
# ---------------------------------------------------------------------------


def _pack_ports_to_ddr(port_tiles: List[np.ndarray], n_streams: int) -> np.ndarray:
    """Round-robin pack per-stream tiles into the data mover's 512-bit DDR layout.

    Packs each tile's RAW STORAGE BYTES (dtype-agnostic -- the PL data mover moves whole
    512-bit words and is type-agnostic; int8/int16/int32 just change how many elements fit
    per word). Returns a little-endian uint32 array (the host buffer element type), with
    contiguous DDR word i owned by stream i % n_streams. Assumes little-endian byte order
    (native on x86/Versal), consistent with the kernel's word reads.
    """
    word_blocks = []
    for tile in port_tiles:
        raw = np.frombuffer(np.ascontiguousarray(tile).tobytes(), dtype=np.uint8)
        if raw.size % _DDR_WORD_BYTES != 0:
            raise NotImplementedError(
                f'stream tile of {raw.size} B is not a multiple of {_DDR_WORD_BYTES} B; '
                f'512-bit padding is not yet supported.'
            )
        word_blocks.append(raw.reshape(-1, _DDR_WORD_BYTES))
    if len(word_blocks) != int(n_streams):
        raise RuntimeError(f'expected {n_streams} stream tiles, got {len(word_blocks)}.')
    words = word_blocks[0].shape[0]
    if any(block.shape[0] != words for block in word_blocks):
        raise NotImplementedError('uneven per-stream word counts; relay/padding is not yet supported.')

    mem = np.empty((words * int(n_streams), _DDR_WORD_BYTES), dtype=np.uint8)
    for stream, block in enumerate(word_blocks):
        mem[stream :: int(n_streams)] = block
    return np.frombuffer(mem.tobytes(), dtype='<u4')


def _check_storage_width(port, tile) -> None:
    """Fail hard if the prepared element width doesn't match the port's byte-aligned dtype.

    The packing is dtype-aware via raw bytes, but only for byte-aligned formats whose
    prepared NumPy dtype matches the boundary width (int8/int16/int32). Sub-byte widths or
    formats whose storage differs (e.g. bfloat16 prepared as float32) are rejected rather
    than silently corrupted.
    """
    if int(port.dtype.width) % 8 != 0:
        raise NotImplementedError(
            f'graph IO dtype {port.dtype.format!r} ({port.dtype.width}-bit) is not byte-aligned; '
            f'sub-byte PLIO packing is not supported.'
        )
    want = int(port.dtype.width) // 8
    got = int(tile.dtype.itemsize)
    if got != want:
        raise NotImplementedError(
            f'graph input {port.tensor!r} port {port.port}: prepared element is {got} B but boundary '
            f'dtype {port.dtype.format!r} is {want} B; dtype-exact packing for this format is unsupported.'
        )


def pack_host_data(model_or_ctx, X=None):
    """Return (ifm_packed, ofm_size_words) for one iteration, in the data mover DDR layout.

    Packs the quantized graph input into the DDR layout the host replays, and reports the
    output size (in 32-bit words) the host needs to size its output buffer. Both are
    graph-agnostic (no forward pass). X defaults to a deterministic pseudo-random input.

    dtype-aware: ``prepare_inputs`` quantizes X to the port's storage NumPy dtype
    (``dtype_for_precision(width, signed)`` for ints, float32 for floats) and the raw bytes
    of that are packed -- the same storage-dtype contract the weights path uses. n-D
    boundaries are handled via the full ``numpy_boundary_shape``.
    """
    from .simulation import _extract_port_tile, _framed, build_io_layout, prepare_inputs

    ctx = get_backend_context(model_or_ctx)
    layout = build_io_layout(ctx)
    in_tensor = next(iter(layout.inputs))
    in_ports = layout.inputs[in_tensor]

    boundary = in_ports[0].numpy_boundary_shape  # full n-D shape; no (batch, feat) assumption

    if X is None:
        X = np.random.default_rng(0).random(boundary, dtype=np.float64) * 2.0 - 1.0

    prepared = prepare_inputs(layout, X, iterations=1, quantize=True)[in_tensor]  # (1, *boundary)
    in_tiles = []
    for p in in_ports:
        tile = _extract_port_tile(prepared, p)[0]  # this port's slice, storage dtype, n-D
        _check_storage_width(p, tile)
        in_tiles.append(_framed(tile[np.newaxis], p))  # the transfer, with any framing padding
    ifm_packed = _pack_ports_to_ddr(in_tiles, len(in_ports))

    # What the graph actually emits, port by port -- the same transfer the PL movers carry, which
    # is not the logical tensor when a port's tile is padded.
    out_total_bytes = sum(port_transfer_bytes(port) for ports in layout.outputs.values() for port in ports)
    if out_total_bytes % 4 != 0:
        raise NotImplementedError(f'graph output is {out_total_bytes} B, not a multiple of 4 B (uint32 host buffer).')
    ofm_size_words = out_total_bytes // 4
    return ifm_packed, ofm_size_words


def host_data_context(model_or_ctx, X=None) -> Dict[str, Any]:
    """Prepare the ``host/data.h`` template context: the DDR-packed graph input and
    the IO word sizes the host needs.

    Python only prepares the values (masked to uint32); the writer renders
    ``templates/firmware/host/data.h.jinja``.
    """
    ifm_packed, ofm_size_words = pack_host_data(model_or_ctx, X)
    return {
        'ifm_packed': [int(v) & 0xFFFFFFFF for v in ifm_packed],
        'ifm_size_words': int(len(ifm_packed)),
        'ofm_size_words': int(ofm_size_words),
    }
