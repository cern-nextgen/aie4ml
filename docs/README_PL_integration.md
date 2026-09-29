# aie4ml — Hardware (PL + Host) Code Generation

This document describes the PL integration code-generation flow added to **aie4ml**: how the compiler emits the **PL (FPGA) data mover** and the **XRT host program**.

---

## Setup (Vitis + board environment)

The hardware / emulation build needs the Vitis tools (`v++`), the
aarch64 cross-compiler, XRT, and the board's rootfs / kernel image.
The `part` passed to a hardware build must name a Vitis `.xpfm` platform, not only a raw
AIE device part. A raw part is sufficient for standalone AIE compilation and simulation,
but it does not describe the programmable-logic resources and interfaces required by
`v++` system linking and packaging.

```bash
    source <>/2025.2/Vitis/settings64.sh
    export XILINX_VERSAL=<>/xilinx-versal-common-v2025.2/
    source $XILINX_VERSAL/environment-setup-cortexa72-cortexa53-amd-linux

    export ROOTFS=<>/xilinx-versal-common-v2025.2/rootfs.ext4
    export IMAGE=<>/xilinx-versal-common-v2025.2/Image
    export PLATFORM_REPO_PATHS=<>/2025.2/Vitis/base_platforms/
```
---

## 1. Opt-in: the `target` option

Hardware PL and Host is **opt-in** and does not affect the existing array/sim flow.

```python
aie_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=cfg, output_dir='proj', backend='aie',
    project_name='proj', batch_size=BATCH, iterations=ITERS, part=PLATFORM,
    target='hardware',          # 'aie' (default) = array-only; 'hardware' = + PL + host
)
```

- `target='aie'` (default): unchanged — emits only AIE source
- `target='hardware'`: additionally emits the PL data mover, connectivity, host, and a Makefile with hw and hw_emu compilation

The generated project still permits `make x86com`, `make x86sim`, and standalone AIE
compilation when its device is a raw part. `make hw` and `make hw_emu` fail explicitly until
the design is regenerated for a matching `.xpfm` platform.
---

## 2. What gets emitted

For `target='aie'` the project is exactly as before. For `target='hardware'` the project also
contains:

```
proj/
├── app.cpp, aie.cfg, src/…          # AIE array project (unchanged)
├── Makefile                         # unified: AIE/sim flow + hw/hw_emu flow
├── system.cfg                       # v++ -l connectivity (PL <-> AIE PLIO)
├── pl/
│   ├── ddr_pl_aie_datamover.cpp     # PL data mover (HLS)
│   └── ddr_pl_aie_datamover.cfg     # v++ -c (HLS->.xo) config
└── host/
    ├── host.cpp                     # XRT host program
    └── data.h                       # DDR-packed input + IO sizes
```

Hardware emission is **not** a compiler pass. It is a projection of the **already-built
physical plan** into extra files, performed by the writer when `emits_system(ctx)` is true.
The three-level IR, the optimization/transport passes, placement, and kernel selection are
unchanged.

---

## 3. Components

### 3.1 `target` option — `frontends/hls4ml/backend.py`
`create_initial_config` accepts `target='aie'` (default), validates it against
`{'aie','hardware'}`, and stores it as `AIEConfig['Target']`. (Extra converter kwargs were
already accepted, so the option plumbs straight through.)

### 3.2 Emission gate — `system_plan.py::emits_system`
```python
def emits_system(model_or_ctx) -> bool:
    ctx = get_backend_context(model_or_ctx)
    return str(ctx.aie_config.get('Target', 'aie')).lower() == 'hardware'
```
Consulted by the writer (whether to emit PL/host) and by the unified Makefile template
(whether to include the hardware targets).

### 3.3 System I/O projection — `system_plan.py::build_system_io`
This is the main method that collects the required information from different IRs.
The single function that projects the resolved IR + physical plan into the flat set of
variables the PL/host templates consume. It is **read-only** and reuses existing machinery:
per-PLIO-port data from `simulation.build_io_layout`, stream counts from the physical plan,
and per-layer weight/bias info from the resolved execution entries.

| Variable | Source | Used by |
|----------|--------|---------|
| `project_name`, `platform` | `ctx.project_config` / `ctx.device` | Makefile, system.cfg |
| `graph_name` (`"dut"`) | constant (matches the emitted ADF graph) | host |
| `pl_freq_hz` | `AIEConfig.PLClockFreqMHz` | system.cfg, host |
| `n_ifm` / `n_ofm` | `plan['graph_input_count']` / `['graph_output_count']` | data mover, system.cfg, host |
| `batch`, `in_feat`, `out_feat` | the graph input's leading axis + IO layout | data mover, host |
| `in_feat_slice` / `out_feat_slice` | `in_feat // n_ifm` / `out_feat // cas_num` | host |
| `cas_num` / `cas_length` | resolved dense parallelism | host RTP loops |
| `max_512_per_stream` | per-stream 64-byte word count | data mover PL memory sizing |
| `max_n_iter` | currently fixed | data mover, host |
| `layers[]` | per weight-bearing layer: headers, symbol prefixes, RTP port names, cas factors | host RTP weight/bias load |


### 3.4 Host input header — `system_plan.py::write_host_data_header`
Generates `host/data.h` containing:
- `ifm_packed` — the quantized graph input, packed into the data mover's DDR layout;
- `IFM_SIZE_WORDS` / `OFM_SIZE_WORDS` — IO sizes (32-bit words per iteration).

Both are graph-agnostic (no forward pass); the host replays `ifm_packed` and sizes its output
buffer from `OFM_SIZE_WORDS`.

---



## 5. Build & run

From Python:
```python
aie_model.build(make_target='hw')       # on-board: libadf.a -> .xo -> .xsa -> host -> sd_card
# or
aie_model.build(make_target='hw_emu')   # hardware emulation
```

From the CLI, in the generated project directory (after sourcing the Vitis/board environment):
```bash
make hw        # build for on-board hardware
make hw_emu    # build for hardware emulation (launch with: make run_emu)
```

Note: Hardware emulation has to be run manually and it not wired through Python.
```bash
make run_emu
# After the system boots in QEMU, run
./host.exe a.xclbin <ITERATIONS>
```

Instructions to run on hardware VEK280 board

* Copy all the files generated in `output/sd_card/*` to `/run/media/mmcblk0p1/`
* Reboot the board
* `cd /run/media/mmcblk0p1/` then run `./host.exe a.xclbin <ITERATIONS>`

---

## 6. Scope and limitations

Supported today:
- **single graph input and single graph output**;
- **multi-stream** per direction — a single tensor sharded across multiple PLIO streams
  (from the dense `cas_num × cas_length` parallelism) is fully handled;
- **multi-layer** graphs — per-layer weights/bias are loaded as RTP;
- **int8** I/O with **128-bit PLIO** and **512-bit-aligned** per-stream sizes.
- `benchmark`, `memory_stream`, and `external_stream` PL data-mover modes;
- configurable iteration counts, with the benchmark mode capped from the selected PL-memory
  block budget;
- ping-pong buffering in the `memory_stream` movers;
- hardware execution with host and optional PL timing;
- URAM or BRAM selection for PL-resident buffers (URAM has received the most testing).

Not yet supported:
- system linking from a raw AIE part; an `.xpfm` platform matching the target board is required;
- per-stream sizes that are not 512-bit aligned (would need padding);
- non-int8 element types / non-128-bit PLIO in the data mover lane math.
- hardware emulation execution is not automated via python
- Tiling in PL is not supported
- No golden verification. The host replays the input, runs the design, and reads the
  output back (it can dump it), but does not compare against a precomputed golden. (Functional
  verification of the array itself is done via the existing x86/AIE simulators.)

---

## 7. What did not change

The compiler core is untouched: the three-level IR, the pass pipeline (lowering, fusion,
transport, placement, materialization), kernel resolvers/variants, and the AIE array project
templates. Hardware support is an additive emission layer gated entirely behind
`target='hardware'`.
