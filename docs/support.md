# Support Reference

The [README](../README.md) lists what each generation supports. This page gives the details that matter when a model
is close to a limit: what each operator accepts, how it can be split across tiles, and what conversion refuses.
Anything outside these limits fails during conversion with a message naming the layer and the reason; nothing falls
back silently.

Quantization is static and per tensor. Scales are powers of two, so rescaling is a shift.

### Parallelism

Each compute kernel that is not fused (Dense/MatMul, Conv2D, etc.) can span several AI Engine tiles. The compiler chooses every layer's split for the whole model at once, keeping the hand-overs between layers direct where it can and falling back to a memory tile
where it must (`AIEConfig`):

- `Optimize: 'resource'` (default) uses the fewest tiles that fit.
- `Optimize: 'performance'` splits the layers with the most multiply-accumulates per tile first, within
  `MaxTiles` (default: the whole array). Multiply-accumulates are a proxy for time, not a timing model.

A layer's split can be fixed per layer (`LayerDirectives` in the ONNX config, or the hls4ml layer config); the
compiler keeps what is given and chooses the rest:

- `parallelism: {cas_length: L}` splits the reduction (input features or channels) over a chain of `L` tiles.
- `parallelism: {cas_num: C}` runs `C` chains side by side, each computing a share of the output features or
  channels (`contract: 'inner'`) or of the rows (`contract: 'outer'`).
- `ports: 'stream'`(exp) moves a Dense or a Conv2D over streams instead of memory buffers.

The choice is in the project's `aie_pipeline.json` (`optimizer`) and the `aie4ml.report` summary.

## Dense and MatMul

- **Formats** (inputs × weights): int8 × int8, int16 × int8 and float32 × float32 on every generation; int16 × int16
  and bfloat16 on AIE-ML and AIE-MLv2; FP8 E4M3 on AIE-MLv2. int8 × int16 is refused on every generation.
- **Dense** has constant weights, an optional bias and a fused ReLU. **MatMul** multiplies two activations; its
  right operand is 2-D and may be broadcast across the left operand's leading axes. A batched right operand is refused.
- **Parallelism**: `cas_length` splits the reduction over a cascade chain; `cas_num` runs parallel chains over the
  output features (`contract: 'inner'`) or the rows (`'outer'`). What a directive leaves open, the compiler chooses for
  the whole model (`AIEConfig.Optimize`, see the README).
- **Microtile**: each generation has a default mmul shape per format. `microtiling: {microtile_m, microtile_k,
  microtile_n}` picks another shape the generation supports; the error lists the allowed ones.
- **One sample**: a Dense whose rows fit one microtile row block (batch 1) runs a kernel that computes one row
  block instead of two, on the fewest-row microtile a following Dense reads directly (2 rows on AIE1 and AIE-ML, 4
  on AIE-MLv2), or on its producer's microtile where the generation offers it.
- **Ports**: buffers by default; `ports: 'stream'` moves each tile's padded block over core streams, for any split.
- **Batch**: `BatchSize` rows are padded to whole microtiles; the host pads the input and trims the output.

## Conv2D

- **Formats**: int8 activations × int8 weights on every generation. On AIE-ML and AIE-MLv2, activations may also be
  int16, in and out, so int16 and int8 layers can follow each other.
- **Window**: any kernel shape (validated up to 7×7), asymmetric zero padding smaller than the kernel, `groups`
  including depthwise, any stride, batch 1. Dilation is refused. An int16 input needs a horizontal stride of 1.
- **Fused into the conv**: bias, ReLU, a following 2×2 stride-2 MaxPool (ReLU on either side of it), and a Flatten
  into a Dense, which reads the conv's output directly. Split over output-channel chains, each chain flattens its
  own channels and the Dense orders its weight rows to match: a Dense with one cascade stage per chain reads each
  chain's slice directly, and any other split goes through a memory tile. Such a flattened output cannot be a graph
  output.
- **Layout**: channels-last, in blocks of 8 channels. A channel count that is not a multiple of 8 is padded with zeros.
- **Parallelism**: `cas_length` splits the input channels over a cascade and `cas_num` splits the output channels
  (`'inner'`) or the output rows (`'outer'`); every split must be whole 8-channel blocks or equal row bands. A row
  split whose window reads neighbouring rows must read from the graph input, since row bands overlap by the window
  height, and its output may feed only a 1×1 conv or the graph output. With a fused pool, each row band must hold
  whole pool windows.
- **Depthwise** (one channel per group): on AIE-ML and AIE-MLv2, an int8 layer with a horizontal stride of 1 and no
  fused pool or flatten runs a channelwise kernel, on one tile or split by rows. Otherwise, and on AIE1, it runs the
  general kernel on block-diagonal weights.
- **Memory**: each tile's input, output and weights must each fit one memory bank (8 KB on AIE1, 16 KB on AIE-ML
  and AIE-MLv2). The compiler splits a layer that does not fit; held by a directive to a split that does not, the
  conversion stops and says why.
- **Stride > 1**: runs on buffer ports. A small retiler kernel on the neighbouring tile regroups the input columns,
  which costs one extra tile and one pipeline stage of latency. A row-only stride needs no retiler. A strided conv
  split by rows must read from the graph input.
- **Graph boundary**: a buffer port carries one 8-channel block, so an input or output with more channels at the
  boundary needs a matching `cas_length` or `cas_num`, or stream ports.
- **Streams** (`ports: 'stream'`): one tile, int8, stride 1, no pool or flatten. They let a frame of several channel
  blocks cross the boundary through one port, at roughly two to four times the cycles of buffer ports.

## Add

Two inputs and the output of the same shape and type; no broadcasting. Parallel chains split the rows or the
features (`cas_num`); there is no cascade.

## LayerNorm

int8 in and out, over the last axis, with constant gamma and beta and a positive epsilon. `cas_num` splits the rows.

## Softmax

int8 in, uint8 (Q8) or int16 (Q15) out, over the last axis. The default computes the exponential exactly in integer
arithmetic (beta). `approximation` selects a faster surrogate, which is accurate only for a model trained with it thus needs QAT (HCCS approx).
`cas_num` splits the rows.

## Folded operators

These never become kernels of their own:

- **ReLU** folds into the Dense or Conv2D before it; a constant power-of-two scale folds into its output shift.
- **MaxPool** folds into the Conv2D before it, as above.
- **BatchNormalization** must already be folded into the preceding Conv2D or Dense: QKeras `QConv2DBatchnorm`, a
  float Dense followed by BatchNormalization (hls4ml folds both), or an ONNX export that folds Conv +
  BatchNormalization (PyTorch in eval mode, onnxruntime `quant_pre_process`). An unfolded BatchNormalization is
  refused, because its per-channel scale folds exactly only into float weights.
- **Flatten / Reshape** of one sample to `[1, K]` between a Conv2D and a Dense.
- **Transpose** of the last two axes.
- **Slice, Split and Concat** along the boundaries of the producing layer's tiles. A slice that cuts through a tile,
  or chained slices, are refused.

## Model structure and data movement

- Layers hand data over directly, tile to tile, when both sides agree on the layout. On AIE-ML and AIE-MLv2 a memory
  tile reorders it otherwise (one stage). AIE1 has no memory tile, so a layout mismatch there is refused.
- An output may feed several layers (branches, residual Adds); each consumer is planned separately.
- The graph input and output move over PLIO ports, split to match the first and last layers' tiles.
- Placement starts at the device's first column with PL interfaces, next to the PLIOs, and spreads into the columns
  before it (from column 1) only when the design does not fit otherwise; each column between a kernel and its PLIO
  adds about 8 cycles of latency. `AIEConfig: ColumnStart` fixes where the placement region starts.
- Not implemented: more than one memory-tile stage between two layers, and AIE-to-PL-to-AIE paths inside a model.

## Frontends

- **ONNX** (recommended): quantized QDQ graphs. Export convolutions channels-last, as `input [N,H,W,C] ->
  Transpose(0,3,1,2) -> Conv`; a graph whose input is NCHW is not reinterpreted. Opset 21 or later is needed for
  int16 QuantizeLinear.
- **hls4ml**: Keras 3 and QKeras v3 through hls4ml 1.4 (or the upstream commit in the README). Supported layers:
  Dense, Conv2D, DepthwiseConv2D, QConv2DBatchnorm, MaxPooling2D, Flatten, ReLU activations and LayerNormalization.
  A SeparableConv2D must be split into its depthwise and pointwise layers.
