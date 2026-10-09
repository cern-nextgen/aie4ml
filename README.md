<p align="center">
  <img src="https://github.com/dimdano/aie4ml/blob/main/docs/aie4ml_logo_big.png" alt="aie4ml" width="600"/>
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/aie4ml.svg)](https://pypi.org/project/aie4ml/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.15946-b31b1b.svg)](https://arxiv.org/abs/2512.15946)

`aie4ml` is an end-to-end compiler that generates **optimized** AIE firmware automatically, which can be then built and simulated directly using **AMD Vitis**. It targets the **AMD AI Engine (AIE)** from model-level frontends and lowers supported operators into AIE graphs and kernels as a standalone AIE project.

- Current hardware targets: AIE1, AIE-ML and AIE-MLv2 devices.
- Current frontend paths: ONNX for explicit operator graphs, and the more user-friendly [`hls4ml`](https://github.com/fastmachinelearning/hls4ml) frontend path.
- Automatic firmware generation optimized for `resource`, `throughput`, or `latency` targets under a given AIE tile budget.

## Supported Operators

The full constraints are in [docs/support.md](docs/support.md).

| Operator | AIE1 | AIE-ML | AIE-MLv2 | Precision (inputs × weights) | Notes |
| --- | :---: | :---: | :---: | --- | --- |
| Dense (Gemm, MatMul with constant weights) | ✅ | ✅ | ✅ | int8 × int8, int16 × int8, float32 on all; int16 × int16 and bfloat16 on AIE-ML and AIE-MLv2; FP8 (E4M3) on AIE-MLv2 | Optional bias, fused ReLU. |
| MatMul (two activations) | ✅ | ✅ | ✅ | As Dense | The right operand is 2-D; it may be broadcast over the left operand's batch axes. |
| Conv2D (including grouped, depthwise and pointwise) | ✅ | ✅ | ✅ | int8 × int8 on all; int16 × int8 on AIE-ML and AIE-MLv2 (int8 or int16 output) | Batch 1, kernels up to 7×7, padding smaller than the kernel, any stride (int16 inputs: horizontal stride 1), no dilation. Fuses bias, ReLU and a following MaxPool. Depthwise runs a channelwise kernel on AIE-ML and AIE-MLv2. |
| MaxPool | ✅ | ✅ | ✅ | As its Conv2D | 2×2, stride 2, directly after a Conv2D (ReLU on either side). |
| BatchNormalization | ✅ | ✅ | ✅ | — | Folded into the preceding Conv2D or Dense (`QConv2DBatchnorm`, Dense + BatchNormalization, or a folded ONNX export). |
| ReLU | ✅ | ✅ | ✅ | — | Fused into the Dense or Conv2D. |
| Add | ✅ | ✅ | ✅ | Both inputs and the output of one type | Same shapes; no broadcasting. |
| LayerNorm | ✅ | ✅ | ✅ | int8 | Last axis. |
| Softmax | ✅ | ✅ | ✅ | int8 in, uint8 or int16 out | Last axis. Exact integer exponential (beta), or a faster surrogate for models trained with it. |
| Flatten / Reshape | ✅ | ✅ | ✅ | — | One sample to `[1, K]`, from a Conv2D into a Dense; no data is copied. |
| Transpose | ✖️ | ✅ | ✅ | — | Of the last two axes. |
| Slice, Split, Concat | ✅ | ✅ | ✅ | — | Along the boundaries of the producing layer's tiles; no data is copied. |

8-bit inputs against 16-bit weights (int8 × int16) are not supported.

## Prerequisites
- AMD Vitis 2026.1.1 and a valid AIE tools license.
  *(aie4ml tracks the newest AIE compiler; older releases may fail to compile some kernels.)*
- Python 3.10+.

## Installation

```bash
pip install "aie4ml[onnx]"     # ONNX frontend
pip install "aie4ml[hls4ml]"   # hls4ml frontend
```

Keras 3 and QKeras v3 models need hls4ml 1.4. Until it is released, install hls4ml from upstream commit:

```bash
pip install qkeras-v3 "tensorflow~=2.16.0" \
    "hls4ml @ git+https://github.com/fastmachinelearning/hls4ml.git@a2abb4d22e7762a870cd46ccf3e07ff6a5622ed5"
```

## Documentation & Tutorials

Documentation and usage: [https://github.com/dimdano/aie4ml](https://github.com/dimdano/aie4ml)

Tutorial 1: [`tutorials/tutorial_1.ipynb`](tutorials/tutorial_1.ipynb)
Tutorial 2: [`tutorials/tutorial_2.ipynb`](tutorials/tutorial_2.ipynb)

General `hls4ml` concepts: [https://fastmachinelearning.org/hls4ml](https://fastmachinelearning.org/hls4ml)


### Frontends

| Frontend | Models |
| --- | --- |
| ONNX (preferred) | Quantized operator graphs with QuantizeLinear/DequantizeLinear boundaries (QDQ). |
| hls4ml (user-friendly) | Keras 3 and QKeras v3: Dense, Conv2D, DepthwiseConv2D, QConv2DBatchnorm, MaxPooling2D, Flatten, ReLU activations and LayerNormalization. Split a SeparableConv2D into its depthwise and pointwise layers. |

## Maintainer

`aie4ml` is developed and maintained by [Dimitrios Danopoulos](https://github.com/dimdano).

## Citation

If `aie4ml` contributes to your research, please cite the corresponding publications:

```bibtex
@INPROCEEDINGS{11552717,
  author={Danopoulos, Dimitrios and Lupi, Enrico and Sun, Chang and Dittmeier, Sebastian and Kagan, Michael and Loncar, Vladimir and Pierini, Maurizio},
  booktitle={2026 IEEE 34th Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)},
  title={AIE4ML: An End-to-End Framework for Compiling Neural Networks for the Next Generation of AMD AI Engines},
  year={2026},
  volume={},
  number={},
  pages={176-184},
  keywords={Tiles;Modeling;Arrays;Kernel;Memory;Information rates;Throughput;System-on-chip;Loading;Engines;ai engines;hls4ml;aie4ml;versal;acceleration;inference},
  doi={10.1109/FCCM68464.2026.00035}}
```

```bibtex
@misc{danopoulos2026tamingexponentialfastsoftmax,
      title={Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference},
      author={Dimitrios Danopoulos and Enrico Lupi and Michael Kagan and Maurizio Pierini},
      year={2026},
      eprint={2604.02292},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.02292},
}
```
