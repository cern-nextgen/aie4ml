// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Direct Conv2D with int8 weights as an implicit GEMM over the Dense mmul core. The input is a channel-blocked
// NHWC frame [CB][rows][cols][8] with a zero border (the kernel zero-fills it); every (ky, kx, cb)
// tap is one M x 8 x 8 mmul on a contiguous pixel window, accumulated over all taps with the
// Dense 2x2 (AIE) or 4x2 (AIE-ML/MLv2) register blocking. The output is the same frame layout for
// the next conv, or a Dense LHS row when the layer is flattened.
//
// A layer may span several tiles, in the Dense contract vocabulary: CAS_LENGTH tiles split the
// reduction (this tile owns CB of the input channel blocks and passes partial sums down the
// cascade), CAS_NUM chains split the output channel blocks (this tile owns NB of them).

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class conv2d_base {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;

  conv2d_base();
};

template<typename ConfigT>
class conv2d_single : public conv2d_base<ConfigT> {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;

  void run(input_buffer<data_t>&   ifm,
           const weight_t (&wts)[ConfigT::WN],
           const bias_t (&bias)[ConfigT::BN],
           output_buffer<result_t>& ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(conv2d_single::run); }
};

template<typename ConfigT>
class conv2d_first : public conv2d_base<ConfigT> {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using bias_t   = typename ConfigT::bias_t;
  using cascade_t = typename ConfigT::cascade_t;

  void run(input_buffer<data_t>&  ifm,
           const weight_t (&wts)[ConfigT::WN],
           const bias_t (&bias)[ConfigT::BN],
           output_cascade<cascade_t>* outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(conv2d_first::run); }
};

template<typename ConfigT>
class conv2d_middle : public conv2d_base<ConfigT> {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using cascade_t = typename ConfigT::cascade_t;

  void run(input_buffer<data_t>&  ifm,
           const weight_t (&wts)[ConfigT::WN],
           input_cascade<cascade_t>*  inCascade,
           output_cascade<cascade_t>* outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(conv2d_middle::run); }
};

template<typename ConfigT>
class conv2d_last : public conv2d_base<ConfigT> {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using cascade_t = typename ConfigT::cascade_t;

  void run(input_buffer<data_t>&  ifm,
           const weight_t (&wts)[ConfigT::WN],
           input_cascade<cascade_t>* inCascade,
           output_buffer<result_t>& ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(conv2d_last::run); }
};
