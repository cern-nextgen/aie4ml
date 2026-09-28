// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include "parameters.h"

using namespace adf;

// Depthwise Conv (one channel per group) on AIE-ML and AIE-MLv2: the Conv buffer ABI, channelwise products only.
template<typename ConfigT>
class depthwise_conv2d_single {
public:
  using data_t = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t = typename ConfigT::bias_t;

  depthwise_conv2d_single();

  void run(input_buffer<data_t>& ifm,
           const weight_t (&wts)[ConfigT::WN],
           const bias_t (&bias)[ConfigT::BN],
           output_buffer<result_t>& ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(depthwise_conv2d_single::run); }
};
