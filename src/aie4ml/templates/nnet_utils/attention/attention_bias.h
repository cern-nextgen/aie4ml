// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention with an additive bias on its logits: O = softmax(Q K^T + U) V, as attention.h computes it.
//
// A chain's first tile reads its queries' rows of U, which start the scores' accumulator at the products' fixed point
// (BIAS_SHIFT): the logits and the bias round once, into the Softmax's input format. A chain of two takes
// attention_values (attention.h) for its second tile.
//
// Buffers, as attention.h, and
//   u  QUERY_BLOCKS x KEY_BLOCKS 8x8 tiles of bias_t (int8 or int16), row band by row band: each tile 8 queries by
//      8 keys
#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "attention.h"
#include "parameters.h"

using namespace adf;

// A chain of one tile: every key, the bias, and the division.
template <typename ConfigT>
class attention_bias_single {
  using bias_t = typename ConfigT::bias_t;
  ATTENTION_EXP
  ATTENTION_RECIP

public:
  attention_bias_single(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd,
                        attention_table<ConfigT> recip_ab, attention_table<ConfigT> recip_cd)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), recip_ab_(recip_ab), recip_cd_(recip_cd) {}
  void run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<int8>& v, input_buffer<bias_t>& u,
           output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_bias_single::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(recip_ab_);
    REGISTER_PARAMETER(recip_cd_);
  }
};

// The first tile of a chain of two: the probabilities, the bias in their logits.
template <typename ConfigT>
class attention_bias_scores {
  using bias_t = typename ConfigT::bias_t;
  ATTENTION_EXP

public:
  attention_bias_scores(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd)
      : exp_ab_(exp_ab), exp_cd_(exp_cd) {}
  void run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<bias_t>& u, output_cascade<acc32>* out);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_bias_scores::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
  }
};
