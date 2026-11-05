// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention head with an additive bias on its logits: O = softmax(Q K^T + U) V, with Q, K and V projected from x
// as attention_head.h does.
//
// A chain's first tile reads its queries' rows of U, as attention_bias.h's does. A chain of two takes
// attention_head_values (attention_head.h) for its second tile.
//
// Buffers, as attention_head.h, and u as attention_bias.h.
#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "attention_head.h"
#include "parameters.h"

using namespace adf;

// A chain of one tile: all three projections, every key, the bias, and the division.
template <typename ConfigT>
class attention_head_bias_single {
  using bias_t = typename ConfigT::bias_t;
  ATTENTION_EXP
  ATTENTION_RECIP
  head_weights<ConfigT> wq_, wk_, wv_;
  head_bias<ConfigT> bq_, bk_, bv_;
  int first_block_;
  ATTENTION_HEAD_TOKENS(k_)
  ATTENTION_HEAD_TOKENS(v_)

public:
  attention_head_bias_single(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd,
                             attention_table<ConfigT> recip_ab, attention_table<ConfigT> recip_cd,
                             head_weights<ConfigT> wq, head_weights<ConfigT> wk, head_weights<ConfigT> wv,
                             head_bias<ConfigT> bq, head_bias<ConfigT> bk, head_bias<ConfigT> bv, int first_block)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), recip_ab_(recip_ab), recip_cd_(recip_cd), wq_(wq), wk_(wk), wv_(wv),
        bq_(bq), bk_(bk), bv_(bv), first_block_(first_block) {}
  void run(input_buffer<int8>& x0, input_buffer<int8>& x1, input_buffer<bias_t>& u, output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_head_bias_single::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(recip_ab_);
    REGISTER_PARAMETER(recip_cd_);
    REGISTER_PARAMETER(wq_);
    REGISTER_PARAMETER(wk_);
    REGISTER_PARAMETER(wv_);
    REGISTER_PARAMETER(bq_);
    REGISTER_PARAMETER(bk_);
    REGISTER_PARAMETER(bv_);
  }
};

// The first tile of a chain of two: Q and K, and the probabilities, the bias in their logits.
template <typename ConfigT>
class attention_head_bias_scores {
  using bias_t = typename ConfigT::bias_t;
  ATTENTION_EXP
  head_weights<ConfigT> wq_, wk_;
  head_bias<ConfigT> bq_, bk_;
  int first_block_;
  ATTENTION_HEAD_TOKENS(k_)

public:
  attention_head_bias_scores(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd,
                             head_weights<ConfigT> wq, head_weights<ConfigT> wk, head_bias<ConfigT> bq,
                             head_bias<ConfigT> bk, int first_block)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), wq_(wq), wk_(wk), bq_(bq), bk_(bk), first_block_(first_block) {}
  void run(input_buffer<int8>& x0, input_buffer<int8>& x1, input_buffer<bias_t>& u, output_cascade<acc32>* out);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_head_bias_scores::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(wq_);
    REGISTER_PARAMETER(wk_);
    REGISTER_PARAMETER(bq_);
    REGISTER_PARAMETER(bk_);
  }
};
