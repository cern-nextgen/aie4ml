// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention head: O = softmax(Q K^T) V, with Q, K and V projected from x in the kernel (attention.h for the
// attention itself).
//
// CAS_NUM chains split the queries. A chain is one tile, or a tile that projects Q and K and scores the keys, handing
// down the cascade the probabilities of the first LOOKED_BLOCKS key blocks and the gaps of the rest, to one that
// looks those up, projects V, weighs the values and divides. Every tile projects every token's K or V itself, so
// nothing is handed over between the projections and the attention.
//
// Each projection is the Dense it replaces, bit for bit: x W plus an int32 bias at the products' fixed point, rounded
// half to even into int8 by Q_SHIFT, K_SHIFT or V_SHIFT.
//
// Buffers:
//   x0, x1  the first and second half of x's columns, one input DMA channel each, in panels of 8: each panel every
//           token's 8 int8 in token order (a tile DMA writes them from a PLIO in one walk)
//   o       the chain's queries' outputs, as attention.h
// Compiled in: each W as 8x8 tiles, (model block f, head block j) at f * HEAD_DIM/8 + j; each bias HEAD_DIM int32.
#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "attention.h"
#include "parameters.h"

using namespace adf;

template <typename ConfigT>
using head_weights = int8 (&)[ConfigT::MODEL_DIM * ConfigT::HEAD_DIM];
template <typename ConfigT>
using head_bias = int32 (&)[ConfigT::HEAD_DIM];

#define ATTENTION_HEAD_TOKENS(name) \
  alignas(aie::vector_decl_align) int8 name[ConfigT::KEY_BLOCKS * ConfigT::HEAD_DIM * 8];

// A chain of one tile: all three projections, every key, and the division.
template <typename ConfigT>
class attention_head_single {
  ATTENTION_EXP
  ATTENTION_RECIP
  head_weights<ConfigT> wq_, wk_, wv_;
  head_bias<ConfigT> bq_, bk_, bv_;
  int first_block_;
  ATTENTION_HEAD_TOKENS(k_)
  ATTENTION_HEAD_TOKENS(v_)

public:
  attention_head_single(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd,
                        attention_table<ConfigT> recip_ab, attention_table<ConfigT> recip_cd,
                        head_weights<ConfigT> wq, head_weights<ConfigT> wk, head_weights<ConfigT> wv,
                        head_bias<ConfigT> bq, head_bias<ConfigT> bk, head_bias<ConfigT> bv, int first_block)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), recip_ab_(recip_ab), recip_cd_(recip_cd), wq_(wq), wk_(wk), wv_(wv),
        bq_(bq), bk_(bk), bv_(bv), first_block_(first_block) {}
  void run(input_buffer<int8>& x0, input_buffer<int8>& x1, output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_head_single::run);
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

// The first tile of a chain of two: Q and K, and the probabilities.
template <typename ConfigT>
class attention_head_scores {
  ATTENTION_EXP
  head_weights<ConfigT> wq_, wk_;
  head_bias<ConfigT> bq_, bk_;
  int first_block_;
  ATTENTION_HEAD_TOKENS(k_)

public:
  attention_head_scores(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd, head_weights<ConfigT> wq,
                        head_weights<ConfigT> wk, head_bias<ConfigT> bq, head_bias<ConfigT> bk, int first_block)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), wq_(wq), wk_(wk), bq_(bq), bk_(bk), first_block_(first_block) {}
  void run(input_buffer<int8>& x0, input_buffer<int8>& x1, output_cascade<acc32>* out);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_head_scores::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(wq_);
    REGISTER_PARAMETER(wk_);
    REGISTER_PARAMETER(bq_);
    REGISTER_PARAMETER(bk_);
  }
};

// The second: V, the probabilities of the key blocks past LOOKED_BLOCKS, P V, and the division.
template <typename ConfigT>
class attention_head_values {
  ATTENTION_EXP
  ATTENTION_RECIP
  head_weights<ConfigT> wv_;
  head_bias<ConfigT> bv_;
  ATTENTION_HEAD_TOKENS(v_)

public:
  attention_head_values(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd,
                        attention_table<ConfigT> recip_ab, attention_table<ConfigT> recip_cd,
                        head_weights<ConfigT> wv, head_bias<ConfigT> bv)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), recip_ab_(recip_ab), recip_cd_(recip_cd), wv_(wv), bv_(bv) {}
  void run(input_cascade<acc32>* in, input_buffer<int8>& x0, input_buffer<int8>& x1, output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_head_values::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(recip_ab_);
    REGISTER_PARAMETER(recip_cd_);
    REGISTER_PARAMETER(wv_);
    REGISTER_PARAMETER(bv_);
  }
};
