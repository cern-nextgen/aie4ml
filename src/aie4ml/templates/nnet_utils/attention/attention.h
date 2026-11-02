// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention of one head: O = softmax(Q K^T) V, with the probabilities left unnormalised.
//
// CAS_NUM chains split the queries. A chain is one tile, or a tile that scores the keys and hands the probabilities
// down the cascade to one that weighs the values and divides.
//
// Not bit-exact with the QDQ graph: the graph rounds every normalised probability to uint8 before P V; the kernel
// keeps them unnormalised (exp(0) = 255) and divides each output once. An output can differ from the graph's by a code
// or so; it is no less accurate (it rounds once where the graph rounds twice), and far cheaper (one division per output
// instead of one per probability).
//
// Buffers, 8x8 int8 tiles row band by row band (a Dense output's microtiled layout), each tile one 8x8 microtile or
// two 4x8 ones (Q_ROWS, K_ROWS, V_ROWS, O_ROWS), whichever its producer or consumer stores:
//   q  QUERY_BLOCKS x HEAD_DIM/8 tiles, the chain's queries
//   k  KEY_BLOCKS x HEAD_DIM/8 tiles, every key in order
//   v  like k, every value
//   o  like q
// Cascade, per query block: P^T, a key block (64 uint8) a word. The exp and reciprocal tables are compiled in, two
// copies each (four lookups a cycle): registered parameters keep no lock, where an RTP port takes one.
#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template <typename ConfigT>
using attention_table = int16 (&)[ConfigT::TABLE_SIZE];

#define ATTENTION_EXP attention_table<ConfigT> exp_ab_, exp_cd_;
#define ATTENTION_RECIP attention_table<ConfigT> recip_ab_, recip_cd_;

// A chain of one tile: every key, and the division.
template <typename ConfigT>
class attention_single {
  ATTENTION_EXP
  ATTENTION_RECIP

public:
  attention_single(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd, attention_table<ConfigT> recip_ab,
                   attention_table<ConfigT> recip_cd)
      : exp_ab_(exp_ab), exp_cd_(exp_cd), recip_ab_(recip_ab), recip_cd_(recip_cd) {}
  void run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<int8>& v, output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_single::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
    REGISTER_PARAMETER(recip_ab_);
    REGISTER_PARAMETER(recip_cd_);
  }
};

// The first tile of a chain of two: the probabilities.
template <typename ConfigT>
class attention_scores {
  ATTENTION_EXP

public:
  attention_scores(attention_table<ConfigT> exp_ab, attention_table<ConfigT> exp_cd) : exp_ab_(exp_ab), exp_cd_(exp_cd) {}
  void run(input_buffer<int8>& q, input_buffer<int8>& k, output_cascade<acc32>* out);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_scores::run);
    REGISTER_PARAMETER(exp_ab_);
    REGISTER_PARAMETER(exp_cd_);
  }
};

// The second: P V, and the division.
template <typename ConfigT>
class attention_values {
  ATTENTION_RECIP

public:
  attention_values(attention_table<ConfigT> recip_ab, attention_table<ConfigT> recip_cd)
      : recip_ab_(recip_ab), recip_cd_(recip_cd) {}
  void run(input_cascade<acc32>* in, input_buffer<int8>& v, output_buffer<int8>& o);
  static void registerKernelClass()
  {
    REGISTER_FUNCTION(attention_values::run);
    REGISTER_PARAMETER(recip_ab_);
    REGISTER_PARAMETER(recip_cd_);
  }
};
