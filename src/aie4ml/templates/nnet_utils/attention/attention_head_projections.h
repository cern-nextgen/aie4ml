// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// The projections every attention head kernel shares (attention_head.cpp, attention_head_bias.cpp): each the Dense it
// replaces, bit for bit (attention_head.h).
#pragma once
#include "attention_core.h"

namespace {

using i32x8 = aie::vector<int32, 8>;

// The 8x8 projection tile (token block b, head block j): x W + bias, rounded as the Dense rounds it.
template <typename C>
__attribute__((always_inline)) inline i8x64 project(const int8* __restrict x0, const int8* __restrict x1,
                                                    const int8* __restrict w, const int32* __restrict bias, int b,
                                                    int j, int shift)
{
  constexpr int HJ = C::MODEL_DIM / 16, NJ = shape<C>::NJ, KB = shape<C>::KB;  // HJ panels in a half of x
  const i32x8 b8 = aie::load_v<8>(bias + 8 * j);
  const auto rows = aie::from_vector<acc32>(aie::concat(b8, b8, b8, b8));  // the bias of 4 rows
  // x's tiles change every block: aie::mmul<8, 8, 8> shuffles them there, nothing to hoist (score_mm's split costs here)
  aie::mmul<8, 8, 8, int8, int8, acc32> acc(aie::concat(rows, rows));
  for (int f = 0; f < HJ; ++f) chess_unroll_loop(*)
    acc.mac(aie::load_v<64>(x0 + 64 * (KB * f + b)), aie::load_v<64>(w + 64 * (f * NJ + j)));
  for (int f = 0; f < HJ; ++f) chess_unroll_loop(*)
    acc.mac(aie::load_v<64>(x1 + 64 * (KB * f + b)), aie::load_v<64>(w + 64 * ((HJ + f) * NJ + j)));
  return acc.template to_vector<int8>(shift);
}

// Every token's projection, in the 8-row tiles the attention reads keys and values in.
template <typename C>
inline void project_tokens(const int8* __restrict x0, const int8* __restrict x1, const int8* __restrict w,
                           const int32* __restrict bias, int shift, int8* __restrict out)
{
  constexpr int NJ = shape<C>::NJ;
  for (int b = 0; b < shape<C>::KB; ++b)
    for (int j = 0; j < NJ; ++j) chess_unroll_loop(*)
      aie::store_v(out + 64 * (NJ * b + j), project<C>(x0, x1, w, bias, b, j, shift));
}

// Q^T of query block b.
template <typename C>
__attribute__((always_inline)) inline void projected_queries(const int8* __restrict x0, const int8* __restrict x1,
                                                             const int8* __restrict w, const int32* __restrict bias,
                                                             int b, i8x64 (&qt)[shape<C>::NJ])
{
  for (int j = 0; j < shape<C>::NJ; ++j) chess_unroll_loop(*)
    qt[j] = aie::transpose(project<C>(x0, x1, w, bias, b, j, C::Q_SHIFT), 8, 8);
}

}  // namespace
