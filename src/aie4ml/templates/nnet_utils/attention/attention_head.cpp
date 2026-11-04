// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention head (attention_head.h): the projections here, the attention in attention_core.h.
#include "attention_core.h"
#include "attention_head.h"

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
  score_mm acc(aie::concat(rows, rows));
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

template <typename ConfigT>
void attention_head_single<ConfigT>::run(input_buffer<int8>& x0, input_buffer<int8>& x1, output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  project_tokens<ConfigT>(x0.data(), x1.data(), wk_, bk_, ConfigT::K_SHIFT, k_);
  project_tokens<ConfigT>(x0.data(), x1.data(), wv_, bv_, ConfigT::V_SHIFT, v_);
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    projected_queries<ConfigT>(x0.data(), x1.data(), wq_, bq_, first_block_ + b, qt);
    probabilities<ConfigT>(qt, k_, exp, p);
    weigh<ConfigT>(p, v_, recip, o.data() + 64 * S::ONJ * b);
  }
}

template <typename ConfigT>
void attention_head_scores<ConfigT>::run(input_buffer<int8>& x0, input_buffer<int8>& x1, output_cascade<acc32>* out)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  project_tokens<ConfigT>(x0.data(), x1.data(), wk_, bk_, ConfigT::K_SHIFT, k_);
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    projected_queries<ConfigT>(x0.data(), x1.data(), wq_, bq_, first_block_ + b, qt);
    probabilities<ConfigT, ConfigT::LOOKED_BLOCKS>(qt, k_, exp, p);
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) write_block(out, p[kb]);
  }
}

template <typename ConfigT>
void attention_head_values<ConfigT>::run(input_cascade<acc32>* in, input_buffer<int8>& x0, input_buffer<int8>& x1,
                                         output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  project_tokens<ConfigT>(x0.data(), x1.data(), wv_, bv_, ConfigT::V_SHIFT, v_);
  for (int b = 0; b < S::QB; ++b) {
    u8x64 p[S::KB];
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) {  // the gaps the first tile left, looked up here
      p[kb] = read_block(in);
      if (kb >= ConfigT::LOOKED_BLOCKS) p[kb] = lookup(exp, p[kb]);
    }
    weigh<ConfigT>(p, v_, recip, o.data() + 64 * S::ONJ * b);
  }
}
