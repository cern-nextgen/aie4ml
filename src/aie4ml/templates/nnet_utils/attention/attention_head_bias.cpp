// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention head with an additive bias (attention_head_bias.h): the projections in attention_head_projections.h,
// the attention in attention_core.h.
#include "attention_head_projections.h"
#include "attention_head_bias.h"

template <typename ConfigT>
void attention_head_bias_single<ConfigT>::run(input_buffer<int8>& x0, input_buffer<int8>& x1, input_buffer<bias_t>& u,
                                              output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  project_tokens<ConfigT>(x0.data(), x1.data(), wk_, bk_, ConfigT::K_SHIFT, k_);
  project_tokens<ConfigT>(x0.data(), x1.data(), wv_, bv_, ConfigT::V_SHIFT, v_);
  reread<int8> keys{k_}, values{v_}, wq{wq_};
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    projected_queries<ConfigT>(x0.data(), x1.data(), wq(), bq_, first_block_ + b, qt);
    probabilities<ConfigT, S::KB>(qt, keys(), exp, p, u.data() + 64 * S::KB * b);
    weigh<ConfigT>(p, values(), recip, o.data() + 64 * S::ONJ * b);
  }
}

template <typename ConfigT>
void attention_head_bias_scores<ConfigT>::run(input_buffer<int8>& x0, input_buffer<int8>& x1, input_buffer<bias_t>& u,
                                              output_cascade<acc32>* out)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  project_tokens<ConfigT>(x0.data(), x1.data(), wk_, bk_, ConfigT::K_SHIFT, k_);
  reread<int8> keys{k_}, wq{wq_};
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    projected_queries<ConfigT>(x0.data(), x1.data(), wq(), bq_, first_block_ + b, qt);
    probabilities<ConfigT, ConfigT::LOOKED_BLOCKS>(qt, keys(), exp, p, u.data() + 64 * S::KB * b);
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) write_block(out, p[kb]);
  }
}
