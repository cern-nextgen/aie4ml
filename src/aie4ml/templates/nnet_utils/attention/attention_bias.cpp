// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention with an additive bias (attention_bias.h) on the shared device code (attention_core.h).
#include "attention_core.h"
#include "attention_bias.h"

template <typename ConfigT>
void attention_bias_single<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<int8>& v,
                                         input_buffer<bias_t>& u, output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  reread<int8> keys{k.data()}, values{v.data()};
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    transposed_queries<ConfigT>(q.data() + 64 * S::NJ * b, qt);
    probabilities<ConfigT, S::KB>(qt, keys(), exp, p, u.data() + 64 * S::KB * b);
    weigh<ConfigT>(p, values(), recip, o.data() + 64 * S::NJ * b);
  }
}

template <typename ConfigT>
void attention_bias_scores<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<bias_t>& u,
                                         output_cascade<acc32>* out)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  reread<int8> keys{k.data()};
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    transposed_queries<ConfigT>(q.data() + 64 * S::NJ * b, qt);
    probabilities<ConfigT, S::KB>(qt, keys(), exp, p, u.data() + 64 * S::KB * b);
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) write_block(out, p[kb]);
  }
}
