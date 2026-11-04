// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention (attention.h) on the shared device code (attention_core.h).
#include "attention_core.h"
#include "attention.h"

template <typename ConfigT>
void attention_single<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<int8>& v,
                                    output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    transposed_queries<ConfigT>(q.data() + 64 * S::NJ * b, qt);
    probabilities<ConfigT>(qt, k.data(), exp, p);
    weigh<ConfigT>(p, v.data(), recip, o.data() + 64 * S::NJ * b);
  }
}

template <typename ConfigT>
void attention_scores<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, output_cascade<acc32>* out)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  for (int b = 0; b < S::QB; ++b) {
    i8x64 qt[S::NJ];
    u8x64 p[S::KB];
    transposed_queries<ConfigT>(q.data() + 64 * S::NJ * b, qt);
    probabilities<ConfigT>(qt, k.data(), exp, p);
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) write_block(out, p[kb]);
  }
}

template <typename ConfigT>
void attention_values<ConfigT>::run(input_cascade<acc32>* in, input_buffer<int8>& v, output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  set_modes();
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  for (int b = 0; b < S::QB; ++b) {
    u8x64 p[S::KB];
    for (int kb = 0; kb < S::KB; ++kb) chess_unroll_loop(*) p[kb] = read_block(in);
    weigh<ConfigT>(p, v.data(), recip, o.data() + 64 * S::NJ * b);
  }
}
