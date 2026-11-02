// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Fused attention (attention.h). Integer throughout, rounding half to even: bit-exact across simulators.
//
// Per query block, transposed so the per-query work runs along lanes (lane key * 8 + query):
//   S^T = K Q^T, int8 logits; m = their maximum
//   P^T = exp table[m - S^T] (uint8, exp(0) = 255), unnormalised
//   l = ones P^T, then theta^T = V^T P^T THETA_GROUP head-dimension tiles at a time on the matrix unit: an 8x8 int32
//   accumulator is a quarter of AIE-ML's, so a whole head dimension's would spill
// In a chain of two, the first tile hands P^T to the second. The division, O = theta / l through a reciprocal table
// in int16, a tile at a time.
#include <aie_api/aie_adf.hpp>
#include "attention.h"

namespace {

using i8x64 = aie::vector<int8, 64>;
using u8x64 = aie::vector<uint8, 64>;
using i8x16 = aie::vector<int8, 16>;
using i8x32 = aie::vector<int8, 32>;
using i16x16 = aie::vector<int16, 16>;
using i16x32 = aie::vector<int16, 32>;
using acc32x16 = aie::accum<acc32, 16>;
using acc32x64 = aie::accum<acc32, 64>;
using score_mm = aie::mmul<8, 8, 8, int8, int8, acc32>;
using value_mm = aie::mmul<8, 8, 8, int8, uint8, acc32>;
using sum_mm = aie::mmul<8, 8, 8, uint8, uint8, acc32>;
using exp_lookup = aie::parallel_lookup<uint8, aie::lut<4, int8>>;
using recip_lookup = aie::parallel_lookup<uint8, aie::lut<4, int16>>;

constexpr int THETA_SHIFT = 7;  // theta (< 128 l < 2^22) to int16
constexpr int THETA_GROUP = 4;

template <typename C>
struct shape {
  static constexpr int NJ = C::HEAD_DIM / 8, QB = C::QUERY_BLOCKS;
  static constexpr int KB = C::KEY_BLOCKS;
};

inline void set_modes()
{
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
}

// The 8x8 tile (row block b, column block j) of a buffer stored in ROWS-row microtiles: one 8-row microtile, or two
// 4-row ones a row band apart.
template <int ROWS, int NJ>
__attribute__((always_inline)) inline i8x64 load_tile(const int8* __restrict base, int b, int j)
{
  const int8* block = base + 64 * NJ * b;
  if constexpr (ROWS == 8) return aie::load_v<64>(block + 64 * j);
  else return aie::concat(aie::load_v<32>(block + 32 * j), aie::load_v<32>(block + 32 * (NJ + j)));
}

template <int ROWS, int NJ>
__attribute__((always_inline)) inline void store_tile(int8* __restrict base, int j, const i8x64& t)
{
  if constexpr (ROWS == 8) {
    aie::store_v(base + 64 * j, t);
  } else {
    aie::store_v(base + 32 * j, t.template extract<32>(0));
    aie::store_v(base + 32 * (NJ + j), t.template extract<32>(1));
  }
}

// P^T of a query block (the 8 queries twice in every 16 lanes of the maximum).
template <typename C>
__attribute__((always_inline)) inline void probabilities(const int8* __restrict q, const int8* __restrict k,
                                                         exp_lookup& exp, u8x64 (&p)[shape<C>::KB])
{
  constexpr int NJ = shape<C>::NJ, KB = shape<C>::KB;
  i8x64 qt[NJ];
  for (int j = 0; j < NJ; ++j) chess_unroll_loop(*) qt[j] = aie::transpose(load_tile<C::Q_ROWS, NJ>(q, 0, j), 8, 8);

  i8x64 s[KB], mx;
  for (int b = 0; b < KB; ++b) chess_unroll_loop(*) {
    score_mm acc;
    acc.mul(load_tile<C::K_ROWS, NJ>(k, b, 0), qt[0]);
    for (int j = 1; j < NJ; ++j) chess_unroll_loop(*) acc.mac(load_tile<C::K_ROWS, NJ>(k, b, j), qt[j]);
    s[b] = acc.template to_vector<int8>(C::SCORE_SHIFT);
    mx = b == 0 ? s[b] : aie::max(mx, s[b]);
  }
  const auto m32 = aie::max(mx.template extract<32>(0), mx.template extract<32>(1));
  const auto m16 = aie::max(m32.template extract<16>(0), m32.template extract<16>(1));
  const i8x16 m = aie::max(m16, aie::shuffle_down_rotate(m16, 8));
  const i8x64 mb = aie::concat(m, m, m, m);
  for (int b = 0; b < KB; ++b) chess_unroll_loop(*) {
    const u8x64 d = aie::vector_cast<uint8>(aie::maxdiff(mb, s[b], true));
    p[b] = aie::vector_cast<uint8>(aie::concat(exp.fetch(d.template extract<32>(0)), exp.fetch(d.template extract<32>(1))));
  }
}

// The sums l = ones P^T: the 8 queries' twice in its first 16 lanes (every row holds them).
template <typename C>
inline acc32x16 sums(const u8x64 (&p)[shape<C>::KB])
{
  const u8x64 ones = aie::broadcast<uint8, 64>(1);
  sum_mm l;
  l.mul(ones, p[0]);
  for (int b = 1; b < shape<C>::KB; ++b) chess_unroll_loop(*) l.mac(ones, p[b]);
  return l.to_accum().template extract<16>(0);
}

// theta^T = V^T P^T of the head-dimension tiles j0 .. j0 + THETA_GROUP - 1 (those below the head dimension).
template <typename C>
inline void thetas(const u8x64 (&p)[shape<C>::KB], const int8* __restrict v, int j0, acc32x64 (&theta)[THETA_GROUP])
{
  constexpr int NJ = shape<C>::NJ;
  value_mm acc[THETA_GROUP];
  for (int g = 0; g < THETA_GROUP; ++g) chess_unroll_loop(*)
    if (j0 + g < NJ) acc[g].mul(aie::transpose(load_tile<C::V_ROWS, NJ>(v, 0, j0 + g), 8, 8), p[0]);
  for (int b = 1; b < shape<C>::KB; ++b) chess_unroll_loop(*)
    for (int g = 0; g < THETA_GROUP; ++g) chess_unroll_loop(*)
      if (j0 + g < NJ) acc[g].mac(aie::transpose(load_tile<C::V_ROWS, NJ>(v, b, j0 + g), 8, 8), p[b]);
  for (int g = 0; g < THETA_GROUP; ++g) chess_unroll_loop(*)
    if (j0 + g < NJ) theta[g] = acc[g].to_accum();
}

// The per-query factors of the division, 32 lanes (the 8 queries 4 times, as in half a theta^T tile): p = 2^e and
// y ~ 2^28 / n from the reciprocal table at round((n - 2^14) / 64), for n = l 2^e in [2^14, 2^15) (255 <= l < 2^15, so
// e <= 7). The shifts are additions: a shift would go through an accumulator and back.
struct normaliser {
  i16x32 pow2, recip;
};

template <typename C>
inline normaliser make_normaliser(const acc32x16& l, recip_lookup& recip)
{
  static_assert(C::KEY_BLOCKS * 8 * 255 < 32768, "the sums and theta / 2^7 must fit int16");
  i16x16 n = l.template to_vector<int16>(0), p = aie::broadcast<int16, 16>(1);
  for (int k = 4; k > 0; k /= 2) chess_unroll_loop(*) {
    const auto small = aie::lt(n, aie::broadcast<int16, 16>(1 << (15 - k)));
    i16x16 nk = n, pk = p;
    for (int d = 0; d < k; ++d) chess_unroll_loop(*) {
      nk = aie::add(nk, nk);
      pk = aie::add(pk, pk);
    }
    n = aie::select(n, nk, small);
    p = aie::select(p, pk, small);
  }
  acc32x16 a;
  a.from_vector(aie::sub(n, aie::broadcast<int16, 16>(1 << 14)), 0);
  const i16x16 y = recip.fetch(a.template to_vector<uint8>(6).template grow<32>()).template extract<16>(0);
  return {aie::concat(p, p), aie::concat(y, y)};
}

// O tile j of a query block (queries in rows) from its theta^T: (theta / 2^7) 2^e y, all in int16.
template <typename C>
inline void normalise(const acc32x64& theta, const normaliser& f, int8* __restrict o, int j)
{
  i8x32 h[2];
  for (int k = 0; k < 2; ++k) chess_unroll_loop(*) {
    const i16x32 t16 = theta.template extract<32>(k).template to_vector<int16>(THETA_SHIFT);
    const i16x32 t = aie::mul(t16, f.pow2).template to_vector<int16>(0);
    h[k] = aie::mul(t, f.recip).template to_vector<int8>(C::OUT_SHIFT);
  }
  store_tile<C::O_ROWS, shape<C>::NJ>(o, j, aie::transpose(aie::concat(h[0], h[1]), 8, 8));
}

// P^T through the cascade: a key block's 64 uint8 as one 512-bit word, bit for bit.
inline void write_block(output_cascade<acc32>* out, const u8x64& p)
{
  acc32x16 a;
  a.from_vector(aie::vector_cast<int32>(p), 0);
  writeincr(out, a);
}

inline u8x64 read_block(input_cascade<acc32>* in)
{
  return aie::vector_cast<uint8>(readincr_v<16>(in).template to_vector<int32>(0));
}

// O of a query block from its P^T: the sums, the division's factors, then theta^T a group of tiles at a time.
template <typename C>
inline void weigh(const u8x64 (&p)[shape<C>::KB], const int8* __restrict v, recip_lookup& recip, int8* __restrict o)
{
  constexpr int NJ = shape<C>::NJ;
  const normaliser f = make_normaliser<C>(sums<C>(p), recip);
  for (int j0 = 0; j0 < NJ; j0 += THETA_GROUP) chess_unroll_loop(*) {
    acc32x64 theta[THETA_GROUP];
    thetas<C>(p, v, j0, theta);
    for (int g = 0; g < THETA_GROUP; ++g) chess_unroll_loop(*)
      if (j0 + g < NJ) normalise<C>(theta[g], f, o, j0 + g);
  }
}

}  // namespace

#define ATTENTION_EXP_LOOKUP                                                                                          \
  set_modes();                                                                                                        \
  exp_lookup exp(aie::lut<4, int8>(256, exp_ab_, exp_cd_));

template <typename ConfigT>
void attention_single<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, input_buffer<int8>& v,
                                    output_buffer<int8>& o)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  recip_lookup recip(aie::lut<4, int16>(256, recip_ab_, recip_cd_));
  for (int b = 0; b < S::QB; ++b) {
    u8x64 p[S::KB];
    probabilities<ConfigT>(q.data() + 64 * S::NJ * b, k.data(), exp, p);
    weigh<ConfigT>(p, v.data(), recip, o.data() + 64 * S::NJ * b);
  }
}

template <typename ConfigT>
void attention_scores<ConfigT>::run(input_buffer<int8>& q, input_buffer<int8>& k, output_cascade<acc32>* out)
{
  using S = shape<ConfigT>;
  ATTENTION_EXP_LOOKUP
  for (int b = 0; b < S::QB; ++b) {
    u8x64 p[S::KB];
    probabilities<ConfigT>(q.data() + 64 * S::NJ * b, k.data(), exp, p);
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
