// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Conv2D core for a pointwise (1x1) conv whose step is a few taps: one output block at a time, its taps -- one
// aligned load each -- unrolled and its steps (every row and register block) software-pipelined, so a step's epilogue
// overlaps the next one's taps. The other cores pipeline each step's taps instead, a loop of a few iterations here.
// Same frame, register tiling and epilogue as conv2d_tile; weights unpadded.

#pragma once
#include "conv2d_core.h"

template<typename ConfigT, bool CASC_IN, bool CASC_OUT>
static inline void conv2d_tile_pointwise(typename ConfigT::data_t* __restrict frame,
                                         const typename ConfigT::weight_t* wts,
                                         const typename ConfigT::bias_t* bias,
                                         typename ConfigT::result_t* __restrict out,
                                         input_cascade<typename ConfigT::cascade_t>* inCascade,
                                         output_cascade<typename ConfigT::cascade_t>* outCascade)
{
  using G = conv2d_geometry<ConfigT>;
  using data_t = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t = typename ConfigT::bias_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;
  constexpr int M = ConfigT::M, MB = ConfigT::MB, NB = ConfigT::NB;
  constexpr int SA = G::SA, SB = G::SB;
  using MMUL = aie::mmul<M, 8, 8, data_t, weight_t, acc_scalar_t>;
  // Constants, not expressions in the calls: chess folds only those into aligned loads.
  constexpr int ALIGN_2 = std::min(2 * SA, G::A_ALIGN), ALIGN_4 = std::min(4 * SA, G::A_ALIGN);
  constexpr int ALIGN_64 = std::min(64, G::A_ALIGN);
  static_assert(ConfigT::KH * ConfigT::KW == 1 && ConfigT::NBP == NB, "a pointwise conv, its blocks unpadded");

  if constexpr (ConfigT::FILLS_BORDER) conv2d_zero_border<ConfigT>(frame);
  if constexpr (ConfigT::POOL && !ConfigT::FLATTEN && !CASC_OUT) conv2d_pool_fill<ConfigT>(out);

  // A pooled flatten: both rows of a pool window in one step, pooled in registers (see conv2d_tile).
  constexpr bool PAIRS = ConfigT::POOL && ConfigT::FLATTEN;
  constexpr int ROWS = PAIRS ? 2 : 1;
  constexpr int ZS = ConfigT::OUT_W_COMPUTED / (MB * M);  // register blocks per row
  constexpr int STEPS = conv2d_rows<ConfigT> / ROWS * ZS;
  for (int nb = 0; nb < NB; ++nb) {
    aie::vector<bias_t, M * 8> bb;
    if constexpr (!CASC_IN) {
      aie::vector<bias_t, 8> b = aie::load_v<8>(bias + nb * 8);
      for (int m = 0; m < M; ++m) bb.template insert<8>(m, b);
    }

    for (int s = 0; s < STEPS; ++s)
      chess_prepare_for_pipelining chess_loop_range(STEPS, STEPS)
    {
      const int oy0 = s / ZS * ROWS, z = s % ZS * MB * M;
      [[maybe_unused]] aie::vector<conv2d_pool_t<ConfigT>, SA> p0, p1;
      if constexpr (PAIRS) p0 = p1 = conv2d_pool_init<ConfigT>();
      for (int dy = 0; dy < ROWS; ++dy)
        chess_flatten_loop
      {
        const int oy = oy0 + dy;
        const data_t* pA = frame + oy * ConfigT::STRIDE_H * G::RB + z * 8;
        MMUL C0, C1, C2, C3;
        if constexpr (CASC_IN) {
          C0 = MMUL(readincr_v<MMUL::size_C>(inCascade));
          C1 = MMUL(readincr_v<MMUL::size_C>(inCascade));
          if constexpr (MB == 4) {
            C2 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            C3 = MMUL(readincr_v<MMUL::size_C>(inCascade));
          }
        } else {
          C0 = bb; C1 = bb;
          if constexpr (MB == 4) { C2 = bb; C3 = bb; }
        }

        const weight_t __aie_dm_resource_a* __restrict pB = (const weight_t __aie_dm_resource_a*)(wts + nb * SB);
        // Unrolled, not flattened: flattening scheduled the AIE1 step longer.
        for (int t = 0; t < G::T; ++t)
          chess_unroll_loop(*)
        {
          const data_t __aie_dm_resource_b* __restrict a = (const data_t __aie_dm_resource_b*)(pA + G::TBL.off[t]);
          aie::vector<weight_t, SB> B = aie::load_v<SB>(pB);
          pB += NB * SB;
          if constexpr (MB == 2) {
            aie::vector<data_t, 2 * SA> w = aie::load_unaligned_v<2 * SA>(a, ALIGN_2);
            C0.mac(w.template extract<SA>(0), B);
            C1.mac(w.template extract<SA>(1), B);
          } else {
            aie::vector<data_t, 4 * SA> w;
            if constexpr (4 * SA <= 64) {
              w = aie::load_unaligned_v<4 * SA>(a, ALIGN_4);
            } else {
              for (int q = 0; q < 4 * SA / 64; ++q) w.template insert<64>(q, aie::load_unaligned_v<64>(a + q * 64, ALIGN_64));
            }
            C0.mac(w.template extract<SA>(0), B);
            C1.mac(w.template extract<SA>(1), B);
            C2.mac(w.template extract<SA>(2), B);
            C3.mac(w.template extract<SA>(3), B);
          }
        }

        if constexpr (CASC_OUT) {
          writeincr(outCascade, C0.to_accum());
          writeincr(outCascade, C1.to_accum());
          if constexpr (MB == 4) {
            writeincr(outCascade, C2.to_accum());
            writeincr(outCascade, C3.to_accum());
          }
        } else if constexpr (PAIRS) {
          p0 = aie::max(p0, conv2d_pool_row<ConfigT>(C0, C1));
          if constexpr (MB == 4) p1 = aie::max(p1, conv2d_pool_row<ConfigT>(C2, C3));
        } else if constexpr (ConfigT::POOL) {
          conv2d_merge_pooled<ConfigT>(out, oy, z, nb, C0, C1);
          if constexpr (MB == 4) conv2d_merge_pooled<ConfigT>(out, oy, z + 2 * M, nb, C2, C3);
        } else {
          // Inlined: an outlined call would spill every accumulator it takes by reference (MLv2 outlines it).
          auto store_tile = [&](int mm, MMUL& acc) __attribute__((always_inline)) {
            aie::vector<result_t, SA> tile = conv2d_activate<ConfigT>(acc);
            if constexpr (ConfigT::FLATTEN) {
              // Dense LHS row: chunk (pixel, nb) sits at row 0 of its M-row slot; the pad rows are
              // don't-care, so each pixel stores the tile rotated to start at itself.
              auto pair = aie::concat(tile, tile).template cast_to<int32>();
              for (int i = 0; i < M; ++i) {
                const int ox = z + mm * M + i;
                if (ox < ConfigT::OUT_W) {
                  aie::store_v(out + ((oy * ConfigT::OUT_W + ox) * NB + nb) * SA,
                               aie::shuffle_down(pair, 2 * i).template extract<M * 2>(0).template cast_to<result_t>());
                }
              }
            } else {
              result_t* o = out + ((nb * ConfigT::OUT_ROWS + ConfigT::OUT_ORIGIN_R + oy) * ConfigT::OUT_COLS +
                                   ConfigT::OUT_ORIGIN_C + z + mm * M) * 8;
              aie::store_v(o, tile);
            }
          };
          store_tile(0, C0); store_tile(1, C1);
          if constexpr (MB == 4) { store_tile(2, C2); store_tile(3, C3); }
        }
      }
      if constexpr (PAIRS && !CASC_OUT) {
        conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2, nb, p0);
        if constexpr (MB == 4) conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2 + M, nb, p1);
      }
    }
  }
}
