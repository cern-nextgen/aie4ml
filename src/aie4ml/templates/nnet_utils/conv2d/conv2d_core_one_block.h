// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Conv2D core for a tile of one output block: conv2d_tile steps blocks in pairs, which for one block
// wastes half its MACs on padding. Same frame, register tiling and epilogue; weights unpadded.

#pragma once
#include "conv2d_core.h"

template<typename ConfigT, bool CASC_IN, bool CASC_OUT>
static inline void conv2d_tile_one_block(typename ConfigT::data_t* frame,
                                         const typename ConfigT::weight_t* wts,
                                         const typename ConfigT::bias_t* bias,
                                         typename ConfigT::result_t* out,
                                         input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                         output_cascade<typename ConfigT::acc_scalar_t>* outCascade)
{
  using G = conv2d_geometry<ConfigT>;
  using data_t = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t = typename ConfigT::bias_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;
  constexpr int M = ConfigT::M, MB = ConfigT::MB;
  constexpr int SA = G::SA, SB = G::SB;
  using MMUL = aie::mmul<M, 8, 8, data_t, weight_t, acc_scalar_t>;
  static_assert(ConfigT::NB == 1 && ConfigT::NBP == 1, "one output block, unpadded");

  if constexpr (ConfigT::FILLS_BORDER) conv2d_zero_border<ConfigT>(frame);
  if constexpr (ConfigT::POOL && !CASC_OUT) conv2d_pool_fill<ConfigT>(out);

  aie::vector<bias_t, M * 8> bb;
  if constexpr (!CASC_IN) {
    aie::vector<bias_t, 8> b = aie::load_v<8>(bias);
    for (int m = 0; m < M; ++m) bb.template insert<8>(m, b);
  }

  for (int oy = 0; oy < conv2d_rows<ConfigT>; ++oy) {
    for (int z = 0; z < ConfigT::OUT_W_COMPUTED; z += MB * M) {
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

      const weight_t __aie_dm_resource_a* __restrict pB = (const weight_t __aie_dm_resource_a*)wts;
      for (int t = 0; t < G::T; ++t)
        chess_prepare_for_pipelining
      {
        const data_t __aie_dm_resource_b* __restrict a = (const data_t __aie_dm_resource_b*)(pA + G::TBL.off[t]);
        aie::vector<weight_t, SB> B = aie::load_v<SB>(pB);
        pB += SB;
        if constexpr (MB == 2) {
          aie::vector<data_t, 2 * SA> w = aie::load_unaligned_v<2 * SA>(a, 8);
          C0.mac(w.template extract<SA>(0), B);
          C1.mac(w.template extract<SA>(1), B);
        } else {
          aie::vector<data_t, 4 * SA> w;
          if constexpr (4 * SA <= 64) {
            w = aie::load_unaligned_v<4 * SA>(a, 8);
          } else {
            for (int q = 0; q < 4 * SA / 64; ++q) w.template insert<64>(q, aie::load_unaligned_v<64>(a + q * 64, 8));
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
      } else if constexpr (ConfigT::POOL) {
        conv2d_store_pooled<ConfigT>(out, oy, z, 0, C0, C1);
        if constexpr (MB == 4) conv2d_store_pooled<ConfigT>(out, oy, z + 2 * M, 0, C2, C3);
      } else {
        // Inlined: an outlined call would spill every accumulator it takes by reference (MLv2 outlines it).
        auto store_tile = [&](int mm, MMUL& acc) __attribute__((always_inline)) {
          aie::vector<result_t, SA> tile = acc.template to_vector<result_t>(ConfigT::SHIFT);
          if constexpr (ConfigT::USE_RELU) tile = aie::max(tile, result_t(0));
          if constexpr (ConfigT::FLATTEN) {
            // Dense LHS row: chunk (pixel, 0) sits at row 0 of its M-row slot; the pad rows are
            // don't-care, so each pixel stores the tile rotated to start at itself.
            auto pair = aie::concat(tile, tile).template cast_to<int32>();
            for (int i = 0; i < M; ++i) {
              const int ox = z + mm * M + i;
              if (ox < ConfigT::OUT_W) {
                aie::store_v(out + (oy * ConfigT::OUT_W + ox) * SA,
                             aie::shuffle_down(pair, 2 * i).template extract<M * 2>(0).template cast_to<result_t>());
              }
            }
          } else {
            result_t* o = out + ((ConfigT::OUT_ORIGIN_R + oy) * ConfigT::OUT_COLS + ConfigT::OUT_ORIGIN_C + z + mm * M) * 8;
            aie::store_v(o, tile);
          }
        };
        store_tile(0, C0); store_tile(1, C1);
        if constexpr (MB == 4) { store_tile(2, C2); store_tile(3, C3); }
      }
    }
  }
}

// The core a tile runs: its own schedule for one output block, the paired one otherwise.
template<typename ConfigT, bool CASC_IN, bool CASC_OUT, typename... Args>
static inline void conv2d_compute(Args... args)
{
  if constexpr (ConfigT::NB == 1)
    conv2d_tile_one_block<ConfigT, CASC_IN, CASC_OUT>(args...);
  else
    conv2d_tile<ConfigT, CASC_IN, CASC_OUT>(args...);
}
