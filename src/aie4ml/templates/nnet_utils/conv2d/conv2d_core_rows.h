// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Conv2D core for a tile of one output block, two output rows a step: twice the one-block core's accumulators, each
// tap's B tile shared by both rows. A fused pool takes both rows of its window from registers and stores once.
// Same frame, register tiling and epilogue as conv2d_tile; weights unpadded.

#pragma once
#include "conv2d_core.h"

// A tile of one block whose rows pair up, and whose step is more than one tap: one tap leaves the one-block core's
// pixel loop innermost, which the compiler already pipelines.
template<typename ConfigT>
inline constexpr bool conv2d_uses_rows_core = ConfigT::NB == 1 && !ConfigT::POINTWISE_CORE &&
                                              conv2d_rows<ConfigT> % 2 == 0 && conv2d_geometry<ConfigT>::T > 1;

template<typename ConfigT, bool CASC_IN, bool CASC_OUT>
static inline void conv2d_tile_rows(typename ConfigT::data_t* frame,
                                    const typename ConfigT::weight_t* wts,
                                    const typename ConfigT::bias_t* bias,
                                    typename ConfigT::result_t* out,
                                    input_cascade<typename ConfigT::cascade_t>* inCascade,
                                    output_cascade<typename ConfigT::cascade_t>* outCascade)
{
  using G = conv2d_geometry<ConfigT>;
  using data_t = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t = typename ConfigT::bias_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;
  using pool_t = conv2d_pool_t<ConfigT>;
  constexpr int M = ConfigT::M, MB = ConfigT::MB;
  constexpr int SA = G::SA, SB = G::SB;
  constexpr int ROW = ConfigT::STRIDE_H * G::RB;  // from one output row's window to the next one's
  using MMUL = aie::mmul<M, 8, 8, data_t, weight_t, acc_scalar_t>;
  // Constants, not expressions in the calls: chess folds only those into aligned loads.
  constexpr int ALIGN_2 = std::min(2 * SA, G::A_ALIGN), ALIGN_4 = std::min(4 * SA, G::A_ALIGN);
  constexpr int ALIGN_64 = std::min(64, G::A_ALIGN);
  static_assert(conv2d_uses_rows_core<ConfigT> && ConfigT::NBP == 1, "one output block, its rows in pairs");

  if constexpr (ConfigT::FILLS_BORDER) conv2d_zero_border<ConfigT>(frame);

  aie::vector<bias_t, M * 8> bb;
  if constexpr (!CASC_IN) {
    aie::vector<bias_t, 8> b = aie::load_v<8>(bias);
    for (int m = 0; m < M; ++m) bb.template insert<8>(m, b);
  }

  for (int oy = 0; oy < conv2d_rows<ConfigT>; oy += 2) {
    for (int z = 0; z < ConfigT::OUT_W_COMPUTED; z += MB * M) {
      const data_t* pA = frame + oy * ROW + z * 8;
      // C: row oy, D: row oy + 1.
      MMUL C0, C1, C2, C3, D0, D1, D2, D3;
      if constexpr (CASC_IN) {
        C0 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        C1 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        if constexpr (MB == 4) {
          C2 = MMUL(readincr_v<MMUL::size_C>(inCascade));
          C3 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        }
        D0 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        D1 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        if constexpr (MB == 4) {
          D2 = MMUL(readincr_v<MMUL::size_C>(inCascade));
          D3 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        }
      } else {
        C0 = bb; C1 = bb; D0 = bb; D1 = bb;
        if constexpr (MB == 4) { C2 = bb; C3 = bb; D2 = bb; D3 = bb; }
      }

      const weight_t __aie_dm_resource_a* __restrict pB = (const weight_t __aie_dm_resource_a*)wts;
      for (int t = 0; t < G::T; ++t)
        chess_prepare_for_pipelining
      {
        const data_t __aie_dm_resource_b* __restrict a = (const data_t __aie_dm_resource_b*)(pA + G::TBL.off[t]);
        aie::vector<weight_t, SB> B = aie::load_v<SB>(pB);
        pB += SB;
        if constexpr (MB == 2) {
          aie::vector<data_t, 2 * SA> w = aie::load_unaligned_v<2 * SA>(a, ALIGN_2);
          aie::vector<data_t, 2 * SA> v = aie::load_unaligned_v<2 * SA>(a + ROW, ALIGN_2);
          C0.mac(w.template extract<SA>(0), B);
          C1.mac(w.template extract<SA>(1), B);
          D0.mac(v.template extract<SA>(0), B);
          D1.mac(v.template extract<SA>(1), B);
        } else {
          aie::vector<data_t, 4 * SA> w, v;
          if constexpr (4 * SA <= 64) {
            w = aie::load_unaligned_v<4 * SA>(a, ALIGN_4);
            v = aie::load_unaligned_v<4 * SA>(a + ROW, ALIGN_4);
          } else {
            for (int q = 0; q < 4 * SA / 64; ++q) {
              w.template insert<64>(q, aie::load_unaligned_v<64>(a + q * 64, ALIGN_64));
              v.template insert<64>(q, aie::load_unaligned_v<64>(a + ROW + q * 64, ALIGN_64));
            }
          }
          C0.mac(w.template extract<SA>(0), B);
          C1.mac(w.template extract<SA>(1), B);
          C2.mac(w.template extract<SA>(2), B);
          C3.mac(w.template extract<SA>(3), B);
          D0.mac(v.template extract<SA>(0), B);
          D1.mac(v.template extract<SA>(1), B);
          D2.mac(v.template extract<SA>(2), B);
          D3.mac(v.template extract<SA>(3), B);
        }
      }

      if constexpr (CASC_OUT) {
        writeincr(outCascade, C0.to_accum());
        writeincr(outCascade, C1.to_accum());
        if constexpr (MB == 4) {
          writeincr(outCascade, C2.to_accum());
          writeincr(outCascade, C3.to_accum());
        }
        writeincr(outCascade, D0.to_accum());
        writeincr(outCascade, D1.to_accum());
        if constexpr (MB == 4) {
          writeincr(outCascade, D2.to_accum());
          writeincr(outCascade, D3.to_accum());
        }
      } else if constexpr (ConfigT::POOL) {
        // Rows oy and oy + 1 are one pool window's: M pooled pixels per register-tile pair.
        auto pool = [&](MMUL& c_lo, MMUL& c_hi, MMUL& d_lo, MMUL& d_hi) __attribute__((always_inline)) {
          return aie::max(aie::max(conv2d_pool_init<ConfigT>(), conv2d_pool_row<ConfigT>(c_lo, c_hi)),
                          conv2d_pool_row<ConfigT>(d_lo, d_hi));
        };
        auto store = [&](int px, aie::vector<pool_t, SA> pooled) __attribute__((always_inline)) {
          if constexpr (ConfigT::FLATTEN) {
            conv2d_store_pooled<ConfigT>(out, oy / 2, px, 0, pooled);
          } else {
            result_t* o = out + ((ConfigT::OUT_ORIGIN_R + oy / 2) * ConfigT::OUT_COLS + ConfigT::OUT_ORIGIN_C + px) * 8;
            if constexpr (std::is_same_v<pool_t, result_t>) aie::store_v(o, pooled);
            else aie::store_v(o, pooled.template pack<result_t>());
          }
        };
        store(z / 2, pool(C0, C1, D0, D1));
        if constexpr (MB == 4) store(z / 2 + M, pool(C2, C3, D2, D3));
      } else {
        // Inlined: an outlined call would spill every accumulator it takes by reference (MLv2 outlines it).
        auto store_tile = [&](int dy, int mm, MMUL& acc) __attribute__((always_inline)) {
          aie::vector<result_t, SA> tile = conv2d_activate<ConfigT>(acc);
          const int row = oy + dy;
          if constexpr (ConfigT::FLATTEN) {
            // Dense LHS row: chunk (pixel, 0) sits at row 0 of its M-row slot; the pad rows are
            // don't-care, so each pixel stores the tile rotated to start at itself.
            auto pair = aie::concat(tile, tile).template cast_to<int32>();
            for (int i = 0; i < M; ++i) {
              const int ox = z + mm * M + i;
              if (ox < ConfigT::OUT_W) {
                aie::store_v(out + (row * ConfigT::OUT_W + ox) * SA,
                             aie::shuffle_down(pair, 2 * i).template extract<M * 2>(0).template cast_to<result_t>());
              }
            }
          } else {
            result_t* o =
                out + ((ConfigT::OUT_ORIGIN_R + row) * ConfigT::OUT_COLS + ConfigT::OUT_ORIGIN_C + z + mm * M) * 8;
            aie::store_v(o, tile);
          }
        };
        store_tile(0, 0, C0); store_tile(0, 1, C1);
        store_tile(1, 0, D0); store_tile(1, 1, D1);
        if constexpr (MB == 4) {
          store_tile(0, 2, C2); store_tile(0, 3, C3);
          store_tile(1, 2, D2); store_tile(1, 3, D3);
        }
      }
    }
  }
}
