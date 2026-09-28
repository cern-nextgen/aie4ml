// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// The Conv2D compute core, shared by every I/O flavour: the tap loop over a channel-blocked frame
// in local memory, and the epilogue. A wrapper's only job is to get a frame into that layout --
// the DMA does it for a buffer port, the kernel does it for a stream port.

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include <algorithm>
#include <limits>
#include "parameters.h"

using namespace adf;

// A channel block is moved as 32-bit words where it is not a whole vector. Naming the word type
// `may_alias` is what makes that access defined: the frame really is addressed as both, and copying
// through `__builtin_memcpy` instead measured 17% slower once a pixel spans more than one block.
using conv2d_word_t __attribute__((may_alias)) = int32;

template<typename ConfigT>
inline void conv2d_check_contract() {
  static_assert(ConfigT::K == 8 && ConfigT::N == 8, "conv2d taps are 8-channel blocks");
  static_assert(ConfigT::MB == 2 || ConfigT::MB == 4, "conv2d blocks 2 or 4 mmul row tiles");
  static_assert(ConfigT::NB == 1 ? ConfigT::NBP == 1 : ConfigT::NBP % 2 == 0 && ConfigT::NBP >= ConfigT::NB,
                "NBP pads NB to an even block count; one block, which the one-block core runs, is not padded");
  static_assert(ConfigT::WN == ConfigT::KH * ConfigT::KW * ConfigT::CB * ConfigT::NBP * 64,
                "weights hold one B tile per tap and output block of this tile");
  static_assert(ConfigT::BN == ConfigT::NBP * 8, "bias holds one value per padded output channel");
  static_assert(ConfigT::IN_COLS * 8 % 32 == 0, "frame rows stay 32-byte aligned");
  static_assert(ConfigT::IN_ORIGIN_R == ConfigT::PAD_T || ConfigT::IN_ORIGIN_R == 0,
                "the image starts either behind this layer's border or at the top of a band");
  static_assert(ConfigT::IN_ORIGIN_C >= ConfigT::PAD_L, "the image sits behind its column border");
  static_assert(ConfigT::IN_ORIGIN_C % ConfigT::M == 0 && ConfigT::OUT_ORIGIN_C % ConfigT::M == 0,
                "origins keep tile stores aligned");
  static_assert(ConfigT::FLATTEN || ConfigT::OUT_COLS % ConfigT::M == 0, "output rows keep tile stores aligned");
  static_assert(ConfigT::IN_ORIGIN_C + ConfigT::IN_W <= ConfigT::IN_COLS, "the image fits the frame columns");
  // A frame holds the whole image, one band of it, or -- under a vertical stride -- only the rows
  // its outputs actually read, which can stop short of the last image row. Either way it holds
  // every row the output rows of one core call read, which the next assert states.
  static_assert(ConfigT::BANDS > 1 || ConfigT::STRIDE_H > 1 ||
                    ConfigT::IN_ORIGIN_R + ConfigT::IN_H <= ConfigT::IN_ROWS,
                "the image fits the frame rows");
  static_assert((ConfigT::OUT_H - 1) * ConfigT::STRIDE_H + ConfigT::KH <= ConfigT::IN_ROWS,
                "input frame covers every output row");
  static_assert(ConfigT::OUT_W_COMPUTED % (ConfigT::MB * ConfigT::M) == 0, "computed width is whole register tiles");
  static_assert(ConfigT::IN_COLS % ConfigT::STRIDE_W == 0, "frame columns divide into polyphase classes");
  static_assert(ConfigT::OUT_W_COMPUTED + (ConfigT::KW - 1 + ConfigT::IN_ORIGIN_C - ConfigT::PAD_L) /
                        ConfigT::STRIDE_W <= ConfigT::IN_COLS / ConfigT::STRIDE_W,
                "input frame covers every column the computed tiles read");
  static_assert(ConfigT::STRIDE_W == 1 || !ConfigT::FILLS_BORDER,
                "a strided frame is delivered with its border, because a kernel store writes whole "
                "register tiles and those land in different polyphase classes");
  static_assert(ConfigT::FLATTEN || ConfigT::OUT_ORIGIN_C + ConfigT::OUT_W_COMPUTED / (ConfigT::POOL ? 2 : 1) <=
                                        ConfigT::OUT_COLS,
                "output frame holds every column the computed tiles write");
}

// A fused pool drops an odd last row, in every kernel of a chain alike.
template<typename ConfigT>
inline constexpr int conv2d_rows = ConfigT::POOL ? ConfigT::OUT_H / 2 * 2 : ConfigT::OUT_H;

// AIE1 has no 8-bit vector ALU: it pools in int16 and packs once on the store.
template<typename ConfigT>
using conv2d_pool_t = std::conditional_t<__AIE_ARCH__ == 10 && sizeof(typename ConfigT::result_t) == 1, int16,
                                         typename ConfigT::result_t>;

// Pools a register-tile pair of one output row along the row: 2M pixels to M. M is even, so no pixel pair
// straddles the tiles, and vectors stay within 1024 bits.
template<typename ConfigT, typename MMUL>
__attribute__((always_inline)) static inline aie::vector<conv2d_pool_t<ConfigT>, ConfigT::M * 8>
conv2d_pool_row(MMUL& lo, MMUL& hi)
{
  using result_t = typename ConfigT::result_t;
  using pool_t = conv2d_pool_t<ConfigT>;
  constexpr int SA = ConfigT::M * 8, PIXEL_WORDS = 8 * sizeof(pool_t) / 4;
  static_assert(ConfigT::M % 2 == 0, "a register tile holds whole pixel pairs");
  auto widen = [](aie::vector<result_t, SA> v) {
    if constexpr (std::is_same_v<pool_t, result_t>) return v; else return v.template unpack<pool_t>();
  };
  const auto row = aie::concat(widen(lo.template to_vector<result_t>(ConfigT::SHIFT)),
                               widen(hi.template to_vector<result_t>(ConfigT::SHIFT)));
  // Word permutes are native; AIE1 runs byte permutes through the multiplier.
  const auto words = row.template cast_to<int32>();
  return aie::max(aie::filter_even(words, PIXEL_WORDS).template cast_to<pool_t>(),
                  aie::filter_odd(words, PIXEL_WORDS).template cast_to<pool_t>());
}

// The max identity a pool window starts from: 0 under a fused ReLU, which the max then applies.
template<typename ConfigT, typename T = conv2d_pool_t<ConfigT>>
__attribute__((always_inline)) static inline aie::vector<T, ConfigT::M * 8> conv2d_pool_init()
{
  return aie::broadcast<T, ConfigT::M * 8>(ConfigT::USE_RELU ? T(0) : std::numeric_limits<T>::lowest());
}

// A pooled frame: the kernel fills it with the identity, then merges each row into it as it goes.
template<typename ConfigT>
static inline void conv2d_pool_fill(typename ConfigT::result_t* out)
{
  using result_t = typename ConfigT::result_t;
  const auto identity = conv2d_pool_init<ConfigT, result_t>();
  for (int nb = 0; nb < ConfigT::NB; ++nb)
    for (int py = 0; py < conv2d_rows<ConfigT> / 2; ++py) {
      result_t* row = out + ((nb * ConfigT::OUT_ROWS + ConfigT::OUT_ORIGIN_R + py) * ConfigT::OUT_COLS +
                             ConfigT::OUT_ORIGIN_C) * 8;
      for (int px = 0; px < ConfigT::OUT_W_COMPUTED / 2; px += ConfigT::M) aie::store_v(row + px * 8, identity);
    }
}

template<typename ConfigT, typename MMUL>
static inline void conv2d_merge_pooled(typename ConfigT::result_t* out, int oy, int x, int nb, MMUL& lo, MMUL& hi)
{
  using result_t = typename ConfigT::result_t;
  using pool_t = conv2d_pool_t<ConfigT>;
  constexpr int SA = ConfigT::M * 8;
  result_t* o = out + ((nb * ConfigT::OUT_ROWS + ConfigT::OUT_ORIGIN_R + oy / 2) * ConfigT::OUT_COLS +
                       ConfigT::OUT_ORIGIN_C + x / 2) * 8;
  const auto have = aie::load_v<SA>(o);
  if constexpr (std::is_same_v<pool_t, result_t>) {
    aie::store_v(o, aie::max(conv2d_pool_row<ConfigT>(lo, hi), have));
  } else {
    const auto merged = aie::max(conv2d_pool_row<ConfigT>(lo, hi), have.template unpack<pool_t>());
    aie::store_v(o, merged.template pack<result_t>());
  }
}

// A pooled flatten pools both rows of a window in registers, then stores M pixels of output block nb, pooled
// row py, from pooled column px: each at row 0 of its M-row Dense LHS slot, the pad rows don't-care.
template<typename ConfigT>
static inline void conv2d_store_pooled(typename ConfigT::result_t* out, int py, int px, int nb,
                                       aie::vector<conv2d_pool_t<ConfigT>, ConfigT::M * 8> pooled)
{
  using result_t = typename ConfigT::result_t;
  constexpr int M = ConfigT::M, SA = M * 8, PIXEL_WORDS = 8 * sizeof(result_t) / 4;
  aie::vector<result_t, SA> tile;
  if constexpr (std::is_same_v<conv2d_pool_t<ConfigT>, result_t>)
    tile = pooled;
  else
    tile = pooled.template pack<result_t>();
  auto pair = aie::concat(tile, tile).template cast_to<int32>();
  for (int i = 0; i < M; ++i) {
    if (px + i < ConfigT::OUT_W / 2)
      aie::store_v(out + ((py * (ConfigT::OUT_W / 2) + px + i) * ConfigT::NB + nb) * SA,
                   aie::shuffle_down(pair, PIXEL_WORDS * i).template extract<SA * sizeof(result_t) / 4>(0)
                       .template cast_to<result_t>());
  }
}

template<typename ConfigT>
struct conv2d_geometry {
  static constexpr int M = ConfigT::M, MB = ConfigT::MB, SA = M * 8, SB = 64;
  static constexpr int RB = ConfigT::IN_COLS * 8;   // input row bytes
  static constexpr int CHB = ConfigT::IN_ROWS * RB;  // input channel-block bytes
  static constexpr int T = ConfigT::KH * ConfigT::KW * ConfigT::CB;
  // Columns of one polyphase class. A frame row holds STRIDE_W classes of PW columns, so it is
  // still IN_COLS columns long and RB is unchanged.
  static constexpr int PW = ConfigT::IN_COLS / ConfigT::STRIDE_W;

  // Byte offset of tap t = (ky, kx, cb) from the window of output pixel (oy, 0); the taps are
  // packed in the same order.
  //
  // Output pixel x reads input column x * STRIDE_W + kx + c0, whose polyphase class is
  // (kx + c0) % STRIDE_W -- the same for every x -- at index x + (kx + c0) / STRIDE_W within that
  // class. So one tap is a fixed offset and consecutive output pixels stay 8 bytes apart, which is
  // what lets the register tile load them together at any stride. At STRIDE_W == 1 this is
  // (kx + c0) * 8, the offset it has always been.
  static constexpr int off(int t) {
    const int cb = t % ConfigT::CB, k = t / ConfigT::CB, kx = k % ConfigT::KW, ky = k / ConfigT::KW;
    const int col = kx + ConfigT::IN_ORIGIN_C - ConfigT::PAD_L;
    return cb * CHB + ky * RB + (col % ConfigT::STRIDE_W) * PW * 8 + (col / ConfigT::STRIDE_W) * 8;
  }
  struct table { int off[T]; };
  static constexpr table build() {
    table r{};
    for (int t = 0; t < T; ++t) r.off[t] = off(t);
    return r;
  }
  static constexpr table TBL = build();
};

// The DMA delivers only the image; the border of the frame is whatever the buffer held before.
// Only the border the taps read is zeroed: rows [0, RR1), columns [RC0, RC1).
template<typename ConfigT>
static inline void conv2d_zero_border(typename ConfigT::data_t* frame) {
  using G = conv2d_geometry<ConfigT>;
  using data_t = typename ConfigT::data_t;
  constexpr int R0 = ConfigT::IN_ORIGIN_R, R1 = R0 + ConfigT::IN_H;
  constexpr int C0 = ConfigT::IN_ORIGIN_C, C1 = C0 + ConfigT::IN_W;
  constexpr int C1_EVEN = (C1 + 1) / 2 * 2;
  constexpr int RR1 = std::min(ConfigT::IN_ROWS, (conv2d_rows<ConfigT> - 1) * ConfigT::STRIDE_H + ConfigT::KH);
  constexpr int RC0 = ConfigT::IN_ORIGIN_C - ConfigT::PAD_L;
  constexpr int RC1 = std::min(ConfigT::IN_COLS, RC0 + ConfigT::OUT_W_COMPUTED + ConfigT::KW - 1);
  const auto z32 = aie::zeros<data_t, 32>();
  const auto z16 = aie::zeros<data_t, 16>();
  // Border rows and image rows in loops of their own: a branch in the row loop keeps it from pipelining.
  auto border_row = [&](data_t* row) __attribute__((always_inline)) {
    for (int c = RC0 / 4 * 4; c < RC1; c += 4) aie::store_v(row + c * 8, z32);
  };
  for (int cb = 0; cb < ConfigT::CB; ++cb) {
    data_t* block = frame + cb * G::CHB;
    for (int r = 0; r < std::min(R0, RR1); ++r) border_row(block + r * G::RB);
    for (int r = R1; r < RR1; ++r) border_row(block + r * G::RB);
    for (int r = R0; r < std::min(R1, RR1); ++r) {
      data_t* row = block + r * G::RB;
      for (int c = RC0 / 2 * 2; c < C0; c += 2) aie::store_v(row + c * 8, z16);
      if constexpr (C1 % 2) {  // one block is not a vector, so it goes as words
        conv2d_word_t* const odd = reinterpret_cast<conv2d_word_t*>(row + C1 * 8);
        for (int w = 0; w < 2 * int(sizeof(data_t)); ++w) odd[w] = 0;
      }
      for (int c = C1_EVEN; c < RC1; c += 2) aie::store_v(row + c * 8, z16);
    }
  }
}

// One tile's share of the layer: every output pixel, this tile's output blocks (NB), its input
// blocks (CB). CASC_IN continues the partial sums of the previous tile in the chain and CASC_OUT
// passes this tile's on; the tile that writes the result is the one that adds the bias.
template<typename ConfigT, bool CASC_IN, bool CASC_OUT>
static inline void conv2d_tile(typename ConfigT::data_t* frame,
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
  constexpr int M = ConfigT::M, MB = ConfigT::MB, NB = ConfigT::NB, NBP = ConfigT::NBP;
  constexpr int SA = G::SA, SB = G::SB;
  using MMUL = aie::mmul<M, 8, 8, data_t, weight_t, acc_scalar_t>;

  if constexpr (ConfigT::FILLS_BORDER) conv2d_zero_border<ConfigT>(frame);
  if constexpr (ConfigT::POOL && !ConfigT::FLATTEN && !CASC_OUT) conv2d_pool_fill<ConfigT>(out);

  // A pooled flatten computes each register block for both rows of its pool windows in turn and pools them in
  // registers; every kernel of its cascade walks the rows in that order, so the partial sums line up. A pooled
  // frame merges row by row instead: a row loop that holds no pool state keeps its software pipelining.
  constexpr bool PAIRS = ConfigT::POOL && ConfigT::FLATTEN;
  constexpr int ROWS = PAIRS ? 2 : 1;
  for (int oy0 = 0; oy0 < conv2d_rows<ConfigT>; oy0 += ROWS) {
    for (int z = 0; z < ConfigT::OUT_W_COMPUTED; z += MB * M) {
      for (int j = 0; j < NB; j += 2) {
        [[maybe_unused]] aie::vector<conv2d_pool_t<ConfigT>, SA> p0, p1, p2, p3;
        if constexpr (PAIRS) p0 = p1 = p2 = p3 = conv2d_pool_init<ConfigT>();
        for (int dy = 0; dy < ROWS; ++dy)
          chess_flatten_loop
        {
          const int oy = oy0 + dy;
          const data_t* pA = frame + oy * ConfigT::STRIDE_H * G::RB + z * 8;
          aie::vector<bias_t, M * 8> bb0, bb1;
          if constexpr (!CASC_IN) {
            aie::vector<bias_t, 8> b0 = aie::load_v<8>(bias + j * 8);
            aie::vector<bias_t, 8> b1 = aie::load_v<8>(bias + (j + 1) * 8);
            for (int m = 0; m < M; ++m) {
              bb0.template insert<8>(m, b0);
              bb1.template insert<8>(m, b1);
            }
          }
          MMUL C00, C01, C10, C11, C20, C21, C30, C31;
          if constexpr (CASC_IN) {
            C00 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            C01 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            C10 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            C11 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            if constexpr (MB == 4) {
              C20 = MMUL(readincr_v<MMUL::size_C>(inCascade));
              C21 = MMUL(readincr_v<MMUL::size_C>(inCascade));
              C30 = MMUL(readincr_v<MMUL::size_C>(inCascade));
              C31 = MMUL(readincr_v<MMUL::size_C>(inCascade));
            }
          } else {
            C00 = bb0; C10 = bb0; C01 = bb1; C11 = bb1;
            if constexpr (MB == 4) { C20 = bb0; C30 = bb0; C21 = bb1; C31 = bb1; }
          }

          const weight_t __aie_dm_resource_a* __restrict pB = (const weight_t __aie_dm_resource_a*)(wts + j * SB);
          for (int t = 0; t < G::T; ++t)
            chess_prepare_for_pipelining
          {
            const data_t __aie_dm_resource_b* __restrict a = (const data_t __aie_dm_resource_b*)(pA + G::TBL.off[t]);
            aie::vector<weight_t, SB> B0 = aie::load_v<SB>(pB);
            aie::vector<weight_t, SB> B1 = aie::load_v<SB>(pB + SB);
            pB += NBP * SB;
            if constexpr (MB == 2) {
              aie::vector<data_t, 2 * SA> w = aie::load_unaligned_v<2 * SA>(a, 8);
              aie::vector<data_t, SA> A0 = w.template extract<SA>(0);
              aie::vector<data_t, SA> A1 = w.template extract<SA>(1);
              C00.mac(A0, B0); C01.mac(A0, B1); C10.mac(A1, B0); C11.mac(A1, B1);
            } else {
              aie::vector<data_t, 4 * SA> w;
              if constexpr (4 * SA <= 64) {
                w = aie::load_unaligned_v<4 * SA>(a, 8);
              } else {
                for (int q = 0; q < 4 * SA / 64; ++q)
                  w.template insert<64>(q, aie::load_unaligned_v<64>(a + q * 64, 8));
              }
              aie::vector<data_t, SA> A0 = w.template extract<SA>(0);
              aie::vector<data_t, SA> A1 = w.template extract<SA>(1);
              aie::vector<data_t, SA> A2 = w.template extract<SA>(2);
              aie::vector<data_t, SA> A3 = w.template extract<SA>(3);
              C00.mac(A0, B0); C01.mac(A0, B1); C10.mac(A1, B0); C11.mac(A1, B1);
              C20.mac(A2, B0); C21.mac(A2, B1); C30.mac(A3, B0); C31.mac(A3, B1);
            }
          }

          if constexpr (CASC_OUT) {
            writeincr(outCascade, C00.to_accum());
            writeincr(outCascade, C01.to_accum());
            writeincr(outCascade, C10.to_accum());
            writeincr(outCascade, C11.to_accum());
            if constexpr (MB == 4) {
              writeincr(outCascade, C20.to_accum());
              writeincr(outCascade, C21.to_accum());
              writeincr(outCascade, C30.to_accum());
              writeincr(outCascade, C31.to_accum());
            }
          } else if constexpr (PAIRS) {
            p0 = aie::max(p0, conv2d_pool_row<ConfigT>(C00, C10));
            p1 = aie::max(p1, conv2d_pool_row<ConfigT>(C01, C11));
            if constexpr (MB == 4) {
              p2 = aie::max(p2, conv2d_pool_row<ConfigT>(C20, C30));
              p3 = aie::max(p3, conv2d_pool_row<ConfigT>(C21, C31));
            }
          } else if constexpr (ConfigT::POOL) {
            conv2d_merge_pooled<ConfigT>(out, oy, z, j, C00, C10);
            if (j + 1 < NB) conv2d_merge_pooled<ConfigT>(out, oy, z, j + 1, C01, C11);
            if constexpr (MB == 4) {
              conv2d_merge_pooled<ConfigT>(out, oy, z + 2 * M, j, C20, C30);
              if (j + 1 < NB) conv2d_merge_pooled<ConfigT>(out, oy, z + 2 * M, j + 1, C21, C31);
            }
          } else {
            // Inlined: an outlined call would spill every accumulator it takes by reference (MLv2 outlines it).
            auto store_tile = [&](int nb, int mm, MMUL& acc) __attribute__((always_inline)) {
              if (nb >= NB) return;
              aie::vector<result_t, SA> tile = acc.template to_vector<result_t>(ConfigT::SHIFT);
              if constexpr (ConfigT::USE_RELU) tile = aie::max(tile, result_t(0));
              if constexpr (ConfigT::FLATTEN) {
                // Dense LHS row: chunk (pixel, nb) sits at row 0 of its M-row slot; the pad rows are
                // don't-care, so each pixel stores the tile rotated to start at itself.
                auto pair = aie::concat(tile, tile).template cast_to<int32>();
                for (int i = 0; i < M; ++i) {
                  const int ox = z + mm * M + i;
                  if (ox < ConfigT::OUT_W) {
                    const int chunk = (oy * ConfigT::OUT_W + ox) * NB + nb;
                    aie::store_v(out + chunk * SA,
                                 aie::shuffle_down(pair, 2 * i).template extract<M * 2>(0)
                                     .template cast_to<result_t>());
                  }
                }
              } else {
                result_t* o = out + ((nb * ConfigT::OUT_ROWS + ConfigT::OUT_ORIGIN_R + oy) * ConfigT::OUT_COLS +
                                     ConfigT::OUT_ORIGIN_C + z + mm * M) * 8;
                aie::store_v(o, tile);
              }
            };
            store_tile(j, 0, C00); store_tile(j, 1, C10);
            store_tile(j + 1, 0, C01); store_tile(j + 1, 1, C11);
            if constexpr (MB == 4) {
              store_tile(j, 2, C20); store_tile(j, 3, C30);
              store_tile(j + 1, 2, C21); store_tile(j + 1, 3, C31);
            }
          }
        }
        if constexpr (PAIRS && !CASC_OUT) {
          conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2, j, p0);
          if (j + 1 < NB) conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2, j + 1, p1);
          if constexpr (MB == 4) {
            conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2 + M, j, p2);
            if (j + 1 < NB) conv2d_store_pooled<ConfigT>(out, oy0 / 2, z / 2 + M, j + 1, p3);
          }
        }
      }
    }
  }
}
