// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include <utility>
#include "frame_fold.h"

using namespace adf;

namespace {
// Frame columns one step writes: a vector of 8-byte pixels.
constexpr int STEP = 16;

// What a step's loads may read past the units they need.
constexpr int READ_SLACK = 128;

// N bytes from `p`, which is G-aligned: one load, or aligned halves where the generation loads it from a wider boundary.
template<int N, int G, typename T>
aie::vector<T, N> load_aligned(const T* p)
{
  if constexpr (aie::vector_ldst_align_v<T, N> <= G)
    return aie::load_v<N>(p);
  else
    return aie::concat(load_aligned<N / 2, G>(p), load_aligned<N / 2, G>(p + N / 2));
}

// N bytes from byte OFF of `row`, which starts A-aligned: aligned loads and a constant shift, where an unaligned load
// computes it from the address. An odd byte, only one-byte pixels' (never AIE1's), loads unaligned: its shift is native.
template<int N, int OFF, int A, typename T>
aie::vector<T, N> load_at(const T* row)
{
  constexpr int G = std::min(A, N), BASE = OFF / G * G, SHIFT = OFF - BASE;
  if constexpr (SHIFT % 2) {
    return aie::load_unaligned_v<N>(row + OFF, 1);
  } else if constexpr (SHIFT == 0) {
    return load_aligned<N, G>(row + BASE);
  } else {
    const aie::vector<T, N> lo = load_aligned<N, G>(row + BASE);
    using E = std::conditional_t<SHIFT % 4 == 0, int32, int16>;
    const aie::vector<T, N> hi = load_aligned<N, G>(row + BASE + N);
    return aie::shuffle_down_fill(lo.template cast_to<E>(), hi.template cast_to<E>(), SHIFT / sizeof(E))
        .template cast_to<T>();
  }
}

// Every SW-th unit of P bytes: halving the stride keeps the even half each time.
template<int SW, int P, typename T, unsigned N>
aie::vector<T, N / SW> every(const aie::vector<T, N>& v)
{
  if constexpr (SW == 1)
    return v;
  else
    return every<SW / 2, P>(aie::filter_even(v, P * SW / 2));
}

// Byte offset of window unit N = (ky, kx) from the source row its output row starts on, in pixels of P bytes.
template<typename ConfigT, int N, int P = ConfigT::UNIT>
inline constexpr int OFFSET = (N / ConfigT::KW * ConfigT::SRC_COLS + N % ConfigT::KW) * P;

// The alignment every step's `row` shares, in pixels of P bytes.
template<typename ConfigT, int P = ConfigT::UNIT>
inline constexpr int ROW_ALIGN =
  std::gcd(std::gcd(ConfigT::STRIDE_H * ConfigT::SRC_COLS * P, STEP * ConfigT::STRIDE_W * P), 32);

// One-byte pixels at stride 1 come in pairs: a window row's units kx and kx + 1 are a source pixel and its right
// neighbour, paired once per source pixel (`pair_pixels`) where each output row's steps would zip them again.
template<typename ConfigT>
inline constexpr bool PAIRED = ConfigT::UNIT == 1 && ConfigT::STRIDE_W == 1;

// Each source pixel of the rows the window reads with its right neighbour: 2-byte pixels.
template<typename ConfigT>
void pair_pixels(const typename ConfigT::data_t* __restrict src, typename ConfigT::data_t* __restrict pairs)
{
  constexpr int BYTES = ((ConfigT::ROWS - 1) * ConfigT::STRIDE_H + ConfigT::KH) * ConfigT::SRC_COLS;
  for (int i = 0; i < BYTES; i += 32)
    chess_prepare_for_pipelining
  {
    const auto [lo, hi] = aie::interleave_zip(aie::load_v<32>(src + i), aie::load_unaligned_v<32>(src + i + 1, 1), 1);
    aie::store_v(pairs + 2 * i, lo);
    aie::store_v(pairs + 2 * i + 32, hi);
  }
}

// Units N to N + G - 1 for a step, each pixel's G units in order, UNIT bytes each; units past the window are zero.
// STRIDE_W units of one window row are each pixel's next source pixels: one load, nothing to pick or interleave; so
// are two of one window row from the pixel pairs, where the source has them (`PAIRED`).
template<typename ConfigT, int N, int G>
aie::vector<typename ConfigT::data_t, STEP * G * ConfigT::UNIT> units(const typename ConfigT::data_t* row,
                                                                       const typename ConfigT::data_t* pairs)
{
  using T = typename ConfigT::data_t;
  constexpr int P = ConfigT::UNIT, SW = ConfigT::STRIDE_W, KW = ConfigT::KW;
  if constexpr (N >= ConfigT::KH * KW) {
    return aie::zeros<T, STEP * G * P>();
  } else if constexpr (PAIRED<ConfigT> && G == 2 && N % KW + G <= KW) {
    return load_at<STEP * 2, OFFSET<ConfigT, N, 2>, ROW_ALIGN<ConfigT, 2>>(pairs);
  } else if constexpr (G == SW && N % KW + G <= KW) {
    return load_at<STEP * SW * P, OFFSET<ConfigT, N>, ROW_ALIGN<ConfigT>>(row);
  } else if constexpr (G == 1) {
    return every<SW, P>(load_at<STEP * SW * P, OFFSET<ConfigT, N>, ROW_ALIGN<ConfigT>>(row));
  } else {
    const auto zipped = aie::interleave_zip(units<ConfigT, N, G / 2>(row, pairs),
                                            units<ConfigT, N + G / 2, G / 2>(row, pairs), G / 2 * P);
    return aie::concat(zipped.first, zipped.second);
  }
}

// Block B of the frame, row by row: each 8-byte pixel holds the window's units from block B's first.
template<typename ConfigT, int B>
void fold_block(const typename ConfigT::data_t* __restrict src, const typename ConfigT::data_t* __restrict pairs,
                typename ConfigT::data_t* __restrict frame)
{
  constexpr int P = ConfigT::UNIT, G = 8 / P;
  for (int oy = 0; oy < ConfigT::ROWS; ++oy)
    chess_prepare_for_pipelining
  {
    const int first = oy * ConfigT::STRIDE_H * ConfigT::SRC_COLS;
    const typename ConfigT::data_t* row = src + first * P;
    typename ConfigT::data_t* out = frame + (B * ConfigT::ROWS + oy) * ConfigT::COLS * 8;
    for (int s = 0; s < ConfigT::COLS / STEP; ++s)
      chess_unroll_loop(*)
    {
      const auto pixels = units<ConfigT, (ConfigT::FIRST_BLOCK + B) * G, G>(row + s * STEP * ConfigT::STRIDE_W * P,
                                                                             pairs + (first + s * STEP) * 2);
      aie::store_v(out + s * STEP * 8, pixels);
    }
  }
}

template<typename ConfigT, int... B>
void fold_blocks(const typename ConfigT::data_t* src, const typename ConfigT::data_t* pairs,
                 typename ConfigT::data_t* frame, std::integer_sequence<int, B...>)
{
  (fold_block<ConfigT, B>(src, pairs, frame), ...);
}
}  // namespace

template<typename ConfigT>
frame_fold<ConfigT>::frame_fold() {
  constexpr int P = ConfigT::UNIT;
  static_assert(sizeof(data_t) == 1 && (P == 1 || P == 2 || P == 4), "pixels of 1, 2 or 4 bytes, whole in a block");
#if defined(__AIENGINE__) && __AIE_ARCH__ == 10
  static_assert(P > 1, "AIE1 moves single bytes through 16-bit lanes: it folds pixels of 2 or 4 bytes");
#endif
  static_assert((ConfigT::STRIDE_W & (ConfigT::STRIDE_W - 1)) == 0 && ConfigT::STRIDE_W * P <= 8,
                "a step loads each unit's strided pixels as one vector");
  static_assert(ConfigT::COLS % STEP == 0, "the frame's columns are whole steps");
  static_assert(ConfigT::SRC_COLS * P % 32 == 0, "source rows start on a 32-byte boundary, where loads are aligned");
  static_assert(((((ConfigT::ROWS - 1) * ConfigT::STRIDE_H + ConfigT::KH - 1) * ConfigT::SRC_COLS +
                  ConfigT::COLS * ConfigT::STRIDE_W + ConfigT::KW - 1) * P + READ_SLACK) <= ConfigT::SRC_BYTES,
                "the source holds every row the window reads, and the slack its loads read past them");
  static_assert(ConfigT::FRAME_BYTES == ConfigT::CB * ConfigT::ROWS * ConfigT::COLS * 8, "frame size");
  constexpr int SRC_ROWS = (ConfigT::ROWS - 1) * ConfigT::STRIDE_H + ConfigT::KH;
  static_assert(!PAIRED<ConfigT> || SRC_ROWS * ConfigT::SRC_COLS + 32 <= ConfigT::SRC_BYTES,
                "pairing reads the rows and one vector past them");
}

template<typename ConfigT>
void frame_fold<ConfigT>::run(input_buffer<data_t>& src, output_buffer<data_t>& frame)
{
  constexpr auto blocks = std::make_integer_sequence<int, ConfigT::CB>{};
  if constexpr (PAIRED<ConfigT>) {
    // Twice the source: a step reads as far past the rows' pairs as it reads past the rows.
    alignas(32) static data_t pairs[2 * ConfigT::SRC_BYTES];
    pair_pixels<ConfigT>(src.data(), pairs);
    fold_blocks<ConfigT>(src.data(), pairs, frame.data(), blocks);
  } else {
    fold_blocks<ConfigT>(src.data(), src.data(), frame.data(), blocks);  // no pairs: their pointer goes unread
  }
}
