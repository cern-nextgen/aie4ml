// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include "frame_retile.h"

using namespace adf;

namespace {
// An 8-byte channel block is not a vector on any generation, so it moves as two words; naming
// the word type `may_alias` makes that defined on an int8 buffer.
using retile_word_t __attribute__((may_alias)) = int32;

// Slot S of a four-pixel group: loaded from `px` when S is in FROM..TO-1, else zero border.
template<int S, int FROM, int TO, typename T>
aie::vector<T, 16> group_pixel(const T* px)
{
  if constexpr (S >= FROM && S < TO)
    return aie::load_v<16>(px + (S - FROM) * 16);
  else
    return aie::zeros<T, 16>();
}

// A group's four stores: zipping its halves by 8-byte block leaves each phase of each block in a quarter.
template<int FROM, int TO, unsigned ALIGN, typename T>
auto split_group(const T* px)
{
  if constexpr (FROM == 0 && TO == 4)
    return aie::interleave_zip(aie::load_unaligned_v<32>(px, ALIGN), aie::load_unaligned_v<32>(px + 32, ALIGN), 8);
  else
    return aie::interleave_zip(aie::concat(group_pixel<0, FROM, TO>(px), group_pixel<1, FROM, TO>(px)),
                               aie::concat(group_pixel<2, FROM, TO>(px), group_pixel<3, FROM, TO>(px)), 8);
}

// N bytes from byte OFF of a source row that starts ROW_ALIGN-aligned. The alignment must reach the
// load as a constant: computed in the call, it compiles to a byte-by-byte copy.
template<int N, int OFF, unsigned ROW_ALIGN, typename T>
aie::vector<T, N> load_at(const T* row)
{
  constexpr unsigned ALIGN = std::gcd(ROW_ALIGN, std::gcd(unsigned(OFF), unsigned(N)));
  return aie::load_unaligned_v<N>(row + OFF, ALIGN);
}

// Zeroes BYTES bytes from `p`, which is ALIGN-aligned (16 or more): 32-byte stores where they fit.
template<int BYTES, unsigned ALIGN, typename T>
void zero_bytes(T* p)
{
  static_assert(BYTES % 16 == 0 && ALIGN % 16 == 0, "whole 16-byte vectors");
  constexpr int HEAD = ALIGN % 32 != 0 && BYTES >= 16 ? 16 : 0;
  constexpr int WHOLE = (BYTES - HEAD) / 32, TAIL = BYTES - HEAD - 32 * WHOLE;
  if constexpr (HEAD > 0)
    aie::store_v(p, aie::zeros<T, 16>());
  for (int i = 0; i < WHOLE; ++i)
    chess_flatten_loop
  {
    aie::store_v(p + HEAD + 32 * i, aie::zeros<T, 32>());
  }
  if constexpr (TAIL > 0)
    aie::store_v(p + HEAD + 32 * WHOLE, aie::zeros<T, 16>());
}

// Zeroes what the column groups leave unwritten: the rows above and below the image, and per image
// row three runs -- before phase 0's groups, between the phases, after phase 1's.
template<typename ConfigT, int IMAGE_ROWS, typename T>
void clear_border(T* out)
{
  constexpr int PB = ConfigT::COLS / 2 * 8, RB = 2 * PB, CHB = ConfigT::ROWS * RB;
  constexpr int START = ConfigT::ORIGIN_C / 4 * 4 / 2 * 8;  // bytes into a phase where the groups start
  constexpr int END = (ConfigT::ORIGIN_C + ConfigT::SRC_W + 3) / 4 * 4 / 2 * 8;
  constexpr int ABOVE = ConfigT::ORIGIN_R * RB, BELOW = CHB - (ConfigT::ORIGIN_R + IMAGE_ROWS) * RB;
  constexpr unsigned ROW_ALIGN = std::gcd(std::gcd(RB, CHB), 32);
  for (int cb = 0; cb < ConfigT::CB; ++cb) {
    T* const block = out + cb * CHB;
    zero_bytes<ABOVE, std::gcd(CHB, 32)>(block);
    zero_bytes<BELOW, std::gcd(CHB - BELOW, std::gcd(CHB, 32))>(block + CHB - BELOW);
    for (int r = 0; r < IMAGE_ROWS; ++r)
      chess_prepare_for_pipelining
    {
      T* const row = block + (ConfigT::ORIGIN_R + r) * RB;
      zero_bytes<START, ROW_ALIGN>(row);
      zero_bytes<PB - END + START, std::gcd(ROW_ALIGN, std::gcd(END, 32))>(row + END);
      zero_bytes<PB - END, std::gcd(ROW_ALIGN, std::gcd(PB + END, 32))>(row + PB + END);
    }
  }
}
}  // namespace

template<typename ConfigT>
frame_retile<ConfigT>::frame_retile() {
  constexpr int W = ConfigT::SRC_W, C = ConfigT::SRC_C;
  constexpr int H = ConfigT::ROWS - ConfigT::ORIGIN_R < ConfigT::SRC_H ? ConfigT::ROWS - ConfigT::ORIGIN_R
                                                                      : ConfigT::SRC_H;  // image rows it reads
  constexpr bool LINEAR = ConfigT::SRC_BASE == 0 && ConfigT::SRC_PIXEL == C && ConfigT::SRC_ROW == W * C &&
                          ConfigT::SRC_BLOCK == 8;
  constexpr bool BLOCKED = C % 8 == 0 && ConfigT::SRC_PIXEL == 8 && ConfigT::SRC_ROW >= W * 8 &&
                           ConfigT::SRC_BLOCK >= H * ConfigT::SRC_ROW && ConfigT::SRC_BASE % 8 == 0;
  static_assert(LINEAR || BLOCKED, "the source is a linear tensor or a channel-blocked frame");
  static_assert(sizeof(data_t) == 1, "the frame holds 8-channel blocks of bytes");
  static_assert(ConfigT::SRC_BASE + (ConfigT::CB - 1) * ConfigT::SRC_BLOCK + (H - 1) * ConfigT::SRC_ROW +
                    (W - 1) * ConfigT::SRC_PIXEL + (C - (ConfigT::CB - 1) * 8) <= ConfigT::SRC_BYTES,
                "the source holds the image rows it reads");
  static_assert(ConfigT::CB * 8 >= ConfigT::SRC_C && ConfigT::CB * 8 < ConfigT::SRC_C + 8,
                "the frame has exactly the channel blocks the tensor needs");
  static_assert(ConfigT::COLS % ConfigT::STRIDE == 0, "frame columns divide into phases");
  static_assert(ConfigT::ORIGIN_C + ConfigT::SRC_W <= ConfigT::COLS, "the tensor fits the frame columns");
  static_assert(ConfigT::ORIGIN_R < ConfigT::ROWS, "the tensor starts inside the frame");
  static_assert(ConfigT::FRAME_BYTES == ConfigT::CB * ConfigT::ROWS * ConfigT::COLS * 8, "frame size");
  static_assert(ConfigT::FRAME_BYTES % 16 == 0, "the frame clears in whole vectors");
}

template<typename ConfigT>
void frame_retile<ConfigT>::run(input_buffer<data_t>& src, output_buffer<data_t>& frame)
{
  constexpr int S = ConfigT::STRIDE, W = ConfigT::SRC_W, C = ConfigT::SRC_C;
  constexpr int PW = ConfigT::COLS / S;
  constexpr int RB = ConfigT::COLS * 8, CHB = ConfigT::ROWS * RB;
  constexpr int AVAIL = ConfigT::ROWS - ConfigT::ORIGIN_R;
  constexpr int ROWS = AVAIL < ConfigT::SRC_H ? AVAIL : ConfigT::SRC_H;  // image rows the frame holds
  const data_t* __restrict in = src.data() + ConfigT::SRC_BASE;
  data_t* __restrict out = frame.data();

  // Four-pixel group paths, for one-block pixels (NARROW) and linear two-block pixels (WIDE): they
  // need an even origin, vector-aligned phases and whole four-column groups in the frame.
  constexpr bool GROUPED = S == 2 && ConfigT::ORIGIN_C % 2 == 0 && ConfigT::COLS % 4 == 0 &&
                           (ConfigT::ORIGIN_C + W + 3) / 4 * 4 <= ConfigT::COLS;
  constexpr bool NARROW = GROUPED && ConfigT::SRC_PIXEL == 8 && W >= 4;
  constexpr bool WIDE = GROUPED && C == 16 && ConfigT::SRC_BASE == 0 && ConfigT::SRC_PIXEL == 16 &&
                        ConfigT::SRC_ROW == W * 16;

  // The frame is written whole every call: an output buffer alternates and holds nothing the last
  // call left, so the border and the unused channel lanes are cleared here, not assumed.
  if constexpr (NARROW || WIDE) {
    clear_border<ConfigT, ROWS>(out);
  } else {
    const auto zero = aie::zeros<data_t, 16>();
    for (int i = 0; i < ConfigT::FRAME_BYTES / 16; ++i)
      chess_prepare_for_pipelining
    {
      aie::store_v(out + i * 16, zero);
    }
  }

  if constexpr (NARROW) {
    // A group's even pixels are phase 0, its odd ones phase 1. Border slots of the first and last
    // group are zero; the last pixels load in pairs, a lone one from the pair ending on it, so no
    // load leaves the row.
    constexpr int FIRST = ConfigT::ORIGIN_C % 4;  // 0 or 2, the image's first pixel within its group
    constexpr int LEAD = FIRST == 0 ? 0 : 2;
    constexpr int WHOLE = (W - LEAD) / 4, TAIL = (W - LEAD) % 4, LAST = LEAD + 4 * WHOLE;
    constexpr int BLOCK_STEP = ConfigT::CB > 1 ? ConfigT::SRC_BLOCK : 0;  // no step with one block
    constexpr unsigned ROW_ALIGN =
      std::gcd(std::gcd(ConfigT::SRC_BASE, ConfigT::SRC_ROW), std::gcd(BLOCK_STEP, 32));
    constexpr unsigned GROUP_ALIGN = std::gcd(ROW_ALIGN, std::gcd(LEAD * 8u, 32u));
    const auto zero = aie::zeros<data_t, 16>();
    auto last_of = [&](const aie::vector<data_t, 16>& pair) { return aie::shuffle_down_fill(pair, zero, 8); };
    for (int cb = 0; cb < ConfigT::CB; ++cb)
    for (int r = 0; r < ROWS; ++r)
      chess_prepare_for_pipelining
    {
      const data_t* const src = in + cb * ConfigT::SRC_BLOCK + r * ConfigT::SRC_ROW;
      data_t* __restrict p0 = out + cb * CHB + (ConfigT::ORIGIN_R + r) * RB + (ConfigT::ORIGIN_C - FIRST) / 2 * 8;
      data_t* __restrict p1 = p0 + PW * 8;
      auto store_group = [&](const aie::vector<data_t, 32>& v) {
        aie::store_v(p0, aie::filter_even(v, 8));
        aie::store_v(p1, aie::filter_odd(v, 8));
        p0 += 16, p1 += 16;
      };
      if constexpr (LEAD > 0)
        store_group(aie::concat(zero, load_at<16, 0, ROW_ALIGN>(src)));
      for (int k = 0; k < WHOLE; ++k)
        chess_flatten_loop
      {
        store_group(aie::load_unaligned_v<32>(src + (LEAD + 4 * k) * 8, GROUP_ALIGN));
      }
      if constexpr (TAIL == 1)
        store_group(aie::concat(last_of(load_at<16, (LAST - 1) * 8, ROW_ALIGN>(src)), zero));
      else if constexpr (TAIL == 2)
        store_group(aie::concat(load_at<16, LAST * 8, ROW_ALIGN>(src), zero));
      else if constexpr (TAIL == 3)
        store_group(aie::concat(load_at<16, LAST * 8, ROW_ALIGN>(src),
                                last_of(load_at<16, (LAST + 1) * 8, ROW_ALIGN>(src))));
    }
  } else if constexpr (S == 2 && ConfigT::SRC_PIXEL == 8 && ConfigT::ORIGIN_C % 2 == 0) {
    // Four pixels a load; even ones go to phase 0 and odd ones to phase 1, one 16-byte store each.
    // Those stores are aligned only where the frame column is a multiple of four, so the pixels
    // before the first such column, and those after the last whole chunk, move one at a time. The
    // counts are constants, so their loops unroll and the loop over rows pipelines. A source row
    // starts only as aligned as the steps that reach it, which the loads are told.
    static_assert((PW * 8) % 16 == 0 && RB % 16 == 0, "both phases of every row start on a vector");
    constexpr int BLOCK_STEP = ConfigT::CB > 1 ? ConfigT::SRC_BLOCK : 0;  // no step with one block
    constexpr unsigned SRC_ALIGN =
      std::gcd(std::gcd(ConfigT::SRC_BASE, ConfigT::SRC_ROW), std::gcd(BLOCK_STEP, 16));
    constexpr int ALIGN = (4 - ConfigT::ORIGIN_C % 4) % 4;
    constexpr int LEAD = W < ALIGN ? W : ALIGN;
    constexpr int CHUNKS = (W - LEAD) / 4, TAIL = (W - LEAD) % 4;
    auto move_pixel = [&](data_t* row, const data_t* src, int c) {
      const int col = ConfigT::ORIGIN_C + c;
      const retile_word_t* sw = reinterpret_cast<const retile_word_t*>(src + c * 8);
      retile_word_t* dw = reinterpret_cast<retile_word_t*>(row + (col % 2) * PW * 8 + (col / 2) * 8);
      dw[0] = sw[0];
      dw[1] = sw[1];
    };
    for (int cb = 0; cb < ConfigT::CB; ++cb)
    for (int r = 0; r < ROWS; ++r)
      chess_prepare_for_pipelining
    {
      const data_t* const src = in + cb * ConfigT::SRC_BLOCK + r * ConfigT::SRC_ROW;
      data_t* const row = out + cb * CHB + (ConfigT::ORIGIN_R + r) * RB;
      for (int c = 0; c < LEAD; ++c)
        chess_flatten_loop
      {
        move_pixel(row, src, c);
      }
      for (int k = 0; k < CHUNKS; ++k)
        chess_flatten_loop
      {
        const int c = LEAD + 4 * k;
        const int col = ConfigT::ORIGIN_C + c;  // a multiple of four: both stores are aligned
        const aie::vector<data_t, 32> v = aie::load_unaligned_v<32>(src + c * 8, SRC_ALIGN);
        aie::store_v(row + (col / 2) * 8, aie::filter_even(v, 8));
        aie::store_v(row + PW * 8 + (col / 2) * 8, aie::filter_odd(v, 8));
      }
      for (int t = 0; t < TAIL; ++t)
        chess_flatten_loop
      {
        move_pixel(row, src, LEAD + 4 * CHUNKS + t);
      }
    }
  } else if constexpr (WIDE) {
    // Each group is four 16-byte stores, a phase of each block; border slots of the first and last
    // group are zero, so no load leaves the row.
    static_assert(ConfigT::CB == 2 && (PW * 8) % 16 == 0 && RB % 16 == 0,
                  "both phases of every row start on a vector");
    constexpr int FIRST = ConfigT::ORIGIN_C % 4;  // the image's first pixel within its group
    constexpr int LEAD = FIRST == 0 ? 0 : (W < 4 - FIRST ? W : 4 - FIRST);
    constexpr int GROUPS = (W - LEAD) / 4, TAIL = (W - LEAD) % 4;
    // Groups start 16-byte aligned, and 32 where every step that reaches one is.
    constexpr unsigned GROUP_ALIGN = std::gcd(std::gcd(ConfigT::SRC_ROW, LEAD * 16), 32);
    for (int r = 0; r < ROWS; ++r)
      chess_prepare_for_pipelining
    {
      const data_t* const src = in + r * ConfigT::SRC_ROW;
      data_t* __restrict b0p0 = out + (ConfigT::ORIGIN_R + r) * RB + (ConfigT::ORIGIN_C - FIRST) / 2 * 8;
      data_t* __restrict b0p1 = b0p0 + PW * 8;
      data_t* __restrict b1p0 = b0p0 + CHB;
      data_t* __restrict b1p1 = b1p0 + PW * 8;
      auto store_group = [&](const auto& zipped) {
        aie::store_v(b0p0, zipped.first.template extract<16>(0));
        aie::store_v(b1p0, zipped.first.template extract<16>(1));
        aie::store_v(b0p1, zipped.second.template extract<16>(0));
        aie::store_v(b1p1, zipped.second.template extract<16>(1));
        b0p0 += 16, b0p1 += 16, b1p0 += 16, b1p1 += 16;
      };
      if constexpr (LEAD > 0)
        store_group(split_group<FIRST, FIRST + LEAD, 16>(src));
      for (int k = 0; k < GROUPS; ++k)
        chess_flatten_loop
      {
        store_group(split_group<0, 4, GROUP_ALIGN>(src + (LEAD + 4 * k) * 16));
      }
      if constexpr (TAIL > 0)
        store_group(split_group<0, TAIL, 16>(src + (LEAD + 4 * GROUPS) * 16));
    }
  } else {
    // One phase at a time, so the frame side advances one pixel slot per step.
    for (int r = 0; r < ROWS; ++r) {
      for (int p = 0; p < S; ++p) {
        const int col = ConfigT::ORIGIN_C + p;
        const int n = (W - p + S - 1) / S;
        for (int cb = 0; cb < ConfigT::CB; ++cb) {
          const data_t* s = in + cb * ConfigT::SRC_BLOCK + r * ConfigT::SRC_ROW + p * ConfigT::SRC_PIXEL;
          data_t* d = out + cb * CHB + (ConfigT::ORIGIN_R + r) * RB + (col % S) * PW * 8 + (col / S) * 8;
          if constexpr (C % 8 == 0) {  // whole blocks, and every pixel starts word-aligned
            for (int i = 0; i < n; ++i)
              chess_prepare_for_pipelining
            {
              const retile_word_t* sw = reinterpret_cast<const retile_word_t*>(s + i * S * ConfigT::SRC_PIXEL);
              retile_word_t* dw = reinterpret_cast<retile_word_t*>(d + i * 8);
              dw[0] = sw[0];
              dw[1] = sw[1];
            }
          } else {
            // The last block of a channel count that does not fill it leaves its other lanes zero.
            const int lanes = C - cb * 8 < 8 ? C - cb * 8 : 8;
            for (int i = 0; i < n; ++i)
              chess_prepare_for_pipelining
            {
              for (int k = 0; k < lanes; ++k) d[i * 8 + k] = s[i * S * ConfigT::SRC_PIXEL + k];
            }
          }
        }
      }
    }
  }
}
