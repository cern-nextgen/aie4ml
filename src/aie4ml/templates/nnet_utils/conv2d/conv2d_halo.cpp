// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_halo.h"

#include "conv2d_core_one_block.h"

using namespace adf;

template<typename ConfigT, int B>
conv2d_halo_base<ConfigT, B>::conv2d_halo_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
  conv2d_check_contract<ConfigT>();
  static_assert(ConfigT::CAS_LENGTH == 1 && ConfigT::PARALLELISM_CONTRACT_OUTER && !ConfigT::STREAM_IO,
                "a halo band is one buffer-port tile of a row split");
  if constexpr (READS) {
    static_assert(role::BELOW >= 0 && ConfigT::IN_ROWS == role::ABOVE + ConfigT::OWN_ROWS + role::BELOW &&
                      ConfigT::IN_ORIGIN_R == 0 && ConfigT::IN_H == ConfigT::IN_ROWS && ConfigT::FILLS_BORDER,
                  "the window is the halo and the band's own rows; its column border is zeroed by the core");
    static_assert(
        (!role::TOP || role::ABOVE <= ConfigT::OWN_ROWS) && (!role::BOTTOM || role::BELOW <= ConfigT::OWN_ROWS),
        "a band's halo comes from its neighbours' own rows alone");
    static_assert(ConfigT::IN_ELEMENTS == ConfigT::CB * ConfigT::IN_ROWS * ConfigT::IN_COLS * 8,
                  "the window is whole frame rows of every channel block");
  }
  if constexpr (ConfigT::SEND_FIRST + ConfigT::SEND_LAST > 0) {
    constexpr int ROWS = conv2d_rows<ConfigT> / (ConfigT::POOL ? 2 : 1);
    static_assert(!ConfigT::FLATTEN && ConfigT::OUT_ELEMENTS == ConfigT::NB * ConfigT::OUT_ROWS * ConfigT::OUT_COLS * 8,
                  "a band writes whole frame rows of every channel block");
    static_assert(ConfigT::OUT_ORIGIN_R == ConfigT::SEND_LAST &&
                      ConfigT::OUT_ROWS == ConfigT::SEND_LAST + ROWS + ConfigT::SEND_FIRST - ConfigT::SEND_STEP &&
                      role::ORIGIN + ROWS <= ConfigT::OUT_ROWS,
                  "its output buffer is its reader's window: its rows, and room for the halo its reader fills");
    static_assert(role::SENT_FIRST <= ROWS && role::SENT_LAST <= ROWS, "a band sends rows of its own");
  }
}

// `rows` whole rows of ROW elements from `src`, or zeros where `src` is null: 32-byte vectors, as frame rows are
// 32-byte aligned.
template<typename T, int ROW>
static inline void conv2d_halo_rows(T* __restrict dst, const T* __restrict src, int rows) {
  constexpr int V = 32 / sizeof(T);
  static_assert(ROW % V == 0, "frame rows are 32-byte aligned");
  const int n = rows * ROW;
  if (src) {
    for (int i = 0; i < n; i += V)
      chess_prepare_for_pipelining
    {
      aie::store_v(dst + i, aie::load_v<V>(src + i));
    }
  } else {
    const auto zero = aie::zeros<T, V>();
    for (int i = 0; i < n; i += V)
      chess_prepare_for_pipelining
    {
      aie::store_v(dst + i, zero);
    }
  }
}

template<typename ConfigT, int B>
void conv2d_halo_base<ConfigT, B>::compute(const data_t* own, const data_t* const* halo,
                                            const weight_t (&wts)[ConfigT::WN], const bias_t (&bias)[ConfigT::BN],
                                            result_t* out, result_t* const* edge) {
  data_t* window = const_cast<data_t*>(own);
  if constexpr (READS) {
    // Its own rows sit in the middle of its window: per channel block, fill the rows above them with the band
    // above's last rows and the rows below with the band below's first ones, or zeros where the image ends.
    constexpr int ROW = ConfigT::IN_COLS * 8, WINDOW = ConfigT::IN_ROWS * ROW;
    constexpr int HT = role::ABOVE, HB = role::BELOW;
    const data_t* top = role::TOP ? halo[0] : nullptr;
    const data_t* bottom = role::BOTTOM ? halo[role::TOP] : nullptr;
    for (int cb = 0; cb < ConfigT::CB; ++cb) {
      data_t* f = window + cb * WINDOW;
      conv2d_halo_rows<data_t, ROW>(f, top ? top + cb * HT * ROW : nullptr, HT);
      conv2d_halo_rows<data_t, ROW>(f + (HT + ConfigT::OWN_ROWS) * ROW, bottom ? bottom + cb * HB * ROW : nullptr, HB);
    }
  }
  // The core writes its rows from OUT_ORIGIN_R, band 0's place in its reader's window; this band's is ORIGIN.
  constexpr int ROW = ConfigT::OUT_COLS * 8, FRAME = ConfigT::OUT_ROWS * ROW;
  conv2d_compute<ConfigT, false, false>(window, wts, bias, out + (role::ORIGIN - ConfigT::OUT_ORIGIN_R) * ROW,
                                        nullptr, nullptr);
  // The rows its neighbours' windows read, per channel block: its first ones for the band above, its last ones
  // for the band below.
  constexpr int SF = role::SENT_FIRST, SL = role::SENT_LAST;
  constexpr int FIRST = role::ORIGIN, LAST = FIRST + conv2d_rows<ConfigT> / (ConfigT::POOL ? 2 : 1) - SL;
  for (int nb = 0; nb < ConfigT::NB; ++nb) {
    if constexpr (role::FIRST)
      conv2d_halo_rows<result_t, ROW>(edge[0] + nb * SF * ROW, out + nb * FRAME + FIRST * ROW, SF);
    if constexpr (role::LAST)
      conv2d_halo_rows<result_t, ROW>(edge[role::FIRST] + nb * SL * ROW, out + nb * FRAME + LAST * ROW, SL);
  }
}

// Each run() hands its ports, as it has them, to compute().
#define CONV2D_HALO_RUN(HALOS, EDGES)                                                                             \
  template<typename ConfigT, int B>                                                                               \
  void conv2d_halo<ConfigT, B, HALOS, EDGES>::run(input_buffer<data_t>& own CONV2D_HALO_IN_##HALOS,               \
                                                  const weight_t (&wts)[ConfigT::WN],                             \
                                                  const bias_t (&bias)[ConfigT::BN],                              \
                                                  output_buffer<result_t>& ofm CONV2D_HALO_OUT_##EDGES) {         \
    const data_t* halo[] = {CONV2D_HALO_INS_##HALOS nullptr};                                                     \
    result_t* edge[] = {CONV2D_HALO_OUTS_##EDGES nullptr};                                                        \
    this->compute(own.data(), halo, wts, bias, ofm.data(), edge);                                                 \
  }
CONV2D_HALO_RUN(0, 0)
CONV2D_HALO_RUN(0, 1)
CONV2D_HALO_RUN(0, 2)
CONV2D_HALO_RUN(1, 0)
CONV2D_HALO_RUN(1, 1)
CONV2D_HALO_RUN(1, 2)
CONV2D_HALO_RUN(2, 0)
CONV2D_HALO_RUN(2, 1)
CONV2D_HALO_RUN(2, 2)
#undef CONV2D_HALO_RUN
