// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv2d.h"
#include <utility>
#include "conv2d_core.h"

using namespace adf;

template<typename F, int... I>
static inline __attribute__((always_inline)) void depthwise_unroll(F&& f, std::integer_sequence<int, I...>)
{
  (f(std::integral_constant<int, I>{}), ...);
}

template<int N, typename F>
static inline __attribute__((always_inline)) void depthwise_unroll(F&& f)
{
  depthwise_unroll(f, std::make_integer_sequence<int, N>{});
}

// The 64 bytes at byte U of a row whose 32-byte chunks are ch[].
template<int U, int CHUNKS, typename T>
static inline __attribute__((always_inline)) aie::vector<T, 64> depthwise_window(const aie::vector<T, 32> (&ch)[CHUNKS])
{
  constexpr int I = U / 32, S = U % 32;
  constexpr auto at = [](int i) { return i < CHUNKS ? i : CHUNKS - 1; };
  const aie::vector<T, 64> lo = aie::concat(ch[at(I)], ch[at(I + 1)]);
  if constexpr (S == 0) {
    return lo;
  } else {
    const aie::vector<T, 64> hi = aie::concat(ch[at(I + 2)], ch[at(I + 3)]);
    return aie::shuffle_down_fill(lo.template cast_to<int32>(), hi.template cast_to<int32>(), S / 4)
        .template cast_to<T>();
  }
}

template<typename ConfigT>
struct depthwise_geometry {
  // One sliding_mul_ch: OUTS output pixels x 8 channels over PTS taps of a frame row.
  static constexpr int OUTS = __AIE_ARCH__ == 20 ? 4 : 8, PTS = OUTS;
  static constexpr int KWP = (ConfigT::KW + 3) / 4 * 4;  // taps per row in the packed weights
  static constexpr int PGROUPS = (ConfigT::KW + PTS - 1) / PTS;
  static constexpr int WIN = (OUTS + PTS) * 8;  // window bytes one call reads
  static constexpr int ZW = ConfigT::MB * ConfigT::M;
  static constexpr int ZBLOCKS = ConfigT::OUT_W_COMPUTED / ZW;
  // Columns past OUT_W are never read, so a single register block stops at the last group they need.
  static constexpr int GROUPS =
      ZBLOCKS == 1 ? std::min(ZW, (ConfigT::OUT_W + OUTS - 1) / OUTS * OUTS) / OUTS : ZW / OUTS;
  static constexpr int ORIGIN = (ConfigT::IN_ORIGIN_C - ConfigT::PAD_L) * 8;
  static constexpr int BASE = ORIGIN / 32 * 32;
  static constexpr int CHUNKS = (ORIGIN + ((GROUPS - 1) * OUTS + (PGROUPS - 1) * PTS) * 8 + WIN - BASE + 31) / 32;
  static constexpr int off(int q, int p) { return ORIGIN - BASE + (q * OUTS + p * PTS) * 8; }
};

template<typename ConfigT>
depthwise_conv2d_single<ConfigT>::depthwise_conv2d_single()
{
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
  conv2d_check_contract<ConfigT>();
  static_assert(__AIE_ARCH__ != 10, "sliding_mul_ch is AIE-ML and AIE-MLv2 only");
  static_assert(ConfigT::DEPTHWISE_CORE, "depthwise kernel requires its compact weight layout");
  static_assert(ConfigT::CAS_LENGTH == 1 && (ConfigT::CAS_NUM == 1 || ConfigT::PARALLELISM_CONTRACT_OUTER),
                "a depthwise tile owns the whole channel axis: row bands only");
  static_assert(ConfigT::CB == ConfigT::NB, "one input block per output block");
  static_assert(ConfigT::WN == ConfigT::NB * ConfigT::KH * depthwise_geometry<ConfigT>::KWP * 8,
                "weights hold each block's taps row by row");
  static_assert(ConfigT::STRIDE_W == 1, "the sliding window steps one pixel");
  static_assert(!ConfigT::POOL && !ConfigT::FLATTEN, "the depthwise core writes the normal Conv frame");
  static_assert(sizeof(typename ConfigT::data_t) == 1, "int8 pixels");
}

template<typename ConfigT>
void depthwise_conv2d_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                            const weight_t (&wts)[ConfigT::WN],
                                            const bias_t (&bias)[ConfigT::BN],
                                            output_buffer<result_t>& ofm)
{
  using G = conv2d_geometry<ConfigT>;
  using D = depthwise_geometry<ConfigT>;
  constexpr int KH = ConfigT::KH, OUTS = D::OUTS, PTS = D::PTS, L = OUTS * 8;
  using CH = aie::sliding_mul_ch_ops<OUTS, 8, PTS, 1, 1, 1, weight_t, data_t, typename ConfigT::acc_scalar_t>;

  data_t* frame = const_cast<data_t*>(ifm.data());
  result_t* out = ofm.data();
  if constexpr (ConfigT::FILLS_BORDER) conv2d_zero_border<ConfigT>(frame);

  for (int block = 0; block < ConfigT::NB; ++block) {
    const auto b = aie::load_v<8>(bias + block * 8);
    aie::vector<bias_t, L> bb;
    for (int m = 0; m < OUTS; ++m) bb.template insert<8>(m, b);

    // Each row's taps in PTS-wide groups; a group past the packed taps is zero.
    aie::vector<weight_t, PTS * 8> w[KH][D::PGROUPS];
    const weight_t* wb = wts + block * KH * D::KWP * 8;
    depthwise_unroll<KH>([&](auto ky) __attribute__((always_inline)) {
      depthwise_unroll<D::PGROUPS>([&](auto p) __attribute__((always_inline)) {
        const weight_t* src = wb + (ky * D::KWP + p * PTS) * 8;
        if constexpr (PTS * 8 == 32 || D::KWP - p * PTS >= PTS)
          w[ky][p] = aie::load_v<PTS * 8>(src);
        else
          w[ky][p] = aie::concat(aie::load_v<32>(src), aie::zeros<weight_t, 32>());
      });
    });

    auto rows = [&](int oy, int z) __attribute__((always_inline)) {
      // Frame rows, z * 8 and BASE are 32-byte aligned, so every chunk load is aligned.
      const data_t* row0 = frame + block * G::CHB + oy * ConfigT::STRIDE_H * G::RB + z * 8 + D::BASE;
      aie::accum<typename ConfigT::acc_scalar_t, L> C[D::GROUPS];
      depthwise_unroll<D::GROUPS>([&](auto q) __attribute__((always_inline)) { C[q].from_vector(bb); });

      depthwise_unroll<KH>([&](auto ky) __attribute__((always_inline)) {
        aie::vector<data_t, 32> ch[D::CHUNKS];
        depthwise_unroll<D::CHUNKS>([&](auto j) __attribute__((always_inline)) {
          ch[j] = aie::load_v<32>(row0 + ky * G::RB + j * 32);
        });
        depthwise_unroll<D::GROUPS>([&](auto q) __attribute__((always_inline)) {
          depthwise_unroll<D::PGROUPS>([&](auto p) __attribute__((always_inline)) {
            constexpr int U = D::off(decltype(q)::value, decltype(p)::value);
            if constexpr (D::WIN == 64) {
              C[q] = CH::mac(C[q], w[ky][p], 0, depthwise_window<U>(ch), 0);
            } else {
              C[q] = CH::mac(C[q], w[ky][p], 0, aie::concat(depthwise_window<U>(ch), depthwise_window<U + 64>(ch)), 0);
            }
          });
        });
      });

      depthwise_unroll<D::GROUPS>([&](auto q) __attribute__((always_inline)) {
        auto tile = C[q].template to_vector<result_t>(ConfigT::SHIFT);
        if constexpr (ConfigT::USE_RELU) tile = aie::max(tile, result_t(0));
        aie::store_v(out + ((block * ConfigT::OUT_ROWS + ConfigT::OUT_ORIGIN_R + oy) * ConfigT::OUT_COLS +
                            ConfigT::OUT_ORIGIN_C + z + q * OUTS) * 8,
                     tile);
      });
    };
    if constexpr (D::ZBLOCKS == 1) {
      for (int oy = 0; oy < ConfigT::OUT_H; ++oy) chess_prepare_for_pipelining { rows(oy, 0); }
    } else {
      for (int oy = 0; oy < ConfigT::OUT_H; ++oy)
        for (int z = 0; z < ConfigT::OUT_W_COMPUTED; z += D::ZW) chess_prepare_for_pipelining { rows(oy, z); }
    }
  }
}
