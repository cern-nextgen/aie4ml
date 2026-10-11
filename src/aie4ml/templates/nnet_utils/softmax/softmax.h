// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>
#include <type_traits>

#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class softmax_base {
public:
    using out_t = typename ConfigT::output_t;

    static constexpr int INV_SHIFT = ConfigT::INV_SHIFT;
    // The reciprocal numerator: (255<<INV_SHIFT) for uint8 codes (OUT_SHIFT=INV_SHIFT),
    // 32767 for int16 weights (OUT_SHIFT=0). floor(NUMER/sum) is the per-row scale.
    static constexpr int32_t NUMER = std::is_same_v<out_t, int16_t> ? 32767 : (255 << INV_SHIFT);

    softmax_base() {
#if defined(__AIENGINE__) && (__cplusplus >= 202002L)
        aie::set_rounding(ConfigT::ROUNDING);
        aie::set_saturation(ConfigT::SATURATION);
#endif
    }

    // Exact floor(NUMER/sum[r]) for a batch of rows. AIE-ML has no vector integer divide and a
    // scalar divide is a 32-step software routine, so seed the reciprocal with one vectorised
    // aie::inv (float) and correct to the exact quotient with a couple of integer steps. The
    // result is bit-identical to the linear kernel's (255<<INV_SHIFT)/sum.
    template <unsigned N>
    static void batched_reciprocal(const int32_t (&sum)[N], int32_t (&invq)[N]);

    template <unsigned N>
    static void batched_clb_reciprocal(const int32_t* sum, int32_t* invq);
};

// HCCS is a calibrated clipped-linear surrogate for attention softmax.
// It is intentionally integer-only. see https://arxiv.org/pdf/2604.02292v1
template <typename ConfigT>
class softmax_i8 : public softmax_base<ConfigT> {
public:
    using base  = softmax_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS      = ConfigT::ROWS;
    static constexpr int COLS      = ConfigT::COLS;
    static constexpr int VEC       = ConfigT::VEC;
    static constexpr int VECS      = COLS / VEC;
    static constexpr int INV_SHIFT = ConfigT::INV_SHIFT;
    static constexpr int OUT_SHIFT = std::is_same_v<out_t, int16_t> ? 0 : INV_SHIFT;

    softmax_i8(int16_t B_i, int8_t S_i, uint8_t DMAX_i);

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_i8::run);
    }

private:
    alignas(aie::vector_decl_align) int16_t scratch[COLS];

    int16_t B_param;
    int8_t S_param;
    uint8_t DMAX_param;

    inline __attribute__((always_inline))
    void softmax_row(const int8* __restrict in_ptr,
                     out_t* __restrict out_ptr,
                     int16_t B,
                     int8_t S,
                     uint8_t DMAX);
};

template <typename ConfigT>
class softmax_i8_tiled : public softmax_base<ConfigT> {
public:
    using base  = softmax_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS      = ConfigT::ROWS;
    static constexpr int COLS      = ConfigT::COLS;
    static constexpr int MT_OUTER  = ConfigT::MICROTILE_OUTER;
    static constexpr int MT_INNER  = ConfigT::MICROTILE_INNER;
    static constexpr int BLK       = MT_OUTER * MT_INNER;   // lanes in one microtile block
    static constexpr int NB        = COLS / MT_INNER;       // feature blocks per row band
    static constexpr int OUT_SHIFT = std::is_same_v<out_t, int16_t> ? 0 : ConfigT::INV_SHIFT;
    // The float reciprocal wants at least a 4-lane vector; pad the stat lanes to that.
    static constexpr int STAT_LANES = (MT_OUTER < 4) ? 4 : MT_OUTER;
    // Microtiles per 64-byte vector: as many as fit and divide the band.
    static constexpr int P = (BLK <= 16 && NB % 4 == 0) ? 4 : (BLK <= 32 && NB % 2 == 0) ? 2 : 1;
    static_assert(MT_INNER == 8 && (MT_OUTER == 2 || MT_OUTER == 4 || MT_OUTER == 8),
                  "the int8 microtiles are 2x8, 4x8 or 8x8");
    // Between passes: each row's max and scale across its lanes (AIE-ML, AIE-MLv2), or one band's int16 scores (AIE1).
    static constexpr int STATS_BYTES  = ROWS * MT_INNER * (1 + 2);
    static constexpr int SCRATCH_BYTES = STATS_BYTES > NB * BLK * 2 ? STATS_BYTES : NB * BLK * 2;
    static_assert(!ConfigT::USE_CLB || std::is_same_v<out_t, int16_t> || (ConfigT::INV_SHIFT >= 14 && ConfigT::INV_SHIFT <= 15),
                  "the uint8 CLB scale 255 << (INV_SHIFT - k) must be a whole int16 for every k in [8, 14]");

    softmax_i8_tiled(int16_t B_i, int8_t S_i, uint8_t DMAX_i);

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_i8_tiled::run);
    }

private:
    int16_t B_param;
    int8_t S_param;
    uint8_t DMAX_param;
    alignas(aie::vector_decl_align) int8_t scratch_[SCRATCH_BYTES];

    static uint32_t clb_code(int32_t q);
    static int16_t exact_scale(int32_t z);
    static constexpr uint64_t diagonal_lanes();
};


template <typename ConfigT>
class softmax_exp_i8_tiled : public softmax_base<ConfigT> {
public:
    using base  = softmax_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS       = ConfigT::ROWS;
    static constexpr int COLS       = ConfigT::COLS;
    static constexpr int MT_OUTER   = ConfigT::MICROTILE_OUTER;
    static constexpr int MT_INNER   = ConfigT::MICROTILE_INNER;
    static constexpr int BLK        = MT_OUTER * MT_INNER;
    static constexpr int NB         = COLS / MT_INNER;
    static constexpr int OUT_SHIFT  = std::is_same_v<out_t, int16_t> ? 0 : ConfigT::INV_SHIFT;
    static constexpr int STAT_LANES = (MT_OUTER < 4) ? 4 : MT_OUTER;

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_exp_i8_tiled::run);
    }
};

template <typename ConfigT>
class softmax_exp_i8 : public softmax_base<ConfigT> {
public:
    using base  = softmax_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS      = ConfigT::ROWS;
    static constexpr int COLS      = ConfigT::COLS;
    static constexpr int VEC       = ConfigT::VEC;
    static constexpr int VECS      = COLS / VEC;
    static constexpr int OUT_SHIFT = std::is_same_v<out_t, int16_t> ? 0 : ConfigT::INV_SHIFT;

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_exp_i8::run);
    }
};
