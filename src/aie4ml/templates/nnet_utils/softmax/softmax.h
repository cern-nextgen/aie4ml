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

    softmax_i8_tiled(int16_t B_i, int8_t S_i, uint8_t DMAX_i);

    void run(input_buffer<in_t>& in, output_buffer<out_t>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(softmax_i8_tiled::run);
    }

private:
    int16_t B_param;
    int8_t S_param;
    uint8_t DMAX_param;
    alignas(aie::vector_decl_align) int16_t scores_[NB * BLK];  // one row band's scores, between passes 2 and 3
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
