#pragma once

#include <adf.h>
#include <aie_api/aie.hpp>

#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class layernorm_base {
public:
    layernorm_base() {
#if defined(__AIENGINE__) && (__cplusplus >= 202002L)
        aie::set_rounding(ConfigT::ROUNDING);
        aie::set_saturation(ConfigT::SATURATION);
#endif
    }

    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS        = ConfigT::ROWS;
    static constexpr int COLS        = ConfigT::COLS;
    static constexpr int VEC         = ConfigT::VEC;
    static constexpr int VECS        = COLS / VEC;
    static constexpr int GAMMA_SHIFT = ConfigT::GAMMA_SHIFT;
    static constexpr int OUT_SHIFT   = ConfigT::OUT_SHIFT;
    static constexpr int NORM_SHIFT  = 15;

    static constexpr int LOG2_COLS = []() constexpr {
        int n = COLS, c = 0;
        while (n > 1) { n >>= 1; ++c; }
        return c;
    }();

protected:
    template <unsigned N>
    static void row_statistics(const int32_t (&sum_x)[N],
                               const int32_t (&sum_sq)[N],
                               int16_t (&mu16)[N],
                               int16_t (&inv_std16)[N]);
};


// layernorm_i8 -- fully-integer LayerNorm over row-contiguous data, int8 -> int8.
//
// Rows are processed a batch at a time in three phases: reductions, the statistics above,
// then the normalise. Each phase's iterations are independent, so they pipeline; doing one
// row end to end instead leaves the scalar statistics stranded between two vector blocks
// with nothing to overlap them.
template <typename ConfigT>
class layernorm_i8 : public layernorm_base<ConfigT> {
public:
    using base  = layernorm_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int ROWS        = base::ROWS;
    static constexpr int COLS        = base::COLS;
    static constexpr int VEC         = base::VEC;
    static constexpr int VECS        = base::VECS;
    static constexpr int GAMMA_SHIFT = base::GAMMA_SHIFT;
    static constexpr int OUT_SHIFT   = base::OUT_SHIFT;
    static constexpr int NORM_SHIFT  = base::NORM_SHIFT;
    static constexpr int LOG2_COLS   = base::LOG2_COLS;

    // Rows whose reductions are in flight together. Larger batches pipeline better, so take
    // the biggest power of two up to 8 that divides the tile -- a tile of 6 rows batches 2 at
    // a time, one of 9 batches singly. Capped at 8 so the batch never scales with the tile:
    // it sizes the vectors below, and a lane count must stay a legal vector width.
    static constexpr int ROWS_PER_BATCH =
        (ROWS % 8 == 0) ? 8 : ((ROWS % 4 == 0) ? 4 : ((ROWS % 2 == 0) ? 2 : 1));

    // Lanes the statistics vector carries. Independent of the batch: 4 x 32b = 128b is the
    // narrowest legal vector, so a batch of 1 or 2 rows still computes on 4 lanes and leaves
    // the surplus ones idle. Their variance reads as zero, which the epsilon floor makes safe
    // to take an inverse square root of, and their results are simply never stored.
    static constexpr int STAT_LANES = (ROWS_PER_BATCH < 4) ? 4 : ROWS_PER_BATCH;

    static_assert(ROWS % ROWS_PER_BATCH == 0, "ROWS per tile must be a whole number of batches");

    layernorm_i8() : base() {}

    void run(input_buffer<in_t>&           in,
             const int16_t (&gamma)[COLS],
             const int16_t (&beta)[COLS],
             output_buffer<out_t>&         out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(layernorm_i8::run);
    }
};


// layernorm_i8_tiled -- the same normalisation over microtiles.
//
// A block holds one lane group per row, so accumulating across the feature axis keeps the rows
// separate and the totals need only a segmented reduce -- log2(inner) rounds once per row
// band, against a full cross-lane reduce per row in the linear kernel.
//
// The per-row statistics are widened to lane groups with a concat of broadcasts, once per row
// band. gamma/beta do not vary along rows at all, so they are widened once in ROM by the packer
// and the innermost loop just loads a block.
template <typename ConfigT>
class layernorm_i8_tiled : public layernorm_base<ConfigT> {
public:
    using base  = layernorm_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int MT_OUTER = ConfigT::MICROTILE_OUTER;   // rows per microtile
    static constexpr int MT_INNER = ConfigT::MICROTILE_INNER;   // features per microtile
    static constexpr int BLK      = MT_OUTER * MT_INNER;        // elements in one microtile

    static constexpr int ROWS        = base::ROWS;
    static constexpr int COLS        = base::COLS;
    static constexpr int GAMMA_SHIFT = base::GAMMA_SHIFT;
    static constexpr int OUT_SHIFT   = base::OUT_SHIFT;
    static constexpr int NORM_SHIFT  = base::NORM_SHIFT;
    static constexpr int LOG2_COLS   = base::LOG2_COLS;

    static constexpr int NB = COLS / MT_INNER;                  // microtiles across the features

    // Row bands one statistics call covers, and the lanes it carries; the surplus lanes read zero variance,
    // which the epsilon floor makes safe to invert, and are never stored. AIE-ML and AIE-MLv2 emulate the
    // float inverse square root as one latency-bound chain, which takes up to 16 rows for the time of one;
    // AIE1 has a float unit, where a call per band costs least.
    static constexpr int BANDS = ROWS / MT_OUTER;
    static constexpr int GROUP = []() constexpr {
        int group = 1;
        for (int g = 1; __AIE_ARCH__ != 10 && g * MT_OUTER <= 16; ++g)
            if (BANDS % g == 0) group = g;
        return group;
    }();
    static constexpr int STAT_LANES = (GROUP * MT_OUTER <= 4) ? 4 : (GROUP * MT_OUTER <= 8) ? 8 : 16;

    static_assert(ROWS % MT_OUTER == 0, "ROWS must be a whole number of microtile row bands");
    static_assert(COLS % MT_INNER == 0, "COLS must be a whole number of microtiles");
    static_assert(BLK * sizeof(in_t) >= 16, "a microtile must fill at least the 128b minimum vector");

    static_assert(MT_INNER >= 8, "tiled LayerNorm needs MT_INNER >= 8 (int16 lane group must reach 128b)");
    static_assert(MT_OUTER >= 2, "tiled LayerNorm needs at least 2 rows per microtile");
    static_assert(MT_OUTER <= 8, "tiled LayerNorm supports at most 8 rows per microtile");
    static_assert(BLK <= 64, "tiled LayerNorm needs BLK <= 64 (widened int16 vector must fit 1024b)");
    static_assert((MT_INNER & (MT_INNER - 1)) == 0, "MT_INNER must be a power of two (segmented reduce halves it)");

    layernorm_i8_tiled() : base() {}

    // gamma/beta arrive pre-widened: NB blocks of BLK, each the microtile's feature slice
    // repeated once per row. The packer builds this (pack_layernorm_param microtile=), so the
    // innermost loop is a single block load with no broadcast or concat.
    void run(input_buffer<in_t>&           in,
             const int16_t (&gamma)[COLS * MT_OUTER],
             const int16_t (&beta)[COLS * MT_OUTER],
             output_buffer<out_t>&         out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(layernorm_i8_tiled::run);
    }
};
