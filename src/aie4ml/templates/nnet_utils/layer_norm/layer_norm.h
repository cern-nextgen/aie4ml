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


// layernorm_i8_tiled -- the same normalisation over microtiles, in three passes over the row bands, each pipelined
// across bands: each row's sum and sum of squares; the statistics of all rows in vectors, one record [-mu | scale]
// per band; then y = (x - mu) * (gamma / sigma) + beta.
//
// gamma/beta arrive widened in ROM: NB blocks of BLK, each the microtile's feature slice repeated once per row.
template <typename ConfigT>
class layernorm_i8_tiled : public layernorm_base<ConfigT> {
public:
    using base  = layernorm_base<ConfigT>;
    using in_t  = typename ConfigT::input_t;
    using out_t = typename ConfigT::output_t;

    static constexpr int MT_OUTER = ConfigT::MICROTILE_OUTER;
    static constexpr int MT_INNER = ConfigT::MICROTILE_INNER;
    static constexpr int BLK      = MT_OUTER * MT_INNER;

    static constexpr int ROWS      = base::ROWS;
    static constexpr int COLS      = base::COLS;
    static constexpr int LOG2_COLS = base::LOG2_COLS;
    static constexpr int NB        = COLS / MT_INNER;
    static constexpr int BANDS     = ROWS / MT_OUTER;

    // Fraction bits: x - mu has MEAN_FRAC, the variance VAR_FRAC, 1/sigma 15 and gamma/sigma SCALE_FRAC.
    static constexpr int MEAN_FRAC  = ConfigT::MEAN_FRAC;
    static constexpr int VAR_FRAC   = 10;
    static constexpr int SCALE_FRAC = ConfigT::SCALE_FRAC;
    static constexpr int F_SHIFT    = ConfigT::GAMMA_SHIFT + 15 - SCALE_FRAC;
    static constexpr int B_SHIFT    = SCALE_FRAC + MEAN_FRAC - ConfigT::BETA_FRAC;
    static constexpr int O_SHIFT    = SCALE_FRAC + MEAN_FRAC - ConfigT::OUT_SHIFT;

    // 32 bytes on AIE-ML: chess miscompiles the last vector of a 64-byte int8 band loop there.
    static constexpr int VEC_BYTES = __AIE_ARCH__ == 10 ? 16 : __AIE_ARCH__ == 20 ? 32 : 64;
    static constexpr int P  = (BLK * 4 <= VEC_BYTES && NB % 4 == 0) ? 4 : (BLK * 2 <= VEC_BYTES && NB % 2 == 0) ? 2 : 1;
    static constexpr int VB = P * BLK;
    static constexpr int NV = NB / P;

    // Pass 1 stores [sums | squares] of each SUB-lane piece's PIECE_ROWS rows, or with ROW_TOTALS [sum, squares] of
    // each row: on AIE1, whose 32-bit shuffles are compare-and-select, and where a piece holds one row.
    static constexpr int LOG2_MT_INNER = MT_INNER == 8 ? 3 : MT_INNER == 16 ? 4 : 5;
    static constexpr int SUB           = BLK < 32 ? BLK : 32;
    static constexpr bool ROW_TOTALS   = __AIE_ARCH__ == 10 || SUB / MT_INNER < 2;
    static constexpr int PIECE_ROWS    = ROW_TOTALS ? 1 : SUB / MT_INNER;
    // Rows per statistics vector, each vector one latency-bound chain; AIE-MLv2 overflows its stack at 32.
    static constexpr int NS     = (__AIE_ARCH__ == 10 || ROWS <= 8) ? 8 : (ROWS < 32 || __AIE_ARCH__ != 20) ? 16 : 32;
    static constexpr int SROWS  = (ROWS + NS - 1) / NS * NS;
    static constexpr int SBANDS = SROWS / MT_OUTER;
    // A band's record: -mu and the scale of each row, padded to a 16-byte load.
    static constexpr int REC = 2 * MT_OUTER < 8 ? 8 : 2 * MT_OUTER;

    static_assert(ROWS % MT_OUTER == 0, "ROWS must be a whole number of microtile row bands");
    static_assert(COLS % MT_INNER == 0, "COLS must be a whole number of microtiles");
    static_assert((MT_INNER == 8 || MT_INNER == 16 || MT_INNER == 32) && (MT_OUTER == 2 || MT_OUTER == 4 ||
                  MT_OUTER == 8) && BLK <= 64, "2, 4 or 8 rows of 8, 16 or 32 features, 64 elements at most");
    static_assert(2 * LOG2_COLS >= VAR_FRAC, "at least 32 features");
    static_assert(F_SHIFT >= 0 && B_SHIFT >= 0 && O_SHIFT >= 0, "right shifts only");

    layernorm_i8_tiled() : base() {}

    void run(input_buffer<in_t>&           in,
             const int16_t (&gamma)[COLS * MT_OUTER],
             const int16_t (&beta)[COLS * MT_OUTER],
             output_buffer<out_t>&         out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(layernorm_i8_tiled::run);
    }

private:
    alignas(aie::vector_decl_align) int32_t sums_[2 * SROWS];
    alignas(aie::vector_decl_align) int16_t records_[SBANDS * REC];
};
