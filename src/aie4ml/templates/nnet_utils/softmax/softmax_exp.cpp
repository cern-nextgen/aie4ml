// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "softmax_kernels.h"

using namespace adf;

// Accurate Softmax: exp of the real logit gaps, no calibration. The linear kernel and AIE1 use the integer exp below,
// much slower than HCCS; the tiled kernel on AIE-ML and AIE-MLv2 computes exp in bf16.

// e = 2^(-(max-x) * K_Q / 2^ZF), Q14, for N lanes. 2nd-order poly for the fractional octave; a
// 4-step binary conditional halving for the integer octave (AIE-ML has no per-lane variable
// shift). Its own function so the ~9 live vectors stay in a child frame, not run's.
template <typename ConfigT, int N>
static aie::vector<int16, N> exp_score16(const aie::vector<int8, N>& vx,
                                         const aie::vector<int16, N>& maxv)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    constexpr int ZF = ConfigT::EXP_ZF;
    const aie::vector<int16, N> vx16 = aie::from_vector<acc_t>(vx).template to_vector<int16>(0);
    const aie::vector<int16, N> d = aie::sub(maxv, vx16);                           // >=0, <=255
    const aie::vector<int16, N> z = aie::mul(d, aie::broadcast<int16, N>(ConfigT::EXP_KQ)).template to_vector<int16>(0);
    const aie::vector<int16, N> zint  = aie::downshift(z, ZF);
    const aie::vector<int16, N> zfrac = aie::bit_and(z, aie::broadcast<int16, N>((int16_t)((1 << ZF) - 1)));
    const aie::vector<int16, N> t     = aie::mul(zfrac, aie::broadcast<int16, N>(ConfigT::EXP_A2)).template to_vector<int16>(ZF);
    const aie::vector<int16, N> inner = aie::add(t, aie::broadcast<int16, N>(ConfigT::EXP_B1));
    const aie::vector<int16, N> t2    = aie::mul(inner, zfrac).template to_vector<int16>(ZF);
    aie::vector<int16, N> e           = aie::add(t2, aie::broadcast<int16, N>(ConfigT::EXP_C0));  // 2^(-zfrac)
    for (int b = 0; b < 4; ++b) {
        const aie::vector<int16, N> sh = aie::broadcast<int16, N>((int16_t)(1 << b));
        e = aie::select(e, aie::downshift(e, 1 << b), aie::ge(aie::bit_and(zint, sh), sh));
    }
    e = aie::select(e, aie::broadcast<int16, N>((int16_t)0),
                    aie::ge(zint, aie::broadcast<int16, N>((int16_t)16)));            // zint>=16 -> 0
    return e;
}


#if __AIE_ARCH__ != 10
#if __AIE_ARCH__ == 22
static constexpr float pow2(int e)
{
    float v = 1.0f;
    for (; e > 0; --e) v *= 2.0f;
    for (; e < 0; ++e) v *= 0.5f;
    return v;
}
#else
// 2^-(d * EXP_K) for every gap d in [0, 256) as bf16 bits, in the lut<4> layout: 8-value chunks twice, a copy in each
// of two banks, so four lookups issue a cycle.
struct exp2_gap_table {
    int16_t bits[512];
};
template <typename Kernel>
static constexpr exp2_gap_table make_exp2_gap_table()
{
    exp2_gap_table t{};
    for (int d = 0; d < 256; ++d) {
        // 2^-x = 2^-n v, n = floor(x), v = 2^-(x - n) in (1/2, 1] from exp's series: bf16 bits ((126 - n) << 7) +
        // round((2v - 1) 128), 0 once the exponent leaves the normal range.
        const double x = d * Kernel::EXP_K;
        const int n = (int)x;
        double term = 1.0, v = 1.0;
        for (int k = 1; k < 20; ++k) { term *= -(x - n) * 0.6931471805599453 / k; v += term; }
        const int16_t b = n > 125 ? 0 : (int16_t)(((126 - n) << 7) + (int)((2.0 * v - 1.0) * 128.0 + 0.5));
        t.bits[(d / 8) * 16 + d % 8] = b;
        t.bits[(d / 8) * 16 + 8 + d % 8] = b;
    }
    return t;
}
template <typename Kernel>
alignas(aie::vector_decl_align) chess_storage(DM_bankA) static const exp2_gap_table exp2_gap_ab = make_exp2_gap_table<Kernel>();
template <typename Kernel>
alignas(aie::vector_decl_align) chess_storage(DM_bankB) static const exp2_gap_table exp2_gap_cd = make_exp2_gap_table<Kernel>();
#endif

// 2^(C - d * EXP_K) in bf16 for gaps d = max - x.
template <typename Kernel, int C, unsigned N>
static inline aie::vector<bfloat16, N> exp2_gap(const aie::vector<uint8, N>& d)
{
#if __AIE_ARCH__ == 22
    constexpr int F = Kernel::K_FRAC;
    // Native exp2 of x = C - d K. x is built as an integer and read as a float through its bits (the API's
    // int-to-float conversion goes lane by lane): accumulated onto 1.5 * 2^(23 - F), whose float ulp is 2^-F.
    constexpr int32_t ONE_HALF_BITS = ((127 + 23 - F) << 23) | (1 << 22);
    constexpr float ONE_HALF = 1.5f * pow2(23 - F);
    const aie::vector<int16, N> d16 = aie::unpack(d).template cast_to<int16>();
    const aie::accum<acc32, N> bits =
        aie::mac(aie::accum<acc32, N>(aie::broadcast<int32, N>(ONE_HALF_BITS)), d16, aie::broadcast<int16, N>(-Kernel::K_Q));
    const aie::accum<accfloat, N> x =
        aie::sub(bits.template cast_to<accfloat>(), aie::accum<accfloat, N>(aie::broadcast<float, N>(ONE_HALF - C)));
    aie::vector<bfloat16, N> e;
    for (unsigned q = 0; q < N / 16; ++q)
        e.insert(q, aie::vector<bfloat16, 16>(::exp2(x.template extract<16>(q))));
    return e;
#else
    // 2^-(d K) from the table, times 2^C in the bf16 exponent.
    using lut_t = aie::lut<4, bfloat16, bfloat16>;
    aie::parallel_lookup<uint8, lut_t> table(lut_t(256, exp2_gap_ab<Kernel>.bits, exp2_gap_cd<Kernel>.bits));
    const aie::vector<bfloat16, N> e = table.fetch(d);
    if constexpr (C == 0) return e;
    else return aie::add(e.template cast_to<int16>(), aie::broadcast<int16, N>(C << 7)).template cast_to<bfloat16>();
#endif
}

// 1 / z in bf16: a bit-trick seed, then two Newton steps y += y (1 - z y) with z in two bf16 parts.
template <unsigned N>
static inline aie::vector<bfloat16, N> reciprocal(const aie::accum<accfloat, N>& z)
{
    const aie::vector<bfloat16, N> z_hi = z.template to_vector<bfloat16>();
    const aie::vector<bfloat16, N> z_lo = aie::sub(z, aie::accum<accfloat, N>(z_hi)).template to_vector<bfloat16>();
    aie::vector<bfloat16, N> y = aie::sub(aie::broadcast<int16, N>(0x7EF3), z_hi.template cast_to<int16>()).template cast_to<bfloat16>();
    for (int step = 0; step < 2; ++step) {
        const aie::vector<bfloat16, N> e =
            aie::msc(aie::msc(aie::accum<accfloat, N>(aie::broadcast<float, N>(1.0f)), z_hi, y), z_lo, y).template to_vector<bfloat16>();
        y = aie::mac(aie::accum<accfloat, N>(y), y, e).template to_vector<bfloat16>();
    }
    return y;
}
#endif


#if __AIE_ARCH__ == 10
// AIE1: the integer exp, recomputed in pass 3.
template <typename ConfigT>
void softmax_exp_i8_tiled<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();

    for (int bm = 0; bm < ROWS / MT_OUTER; ++bm) {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;

        aie::vector<int8, BLK> vmax = aie::broadcast<int8, BLK>(-128);
        for (int bn = 0; bn < NB; ++bn)
            vmax = aie::max(vmax, load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK));
        for (int step = MT_INNER / 2; step >= 1; step >>= 1)
            vmax = aie::max(vmax, aie::shuffle_down(vmax, step));
        int16_t row_max[STAT_LANES] = {};
        for (int m = 0; m < MT_OUTER; ++m) row_max[m] = (int16_t)vmax.get(m * MT_INNER);
        const aie::vector<int16, BLK> max16 = spread_rows<int16, MT_OUTER, MT_INNER>(row_max);

        aie::vector<int32, BLK> ssum = aie::zeros<int32, BLK>();
        for (int bn = 0; bn < NB; ++bn) {
            const aie::vector<int8, BLK> vx = load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK);
            const aie::vector<int16, BLK> e = exp_score16<ConfigT, BLK>(vx, max16);
            ssum = aie::add(ssum, aie::from_vector<acc_t>(e).template to_vector<int32>(0));
        }
        int32_t sum[STAT_LANES] = {};
        constexpr int SUB = (BLK < 32) ? BLK : 32;
        for (int sb = 0; sb < BLK / SUB; ++sb) {
            aie::vector<int32, SUB> sv = ssum.template extract<SUB>(sb);
            for (int step = MT_INNER / 2; step >= 1; step >>= 1)
                sv = aie::add(sv, aie::shuffle_down(sv, step));
            for (int g = 0; g < SUB / MT_INNER; ++g)
                sum[sb * (SUB / MT_INNER) + g] = sv.get(g * MT_INNER);
        }
        int32_t invq[STAT_LANES];
        base::template batched_reciprocal<STAT_LANES>(sum, invq);
        int16_t inv16[STAT_LANES] = {};
        for (int m = 0; m < MT_OUTER; ++m) {
            const int32_t v = invq[m];
            inv16[m] = (int16_t)(v > 32767 ? 32767 : (v < 0 ? 0 : v));
        }
        const aie::vector<int16, BLK> inv_v = spread_rows<int16, MT_OUTER, MT_INNER>(inv16);

        for (int bn = 0; bn < NB; ++bn) {
            const aie::vector<int8, BLK> vx = load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK);
            const aie::vector<int16, BLK> e = exp_score16<ConfigT, BLK>(vx, max16);
            aie::accum<acc_t, BLK> prod = aie::mul(e, inv_v);
            *aie::begin_vector<BLK>(dst + bn * BLK) = prod.template to_vector<out_t>(OUT_SHIFT);
        }
    }
}
#else
#if __AIE_ARCH__ == 20
// On AIE-ML Chess fails to pipeline a band loop around a rolled microtile loop ("Incorrect DSFG after transforming
// pipelined non leaf loop", e.g. three 4x8 microtiles a band); unrolled, the band loop is a leaf loop.
#define MICROTILE_LOOP chess_unroll_loop(*)
#else
#define MICROTILE_LOOP
#endif

// AIE-ML, AIE-MLv2: the tiled HCCS's three passes over row bands, with the score 2^(-d K) in bf16 (the conversion
// to integer codes goes through the float bits too), row sums by a bf16 matrix product with ones, and round to nearest.
template <typename ConfigT>
void softmax_exp_i8_tiled<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    constexpr int VB = P * BLK, BANDS = ROWS / MT_OUTER, NV = NB / P;
    constexpr int SUM_ROWS = P * MT_OUTER < 4 ? 4 : P * MT_OUTER;   // bf16 matrix products take 4 or 8 rows
    constexpr int EXP_LANES = VB < 32 ? VB : 32;                     // exp a half vector at a time: fewer live registers
    using sum_mm = aie::mmul<SUM_ROWS, 8, 8, bfloat16, bfloat16>;
    aie::set_rounding(aie::rounding_mode::conv_even);
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();
    int8_t*   __restrict row_max = scratch_;
    bfloat16* __restrict row_inv = (bfloat16*)(scratch_ + ROWS * MT_INNER);
    const aie::vector<bfloat16, 64> ones = aie::broadcast<bfloat16, 64>(1.0f);
    // round(p) of a float p < 2^22: p accumulated onto 1.5 * 2^23, whose ulp is 1, and the constant's bits taken off.
    const aie::accum<accfloat, EXP_LANES> round_base(aie::broadcast<float, EXP_LANES>(12582912.0f));
    const aie::accum<acc32, EXP_LANES> round_bits(aie::broadcast<int32, EXP_LANES>(0x4B400000));

    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        *aie::begin_vector<BLK>(row_max + bm * BLK) = band_row_max<ConfigT, P>(in_ptr + bm * NB * BLK);
    }

    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t* __restrict band = in_ptr + bm * NB * BLK;
        const aie::vector<uint8, VB> max_u = repeat<P>(aie::load_v<BLK>(row_max + bm * BLK)).template cast_to<uint8>();
        sum_mm z(aie::zeros<accfloat, SUM_ROWS * 8>());
        for (int bv = 0; bv < NV; ++bv) MICROTILE_LOOP {
            const aie::vector<uint8, VB> d =
                aie::sub(max_u, load_microtiles<ConfigT, P, BLK, MT_INNER, MT_OUTER>(band + bv * VB).template cast_to<uint8>());
            aie::vector<bfloat16, SUM_ROWS * 8> e = aie::zeros<bfloat16, SUM_ROWS * 8>();
            for (int h = 0; h < VB / EXP_LANES; ++h)
                e.insert(h, exp2_gap<softmax_exp_i8_tiled, 0>(d.template extract<EXP_LANES>(h)));
            z.mac(e, ones);
        }
        const aie::accum<accfloat, BLK> zb = fold_add<P>(z.to_accum().template extract<VB>(0));
        *aie::begin_vector<BLK>(row_inv + bm * BLK) = reciprocal(zb);
    }

    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;
        const aie::vector<uint8, VB> max_u = repeat<P>(aie::load_v<BLK>(row_max + bm * BLK)).template cast_to<uint8>();
        const aie::vector<bfloat16, VB> inv = repeat<P>(aie::load_v<BLK>(row_inv + bm * BLK));
        for (int bv = 0; bv < NV; ++bv) MICROTILE_LOOP {
            const aie::vector<uint8, VB> d =
                aie::sub(max_u, load_microtiles<ConfigT, P, BLK, MT_INNER, MT_OUTER>(band + bv * VB).template cast_to<uint8>());
            for (int h = 0; h < VB / EXP_LANES; ++h) {
                const aie::accum<accfloat, EXP_LANES> p = aie::mac(
                    round_base, exp2_gap<softmax_exp_i8_tiled, OUT_FRAC>(d.template extract<EXP_LANES>(h)), inv.template extract<EXP_LANES>(h));
                *aie::begin_vector<EXP_LANES>(dst + bv * VB + h * EXP_LANES) =
                    aie::sub(p.template cast_to<acc32>(), round_bits).template to_vector<out_t>(0);
            }
        }
    }
}
#endif


// Linear accurate Softmax: row-contiguous, batched reciprocal like the linear HCCS kernel.
template <typename ConfigT>
void softmax_exp_i8<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const int8* __restrict in_ptr  = (const int8*)in.data();
          out_t* __restrict out_ptr = (out_t*)out.data();
    constexpr int VEC  = ConfigT::VEC;
    constexpr int VECS = COLS / VEC;

    int16_t rmax[ROWS];
    int32_t rsum[ROWS];
    for (int row = 0; row < ROWS; ++row) {
        const int8* __restrict rp = in_ptr + row * COLS;
        auto it = aie::cbegin_vector<VEC>(rp);
        aie::vector<int8, VEC> vmax = *it++;
        for (int i = 1; i < VECS; ++i) vmax = aie::max(vmax, *it++);
        const int16_t max_v = (int16_t)aie::reduce_max(vmax);
        rmax[row] = max_v;
        const aie::vector<int16, VEC> mv = aie::broadcast<int16, VEC>(max_v);
        aie::vector<int32, VEC> vsum = aie::zeros<int32, VEC>();
        it = aie::cbegin_vector<VEC>(rp);
        for (int i = 0; i < VECS; ++i) {
            const aie::vector<int16, VEC> e = exp_score16<ConfigT, VEC>(*it++, mv);
            vsum = aie::add(vsum, aie::from_vector<acc_t>(e).template to_vector<int32>(0));
        }
        rsum[row] = aie::reduce_add(vsum);
    }

    int32_t rinv[ROWS];
    constexpr int RW = 16;
    for (int g = 0; g < ROWS; g += RW) {
        int32_t cs[RW], ci[RW];
        const int n = (ROWS - g < RW) ? (ROWS - g) : RW;
        for (int j = 0; j < n; ++j)  cs[j] = rsum[g + j];
        for (int j = n; j < RW; ++j) cs[j] = 1;
        base::template batched_reciprocal<RW>(cs, ci);
        for (int j = 0; j < n; ++j)  rinv[g + j] = ci[j];
    }

    for (int row = 0; row < ROWS; ++row) {
        const int8* __restrict rp = in_ptr + row * COLS;
              out_t* __restrict op = out_ptr + row * COLS;
        const aie::vector<int16, VEC> mv = aie::broadcast<int16, VEC>(rmax[row]);
        const aie::vector<int16, VEC> inv_vec = aie::broadcast<int16, VEC>((int16_t)rinv[row]);
        auto it = aie::cbegin_vector<VEC>(rp);
        auto ot = aie::begin_vector<VEC>(op);
        for (int i = 0; i < VECS; ++i) {
            const aie::vector<int16, VEC> e = exp_score16<ConfigT, VEC>(*it++, mv);
            aie::accum<acc_t, VEC> prod = aie::mul(e, inv_vec);
            *ot++ = prod.template to_vector<out_t>(OUT_SHIFT);
        }
    }
}
