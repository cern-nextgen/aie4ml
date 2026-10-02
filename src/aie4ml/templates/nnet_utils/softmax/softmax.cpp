// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "softmax.h"

using namespace adf;

template <typename ConfigT>
softmax_i8<ConfigT>::softmax_i8(
    int16_t B_i,
    int8_t S_i,
    uint8_t DMAX_i)
    : softmax_base<ConfigT>(),
      B_param(B_i),
      S_param(S_i),
      DMAX_param(DMAX_i)
{}

template <typename ConfigT>
inline __attribute__((always_inline)) void softmax_i8<ConfigT>::softmax_row(
    const int8* __restrict in_ptr,
    out_t* __restrict out_ptr,
    int16_t B,
    int8_t S,
    uint8_t DMAX) {
    using acc_t = typename ConfigT::acc_scalar_t;
    constexpr int VEC = ConfigT::VEC;
    constexpr int VECS = COLS / VEC;

    auto vin_it = aie::cbegin_vector<VEC>(in_ptr);
    aie::vector<int8, VEC> vmax = *vin_it++;
    for (int i = 1; i < VECS; ++i) {
        vmax = aie::max(vmax, *vin_it++);
    }
    int8 max_val = aie::reduce_max(vmax);

    vin_it = aie::cbegin_vector<VEC>(in_ptr);

    auto scratch_it = aie::begin_vector<VEC>(scratch);
#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 10
    // AIE1 has no acc32; Vitis schedules this widened acc48 path faster than int8/acc48,
    // and its unsupported accumulator-to-accumulator add requires the int32 vector sum.
    aie::vector<int32, VEC> score_sum = aie::zeros<int32, VEC>();
    const aie::vector<int16, VEC> max16 = aie::broadcast<int16, VEC>((int16_t)max_val);
    const aie::vector<int16, VEC> dmax16 = aie::broadcast<int16, VEC>((int16_t)DMAX);
#else
    aie::accum<acc_t, VEC> acc_sum = aie::zeros<acc_t, VEC>();
    uint8 max_u = (uint8)max_val;
    aie::vector<uint8, VEC> max_u_vec = aie::broadcast<uint8, VEC>(max_u);
#endif

    for (int i = 0; i < VECS; ++i) {
        aie::vector<int8, VEC> x = *vin_it++;
        aie::accum<acc_t, VEC> acc;
#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 10
        const aie::vector<int16, VEC> x16 = aie::from_vector<acc_t>(x).template to_vector<int16>(0);
        const aie::vector<int16, VEC> d = aie::min(aie::sub(max16, x16), dmax16);
        acc.from_vector(aie::broadcast<int32, VEC>(B));
        acc = aie::mac(acc, d, aie::broadcast<int16, VEC>((int16_t)(-S)));
        score_sum = aie::add(score_sum, acc.template to_vector<int32>(0));
#else
        aie::vector<uint8, VEC> xu = x.template cast_to<uint8>();
        aie::vector<uint8, VEC> d = aie::sub(max_u_vec, xu);
        d = aie::min(d, (uint8)DMAX);
        aie::vector<int8, VEC> d8 = d.template cast_to<int8>();
        acc.from_vector(aie::broadcast<int32, VEC>(B));
        acc = aie::mac(acc, d8, aie::broadcast<int8, VEC>(-S));
        acc_sum = aie::add(acc_sum, acc);
#endif
        aie::vector<int16, VEC> score16 = acc.template to_vector<int16>(0);
        *scratch_it++ = score16;
    }

#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 10
    int32_t sum = aie::reduce_add(score_sum);
#else
    int32_t sum = aie::reduce_add(acc_sum.template to_vector<int32_t>());
#endif

    int32_t inv_q0;
    if constexpr (ConfigT::USE_CLB) {
        int leading_zeros = clb(sum);
        int k = 31 - leading_zeros;
        if constexpr (std::is_same_v<out_t, int16_t>) {
            inv_q0 = 32767 >> k;
        } else {
            inv_q0 = 255 << (INV_SHIFT - k);
        }
    } else {
        if constexpr (std::is_same_v<out_t, int16_t>) {
            inv_q0 = 32767 / sum;
        } else {
            inv_q0 = (255 << INV_SHIFT) / sum;
        }
    }

    aie::vector<int16, VEC> inv_vec = aie::broadcast<int16, VEC>(inv_q0);

    auto scratch_rd = aie::cbegin_vector<VEC>(scratch);
    auto out_it = aie::begin_vector<VEC>(out_ptr);
    for (int i = 0; i < VECS; ++i) {
        aie::vector<int16, VEC> v = *scratch_rd++;
        aie::accum<acc_t, VEC> prod = aie::mul(v, inv_vec);
        *out_it++ = prod.template to_vector<out_t>(OUT_SHIFT);
    }
}

template <typename ConfigT>
void softmax_i8<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out) {
    auto in_ptr = (const in_t*)in.data();
    auto out_ptr = (out_t*)out.data();

    const int16_t B = B_param;
    const int8_t S = S_param;
    const uint8_t DMAX = DMAX_param;
    for (int row = 0; row < ROWS; ++row) {
        softmax_row(in_ptr + row * COLS, out_ptr + row * COLS, B, S, DMAX);
    }
}


// Exact floor(NUMER/sum[r]) for a batch of rows. Seed with one vectorised aie::inv (float) at a
// high fixed-point shift, then correct to the exact integer quotient with a few integer steps.
// The while-loops absorb aie::inv's approximation error, so the result is bit-identical to the
// linear kernel's scalar (255<<INV_SHIFT)/sum -- no float ever reaches the output.
template <typename ConfigT>
template <unsigned N>
void softmax_base<ConfigT>::batched_reciprocal(const int32_t (&sum)[N], int32_t (&invq)[N])
{
    constexpr int SHIFT = 24;
    aie::vector<int32, N> sv;
    for (unsigned r = 0; r < N; ++r) sv[r] = sum[r] > 0 ? sum[r] : 1;

    const aie::vector<float, N> rf   = aie::inv(aie::to_float(sv, 0));     // ~1/sum
    const aie::vector<int32, N> recip = aie::to_fixed<int32>(rf, SHIFT);   // ~round(2^SHIFT/sum)

    for (unsigned r = 0; r < N; ++r) {
        const int64_t s = sum[r] > 0 ? sum[r] : 1;
        int64_t q = ((int64_t)NUMER * (int64_t)recip.get(r)) >> SHIFT;
        if (q < 0) q = 0;
        while (q * s > (int64_t)NUMER) --q;
        while ((q + 1) * s <= (int64_t)NUMER) ++q;
        invq[r] = (int32_t)q;
    }
}


// CLB is a single scalar instruction and has no vector API. Spell out the fixed-size row band so
// AIE1 sees independent operations; a loop here can create an unschedulable dependency hazard.
template <typename ConfigT>
template <unsigned N>
#if defined(__AIE_ARCH__) && __AIE_ARCH__ == 10
__attribute__((noinline))
#endif
void softmax_base<ConfigT>::batched_clb_reciprocal(const int32_t* sum, int32_t* invq)
{
    static_assert(N == 2 || N == 4 || N == 8);
    if constexpr (std::is_same_v<out_t, int16_t>) {
        invq[0] = 32767 >> (31 - clb(sum[0]));
        invq[1] = 32767 >> (31 - clb(sum[1]));
        if constexpr (N >= 4) {
            invq[2] = 32767 >> (31 - clb(sum[2]));
            invq[3] = 32767 >> (31 - clb(sum[3]));
        }
        if constexpr (N == 8) {
            invq[4] = 32767 >> (31 - clb(sum[4]));
            invq[5] = 32767 >> (31 - clb(sum[5]));
            invq[6] = 32767 >> (31 - clb(sum[6]));
            invq[7] = 32767 >> (31 - clb(sum[7]));
        }
    } else {
        invq[0] = 255 << (INV_SHIFT - (31 - clb(sum[0])));
        invq[1] = 255 << (INV_SHIFT - (31 - clb(sum[1])));
        if constexpr (N >= 4) {
            invq[2] = 255 << (INV_SHIFT - (31 - clb(sum[2])));
            invq[3] = 255 << (INV_SHIFT - (31 - clb(sum[3])));
        }
        if constexpr (N == 8) {
            invq[4] = 255 << (INV_SHIFT - (31 - clb(sum[4])));
            invq[5] = 255 << (INV_SHIFT - (31 - clb(sum[5])));
            invq[6] = 255 << (INV_SHIFT - (31 - clb(sum[6])));
            invq[7] = 255 << (INV_SHIFT - (31 - clb(sum[7])));
        }
    }
}


// Fill a BLK-lane vector where lane group m is broadcast(s[m]): the per-row statistic spread to
// its lane group. Concat of broadcasts, one shot. int16 constituents keep every piece a legal
// vector width even for MT_INNER=8.
template <typename T, unsigned R, unsigned W>
static inline aie::vector<T, R * W> spread_rows(const T* s)
{
    if constexpr (R == 1)
        return aie::broadcast<T, W>(s[0]);
    else if constexpr (R == 2)
        return aie::concat(aie::broadcast<T, W>(s[0]), aie::broadcast<T, W>(s[1]));
    else if constexpr (R == 4)
        return aie::concat(aie::broadcast<T, W>(s[0]), aie::broadcast<T, W>(s[1]),
                           aie::broadcast<T, W>(s[2]), aie::broadcast<T, W>(s[3]));
    else
        return aie::concat(aie::concat(aie::broadcast<T, W>(s[0]), aie::broadcast<T, W>(s[1]),
                                       aie::broadcast<T, W>(s[2]), aie::broadcast<T, W>(s[3])),
                           aie::concat(aie::broadcast<T, W>(s[4]), aie::broadcast<T, W>(s[5]),
                                       aie::broadcast<T, W>(s[6]), aie::broadcast<T, W>(s[7])));
}


template <typename ConfigT>
softmax_i8_tiled<ConfigT>::softmax_i8_tiled(int16_t B_i, int8_t S_i, uint8_t DMAX_i)
    : softmax_base<ConfigT>(),
      B_param(B_i),
      S_param(S_i),
      DMAX_param(DMAX_i)
{}

template <typename ConfigT, int N, int MTI, int MTO, typename T>
static inline aie::vector<T, N> load_microtile(const T* p)
{
    const aie::vector<T, N> v = *aie::cbegin_vector<N>(p);
    if constexpr (ConfigT::TRANSPOSE_INPUT) return aie::transpose(v, MTI, MTO);
    return v;
}


template <typename ConfigT>
void softmax_i8_tiled<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();

    const int32_t B    = B_param;
    const int8_t  S    = S_param;
    const int32_t DMAX = DMAX_param;
    const aie::vector<int8, BLK> s_v = aie::broadcast<int8, BLK>(S);
    const aie::vector<int16, BLK> nb_v = aie::broadcast<int16, BLK>((int16)NB);

    for (int bm = 0; bm < ROWS / MT_OUTER; ++bm) {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;

        // ---- pass 1: per-row max, segmented over the MT_INNER lane groups ----
        aie::vector<int8, BLK> vmax = aie::broadcast<int8, BLK>(-128);
        for (int bn = 0; bn < NB; ++bn)
            vmax = aie::max(vmax, load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK));
        for (int step = MT_INNER / 2; step >= 1; step >>= 1)
            vmax = aie::max(vmax, aie::shuffle_down(vmax, step));

        // min(max-x, DMAX) = max - max(x, lo), lo = max-DMAX clamped to int8 (below -128 no x reaches it),
        // so score = B - S*(max - y) = c + S*y with y = max(x, lo) and c = B - S*max.
        int16_t lo16[STAT_LANES] = {}, c16[STAT_LANES] = {};
        for (int m = 0; m < MT_OUTER; ++m) {
            const int32_t row_max = vmax.get(m * MT_INNER);
            lo16[m] = (int16_t)(row_max - DMAX < -128 ? -128 : row_max - DMAX);
            c16[m] = (int16_t)(B - S * row_max);
        }
        const aie::vector<int8, BLK> lo_v =
            aie::accum<acc_t, BLK>(spread_rows<int16, MT_OUTER, MT_INNER>(lo16)).template to_vector<int8>(0);
        const aie::vector<int16, BLK> c_v = spread_rows<int16, MT_OUTER, MT_INNER>(c16);
        const aie::accum<acc_t, BLK> c_acc(c_v);

        // ---- pass 2: scores to the scratch band, and their per-row sum Z = NB*c + S*sum(y) per lane ----
        aie::accum<acc_t, BLK> zsum = aie::mul(c_v, nb_v);
        for (int bn = 0; bn < NB; ++bn) {
            const aie::vector<int8, BLK> y =
                aie::max(load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK), lo_v);
            *aie::begin_vector<BLK>(scores_ + bn * BLK) = aie::mac(c_acc, y, s_v).template to_vector<int16>(0);
            zsum = aie::mac(zsum, y, s_v);
        }
        aie::vector<int32, BLK> zv = zsum.template to_vector<int32>(0);
        int32_t sum[STAT_LANES] = {};
        constexpr int SUB = (BLK < 32) ? BLK : 32;
        for (int sb = 0; sb < BLK / SUB; ++sb) {
            aie::vector<int32, SUB> sv = zv.template extract<SUB>(sb);
            for (int step = MT_INNER / 2; step >= 1; step >>= 1)
                sv = aie::add(sv, aie::shuffle_down(sv, step));
            for (int g = 0; g < SUB / MT_INNER; ++g)
                sum[sb * (SUB / MT_INNER) + g] = sv.get(g * MT_INNER);
        }

        // ---- one reciprocal batch for the whole row band ----
        int32_t invq[STAT_LANES] = {};
        if constexpr (ConfigT::USE_CLB) {
            base::template batched_clb_reciprocal<MT_OUTER>(sum, invq);
        } else {
            base::template batched_reciprocal<STAT_LANES>(sum, invq);
        }
        int16_t inv16[STAT_LANES] = {};
        for (int m = 0; m < MT_OUTER; ++m) inv16[m] = (int16_t)invq[m];   // <= 32767: Z >= 256 by the HCCS constraints
        const aie::vector<int16, BLK> inv_v = spread_rows<int16, MT_OUTER, MT_INNER>(inv16);

        // ---- pass 3: normalise the stored scores ----
        for (int bn = 0; bn < NB; ++bn) {
            const aie::vector<int16, BLK> score16 = *aie::cbegin_vector<BLK>(scores_ + bn * BLK);
            *aie::begin_vector<BLK>(dst + bn * BLK) = aie::mul(score16, inv_v).template to_vector<out_t>(OUT_SHIFT);
        }
    }
}


// --------------------------------------------------------------------------- //
// Accurate integer exp Softmax (real 2nd-order integer exp, no calibration)
//
// TODO(perf): these kernels are much slower than the HCCS surrogate (the poly plus the
// per-lane 2^(-z_int) conditional shift). They are accurate and correct but slow -- optimise.
// --------------------------------------------------------------------------- //

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


// Tiled accurate Softmax: same microtile reduce/reciprocal/normalise as the HCCS tiled kernel,
// only the score differs. Recompute in pass 3 (not a stored scratch): the kernel overlaps the
// recompute with the row loads, and storing only adds memory traffic to a stall-bound kernel.
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
