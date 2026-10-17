// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm.h"

using namespace adf;


// Mean, variance and the Q15 reciprocal standard deviation for a batch of rows.
//
// AIE-ML/AIE-MLv2 has no scalar FPU: touching a float one element at a time pulls in the software
// float32 routines, which cost far more than they save and overflow the kernel's stack.
// Every float step here stays in the vector unit.
template <typename ConfigT>
template <unsigned N>
void layernorm_base<ConfigT>::row_statistics(const int32_t (&sum_x)[N],
                                             const int32_t (&sum_sq)[N],
                                             int16_t (&mu16)[N],
                                             int16_t (&inv_std16)[N])
{
    aie::vector<int32, N> var_i;
    for (unsigned r = 0; r < N; ++r) {
        const int32_t mu = (sum_x[r] + (1 << (LOG2_COLS - 1))) >> LOG2_COLS;
        // sum((x-mu)^2) = sum_sq - 2*mu*sum_x + N*mu^2. The naive sum_sq/N - mu^2 loses
        // precision catastrophically for shifted inputs.
        const int32_t centered_sq = sum_sq[r] - 2 * mu * sum_x[r] + COLS * mu * mu;
        int32_t var = centered_sq >> LOG2_COLS;
        if (var < 0) var = 0;
        mu16[r]  = (int16_t)mu;
        var_i[r] = var + (int32_t)ConfigT::EPS_Q0;
    }

    const aie::vector<float, N> inv_f = aie::invsqrt(aie::to_float(var_i, 0));
    const aie::vector<int32, N> inv_q = aie::to_fixed<int32>(inv_f, NORM_SHIFT);

    for (unsigned r = 0; r < N; ++r) {
        const int32_t v = inv_q.get(r);
        inv_std16[r] = (int16_t)(v > 32767 ? 32767 : (v < 0 ? 0 : v));
    }
}



template <typename ConfigT>
void layernorm_i8<ConfigT>::run(input_buffer<in_t>&    in,
                                const int16_t (&gamma)[COLS],
                                const int16_t (&beta)[COLS],
                                output_buffer<out_t>&  out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();

    // gamma and beta do not vary with the row, so they are loaded once rather than per row.
    aie::vector<int16, VEC> gamma_v[VECS];
    aie::vector<int16, VEC> beta_v[VECS];
    {
        auto vgamma_it = aie::cbegin_vector<VEC>((const int16_t*)gamma);
        auto vbeta_it  = aie::cbegin_vector<VEC>((const int16_t*)beta);
        for (int v = 0; v < VECS; ++v) {
            gamma_v[v] = *vgamma_it++;
            beta_v[v]  = *vbeta_it++;
        }
    }

    const aie::vector<int8, VEC> ones8 = aie::broadcast<int8, VEC>(1);

    for (int base_row = 0; base_row < ROWS; base_row += ROWS_PER_BATCH) {

        // Sized to the vector width, not the batch: any lane past the batch stays zero and its
        // statistics are computed but never stored.
        int32_t sum_x[STAT_LANES]  = {};
        int32_t sum_sq[STAT_LANES] = {};

        for (int r = 0; r < ROWS_PER_BATCH; ++r)
            chess_prepare_for_pipelining
            chess_loop_range(ROWS_PER_BATCH, ROWS_PER_BATCH)
        {
            auto vin_it = aie::cbegin_vector<VEC>(in_ptr + (base_row + r) * COLS);
            aie::accum<acc_t, VEC> acc_sum = aie::zeros<acc_t, VEC>();
            aie::accum<acc_t, VEC> acc_sq  = aie::zeros<acc_t, VEC>();
            for (int v = 0; v < VECS; ++v) {
                const aie::vector<int8, VEC> vx = *vin_it++;
                acc_sum = aie::mac(acc_sum, vx, ones8);
                acc_sq  = aie::mac_square(acc_sq, vx);
            }
            // A row spans the whole vector here, so the lanes collapse all the way down.
            sum_x[r]  = aie::reduce_add(acc_sum.template to_vector<int32>(0));
            sum_sq[r] = aie::reduce_add(acc_sq.template to_vector<int32>(0));
        }

        int16_t mu16[STAT_LANES];
        int16_t inv_std16[STAT_LANES];
        base::template row_statistics<STAT_LANES>(sum_x, sum_sq, mu16, inv_std16);

        for (int r = 0; r < ROWS_PER_BATCH; ++r)
            chess_prepare_for_pipelining
            chess_loop_range(ROWS_PER_BATCH, ROWS_PER_BATCH)
        {
            const aie::vector<int16, VEC> inv_std_vec = aie::broadcast<int16, VEC>(inv_std16[r]);
            const aie::vector<int16, VEC> mu_vec      = aie::broadcast<int16, VEC>(mu16[r]);

            auto vin_it  = aie::cbegin_vector<VEC>(in_ptr + (base_row + r) * COLS);
            auto vout_it = aie::begin_vector<VEC>(out_ptr + (base_row + r) * COLS);

            for (int v = 0; v < VECS; ++v) {
                const aie::vector<int8, VEC> vx = *vin_it++;

                const aie::vector<int16, VEC> vd16 =
                    aie::sub(aie::from_vector<acc_t>(vx), mu_vec).template to_vector<int16>(0);

                const aie::accum<acc_t, VEC> acc_fs = aie::mul(inv_std_vec, gamma_v[v]);
                const aie::vector<int16, VEC> fscale = acc_fs.template to_vector<int16>(GAMMA_SHIFT);

                aie::accum<acc_t, VEC> acc_out = aie::mul(vd16, fscale);
                acc_out = aie::add(acc_out, beta_v[v]);

                *vout_it++ = acc_out.template to_vector<out_t>(NORM_SHIFT - OUT_SHIFT);
            }
        }
    }
}


namespace layernorm_tiled {

// P consecutive microtiles as one vector, each transposed on its own when the input is.
template <typename ConfigT, int P, int BLK>
inline aie::vector<int8, P * BLK> load_microtiles(const int8* p)
{
    constexpr int MTI = ConfigT::MICROTILE_INNER, MTO = ConfigT::MICROTILE_OUTER;
    if constexpr (!ConfigT::TRANSPOSE_INPUT)
        return aie::load_v<P * BLK>(p);
    else if constexpr (P == 1)
        return aie::transpose(aie::load_v<BLK>(p), MTI, MTO);
    else if constexpr (P == 2)
        return aie::concat(aie::transpose(aie::load_v<BLK>(p), MTI, MTO),
                           aie::transpose(aie::load_v<BLK>(p + BLK), MTI, MTO));
    else
        return aie::concat(aie::transpose(aie::load_v<BLK>(p), MTI, MTO),
                           aie::transpose(aie::load_v<BLK>(p + BLK), MTI, MTO),
                           aie::transpose(aie::load_v<BLK>(p + 2 * BLK), MTI, MTO),
                           aie::transpose(aie::load_v<BLK>(p + 3 * BLK), MTI, MTO));
}

// AIE1 multiplies int8 in int16 lanes.
template <unsigned N>
inline auto widen(const aie::vector<int8, N>& x)
{
#if __AIE_ARCH__ == 10
    return aie::unpack(x);
#else
    return x;
#endif
}

template <int P, typename T, unsigned N>
inline aie::vector<T, P * N> repeat(const aie::vector<T, N>& v)
{
    if constexpr (P == 1) return v;
    else if constexpr (P == 2) return aie::concat(v, v);
    else return aie::concat(v, v, v, v);
}

// The int32 lane sums of v's P equal parts. AIE1 adds accumulators only through vectors.
template <int P, typename A, unsigned N>
inline aie::vector<int32, N / P> fold_sum(const aie::accum<A, N>& v)
{
    if constexpr (P == 1)
        return v.template to_vector<int32>(0);
#if __AIE_ARCH__ == 10
    else
        return aie::add(fold_sum<P / 2>(v.template extract<N / 2>(0)), fold_sum<P / 2>(v.template extract<N / 2>(1)));
#else
    else
        return fold_sum<P / 2>(aie::add(v.template extract<N / 2>(0), v.template extract<N / 2>(1)));
#endif
}

// Lane i is v[i / W].
template <int W, typename T, unsigned N>
inline aie::vector<T, N * W> repeat_lanes(const aie::vector<T, N>& v)
{
    if constexpr (W == 1) return v;
    else {
        const auto [lo, hi] = aie::interleave_zip(v, v, 1);
        return repeat_lanes<W / 2>(aie::concat(lo, hi));
    }
}

// Lanes [PART * BLK, (PART + 1) * BLK) of repeat_lanes<W>(v), halved before any vector would pass 64 lanes.
template <int W, int BLK, int PART, typename T>
inline aie::vector<T, BLK> spread(const aie::vector<T, 8>& v)
{
    if constexpr (8 * W <= 64) {
        return repeat_lanes<W>(v).template extract<BLK>(PART);
    } else {
        static_assert((4 * W) % BLK == 0, "the lanes lie within one half");
        const auto [lo, hi] = aie::interleave_zip(v, v, 1);
        if constexpr (PART * BLK < 4 * W) return spread<W / 2, BLK, PART>(lo);
        else return spread<W / 2, BLK, PART - 4 * W / BLK>(hi);
    }
}

// The sum of each run of 2^W lanes.
template <int W, unsigned N>
inline aie::vector<int32, (N >> W)> pair_sums(const aie::vector<int32, N>& v)
{
    if constexpr (W == 0) return v;
    else return pair_sums<W - 1>(aie::add(aie::filter_even(v, 1), aie::filter_odd(v, 1)));
}

// v / 2^S rounded half up, S of either sign.
template <int S, typename T>
inline T round_shift(T v)
{
    if constexpr (S <= 0) return v << -S;
    else return (v + ((T)1 << (S - 1))) >> S;
}

// m, p <- m 2^S, p 2^(S/2) in the lanes where m < 2^(30-S).
template <int S, unsigned N>
inline void normalize_step(aie::vector<int32, N>& m, aie::vector<int16, N>& p)
{
    const auto low = aie::lt(m, int32(1 << (30 - S)));
    m = aie::select(m, aie::upshift(m, S), low);
    p = aie::select(p, aie::upshift(p, S / 2), low);
}

// c[2 hi + q] in each lane.
template <unsigned N, typename Mask>
inline aie::vector<int16, N> pick(const int16_t (&c)[4], const Mask& hi, const Mask& q)
{
    return aie::select(aie::select(aie::broadcast<int16, N>(c[0]), aie::broadcast<int16, N>(c[1]), q),
                       aie::select(aie::broadcast<int16, N>(c[2]), aie::broadcast<int16, N>(c[3]), q), hi);
}

// round(2^20 / sqrt(w)) for 2^10 <= w < 2^30, saturated to int16, in 16-bit lanes (AIE-ML has no float unit and
// emulates 32-bit products): w = 2^-2k m with m in [2^28, 2^30), so 2^20 / sqrt(w) = 2^k y / 2^8 with
// y = 2^14 / sqrt(v), v = m / 2^28 in [1, 4), a quadratic in u = v - a on four pieces [a, b) of geometric width.
template <unsigned N>
inline __attribute__((always_inline)) aie::vector<int16, N> inv_sqrt_q20(const aie::vector<int32, N>& w)
{
    using acc_t = aie::accum<acc32, N>;
    static constexpr int16_t A[4]  = {8192, 11585, 16384, 23170};   // the pieces' starts, Q13
    static constexpr int16_t C2[4] = {7840, 3297, 1388, 581};       // Q15
    static constexpr int16_t C1[4] = {-15772, -9381, -5580, -3314}; // Q15
    static constexpr int16_t C0[4] = {16377, 13772, 11581, 9738};   // Q14
    aie::vector<int32, N> m = w;
    aie::vector<int16, N> p = aie::broadcast<int16, N>(1);
    normalize_step<16>(m, p);
    normalize_step<8>(m, p);
    normalize_step<4>(m, p);
    normalize_step<2>(m, p);
    const aie::vector<int16, N> v = acc_t(aie::downshift(m, 15)).template to_vector<int16>(0);   // Q13
    const auto hi = aie::ge(v, A[2]);
    const auto q  = aie::ge(v, aie::select(aie::broadcast<int16, N>(A[1]), aie::broadcast<int16, N>(A[3]), hi));
    const aie::vector<int16, N> u = aie::sub(v, pick<N>(A, hi, q));
    const aie::vector<int16, N> t = aie::mac(acc_t(pick<N>(C1, hi, q), 13), pick<N>(C2, hi, q), u).template to_vector<int16>(13);
    const aie::vector<int16, N> y = aie::mac(acc_t(pick<N>(C0, hi, q), 14), t, u).template to_vector<int16>(14);
    return aie::mul(y, p).template to_vector<int16>(8);
}

}  // namespace layernorm_tiled


template <typename ConfigT>
void layernorm_i8_tiled<ConfigT>::run(input_buffer<in_t>&    in,
                                        const int16_t (&gamma)[COLS * MT_OUTER],
                                        const int16_t (&beta)[COLS * MT_OUTER],
                                        output_buffer<out_t>&  out)
{
    using namespace layernorm_tiled;
    using acc_t  = typename ConfigT::acc_scalar_t;
    using lane_t = typename decltype(widen(aie::vector<int8, VB>()))::value_type;
    const in_t*    __restrict in_ptr  = in.data();
          out_t*   __restrict out_ptr = out.data();
          int32_t* __restrict sums    = sums_;
          int16_t* __restrict records = records_;

    // Pass 1: each row's sum and sum of squares.
    const aie::vector<lane_t, VB> ones = aie::broadcast<lane_t, VB>(1);
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t* __restrict band = in_ptr + bm * NB * BLK;
        aie::accum<acc_t, VB> s1 = aie::zeros<acc_t, VB>(), s2 = aie::zeros<acc_t, VB>();
        for (int bv = 0; bv < NV; ++bv) {
            const aie::vector<lane_t, VB> x = widen(load_microtiles<ConfigT, P, BLK>(band + bv * VB));
            s1 = aie::mac(s1, x, ones);
            s2 = aie::mac_square(s2, x);
        }
        int32_t* __restrict totals = sums + bm * 2 * MT_OUTER;
        if constexpr (ROW_TOTALS) {
            static_assert(P == 1, "one microtile a vector");
            for (int m = 0; m < MT_OUTER; ++m) {
                totals[2 * m]     = aie::reduce_add(s1.template extract<MT_INNER>(m).template to_vector<int32>(0));
                totals[2 * m + 1] = aie::reduce_add(s2.template extract<MT_INNER>(m).template to_vector<int32>(0));
            }
        } else {
            const aie::vector<int32, BLK> a = fold_sum<P>(s1), b = fold_sum<P>(s2);
            for (int p = 0; p < BLK / SUB; ++p)
                aie::store_v(totals + p * 2 * PIECE_ROWS,
                             pair_sums<LOG2_MT_INNER - 1>(aie::concat(pair_sums<1>(a.template extract<SUB>(p)),
                                                                      pair_sums<1>(b.template extract<SUB>(p)))));
        }
    }

    // Statistics, one record [-mu | scale] per band: var = (n s2 - s1^2) / n^2, scale = 2^15 / sqrt(var + eps).
#if __AIE_ARCH__ == 10
    // AIE1 has a scalar float inverse square root; a band's rows, unrolled, overlap their chains.
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        int16_t* __restrict rec = records + bm * REC;
        for (int m = 0; m < MT_OUTER; ++m) chess_unroll_loop(*) {
            const int32_t s1 = sums[2 * (bm * MT_OUTER + m)], s2 = sums[2 * (bm * MT_OUTER + m) + 1];
            const int32_t mu = round_shift<LOG2_COLS - MEAN_FRAC>(s1);
            using nvar_t = std::conditional_t<COLS <= 256, int32_t, int64_t>;   // n s2, s1^2 < 2^31 up to 256 features
            const int32_t var = (int32_t)round_shift<2 * LOG2_COLS - VAR_FRAC>((nvar_t)s2 * COLS - (nvar_t)s1 * s1);
            const float w = (float)(var + (ConfigT::EPS_Q0 << VAR_FRAC));
            const int32_t y = (aie::to_fixed<int32>(aie::invsqrt(w), 21) + 1) >> 1;
            rec[m]            = (int16_t)-mu;
            rec[MT_OUTER + m] = (int16_t)(y > 32767 ? 32767 : y);
        }
    }
#else
    for (int g = 0; g < SROWS / NS; ++g)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const aie::vector<int32, 2 * NS> v = aie::load_v<2 * NS>(sums + g * 2 * NS);
        const aie::vector<int32, NS> s1 = aie::filter_even(v, PIECE_ROWS), s2 = aie::filter_odd(v, PIECE_ROWS);
        // A sum of up to 256 int8 fits int16, so s1^2 is a 16-bit product.
        aie::vector<int16, NS> neg_mu;
        if constexpr (LOG2_COLS >= MEAN_FRAC)
            neg_mu = aie::accum<acc32, NS>(aie::neg(s1)).template to_vector<int16>(LOG2_COLS - MEAN_FRAC);
        else
            neg_mu = aie::accum<acc32, NS>(aie::upshift(aie::neg(s1), MEAN_FRAC - LOG2_COLS)).template to_vector<int16>(0);
        aie::accum<acc64, NS> nvar = aie::mul(s2, aie::broadcast<int16, NS>(COLS));
        if constexpr (COLS <= 256) {
            const aie::vector<int16, NS> s1h = aie::accum<acc32, NS>(s1).template to_vector<int16>(0);
            nvar = aie::msc(nvar, s1h, s1h);
        } else {
            nvar = aie::sub(nvar, aie::mul(s1, s1));
        }
        const aie::vector<int32, NS> var = nvar.template to_vector<int32>(2 * LOG2_COLS - VAR_FRAC);
        const aie::vector<int16, NS> scale = inv_sqrt_q20(aie::add(var, ConfigT::EPS_Q0 << VAR_FRAC));
        const auto [lo, hi] = aie::interleave_zip(neg_mu, scale, MT_OUTER);
        int16_t* __restrict rec = records + g * (NS / MT_OUTER) * REC;
        if constexpr (REC == 2 * MT_OUTER) {
            aie::store_v(rec, lo);
            aie::store_v(rec + NS, hi);
        } else {   // 2-row bands: records padded to 8 lanes
            const auto [a, b] = aie::interleave_zip(lo, aie::zeros<int16, NS>(), 4);
            const auto [c, d] = aie::interleave_zip(hi, aie::zeros<int16, NS>(), 4);
            aie::store_v(rec, a);
            aie::store_v(rec + NS, b);
            aie::store_v(rec + 2 * NS, c);
            aie::store_v(rec + 3 * NS, d);
        }
    }
#endif

    // Pass 2: y = (x - mu) * (gamma / sigma) + beta.
#if __AIE_ARCH__ != 10
    const aie::vector<lane_t, VB> unit = aie::broadcast<lane_t, VB>(1 << MEAN_FRAC);
#endif
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;
        const int16_t* __restrict rec = records + bm * REC;
        // Row m's -mu and scale in lanes m * MT_INNER + k.
        aie::vector<int16, BLK> neg_mu, scale;
        if constexpr (REC == 8) {
            const aie::vector<int16, 8> r = aie::load_v<8>(rec);
            neg_mu = spread<MT_INNER, BLK, 0>(r);
            scale  = spread<MT_INNER, BLK, 1>(r);
        } else {
            neg_mu = spread<MT_INNER, BLK, 0>(aie::load_v<8>(rec));
            scale  = spread<MT_INNER, BLK, 0>(aie::load_v<8>(rec + 8));
        }
#if __AIE_ARCH__ == 10
        const aie::vector<int16, VB> mu_v = repeat<P>(neg_mu);
#else
        const aie::accum<acc_t, VB> mu_acc(repeat<P>(neg_mu));
#endif
        const aie::vector<int16, VB> scale_v = repeat<P>(scale);
        for (int bv = 0; bv < NV; ++bv) {
            const aie::vector<lane_t, VB> x = widen(load_microtiles<ConfigT, P, BLK>(band + bv * VB));
#if __AIE_ARCH__ == 10
            // AIE1 has four accumulators: centring in vector lanes leaves two for the next block.
            const aie::vector<int16, VB> d = aie::add(aie::upshift(x, MEAN_FRAC), mu_v);
#else
            const aie::vector<int16, VB> d = aie::mac(mu_acc, x, unit).template to_vector<int16>(0);
#endif
            const aie::vector<int16, VB> f =
                aie::mul(scale_v, aie::load_v<VB>(gamma + bv * VB)).template to_vector<int16>(F_SHIFT);
            const aie::accum<acc_t, VB> b(aie::load_v<VB>(beta + bv * VB), B_SHIFT);
            aie::store_v(dst + bv * VB, aie::mac(b, d, f).template to_vector<out_t>(O_SHIFT));
        }
    }
}
