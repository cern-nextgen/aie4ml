// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "softmax_kernels.h"

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


template <typename ConfigT>
softmax_i8_tiled<ConfigT>::softmax_i8_tiled(int16_t B_i, int8_t S_i, uint8_t DMAX_i)
    : softmax_base<ConfigT>(),
      B_param(B_i),
      S_param(S_i),
      DMAX_param(DMAX_i)
{}

// The CLB scale of a row is CLB_UNIT * code, from q = Z >> 8 in [1, 127] (256 <= Z <= 32767 by the HCCS constraints)
// with k = floor(log2 Z) = 39 - clb(q): 255 << (INV_SHIFT - k) for uint8 out, 32767 >> k = 127 >> (k - 8) for int16.
template <typename ConfigT>
inline uint32_t softmax_i8_tiled<ConfigT>::clb_code(int32_t q)
{
    const int lz = clb(q);
    return std::is_same_v<out_t, int16_t> ? (127u >> (31 - lz)) : (1u << (lz + ConfigT::INV_SHIFT - 39));
}

template <typename ConfigT>
inline int16_t softmax_i8_tiled<ConfigT>::exact_scale(int32_t z)
{
    return (int16_t)(base::NUMER / z);   // <= 32767: Z >= 256 by the HCCS constraints
}

// The lanes k*MT_INNER + k of an MT_INNER square: its diagonal.
template <typename ConfigT>
constexpr uint64_t softmax_i8_tiled<ConfigT>::diagonal_lanes()
{
    uint64_t lanes = 0;
    for (int k = 0; k < MT_INNER; ++k) lanes |= 1ull << (k * MT_INNER + k);
    return lanes;
}


#if __AIE_ARCH__ == 10
// AIE1: one row band at a time, in int16 lanes (its int8 arithmetic widens through 16-bit lanes, and spills, on every
// step; one unpack per block widens it once), with the band's scores kept for the normalising pass.
template <typename ConfigT>
void softmax_i8_tiled<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();
    int16_t* __restrict scores = (int16_t*)scratch_;

    const int32_t B    = B_param;
    const int8_t  S    = S_param;
    const int32_t DMAX = DMAX_param;
    const aie::vector<int16, BLK> s_v = aie::broadcast<int16, BLK>(S);
    const aie::vector<int16, BLK> nb_v = aie::broadcast<int16, BLK>((int16)NB);

    for (int bm = 0; bm < ROWS / MT_OUTER; ++bm) {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;

        // ---- pass 1: per-row max, segmented over the MT_INNER lane groups ----
        aie::vector<int16, BLK> vmax = aie::broadcast<int16, BLK>(-128);
        for (int bn = 0; bn < NB; ++bn)
            vmax = aie::max(vmax, aie::unpack(load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK)));
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
        const aie::vector<int16, BLK> lo_v = spread_rows<int16, MT_OUTER, MT_INNER>(lo16);
        const aie::vector<int16, BLK> c_v = spread_rows<int16, MT_OUTER, MT_INNER>(c16);
        const aie::accum<acc_t, BLK> c_acc(c_v);

        // ---- pass 2: scores to the scratch band, and their per-row sum Z = NB*c + S*sum(y) per lane ----
        aie::accum<acc_t, BLK> zsum = aie::mul(c_v, nb_v);
        for (int bn = 0; bn < NB; ++bn) {
            const aie::vector<int16, BLK> y =
                aie::max(aie::unpack(load_microtile<ConfigT, BLK, MT_INNER, MT_OUTER>(band + bn * BLK)), lo_v);
            *aie::begin_vector<BLK>(scores + bn * BLK) = aie::mac(c_acc, y, s_v).template to_vector<int16>(0);
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
            const aie::vector<int16, BLK> score16 = *aie::cbegin_vector<BLK>(scores + bn * BLK);
            *aie::begin_vector<BLK>(dst + bn * BLK) = aie::mul(score16, inv_v).template to_vector<out_t>(OUT_SHIFT);
        }
    }
}

#else
// AIE-ML, AIE-MLv2: three passes, each a loop over the row bands that the compiler pipelines across bands, handing
// each row's max and scale to the next pass through the scratch. P microtiles side by side fill one 64-byte vector;
// stacked they are a (P*MT_OUTER) x MT_INNER matrix whose rows are rows of the band, so the row reductions and
// broadcasts run on the matrix unit as products C = A * R with a constant MT_INNER square R, on the data as it lies.
template <typename ConfigT>
void softmax_i8_tiled<ConfigT>::run(input_buffer<in_t>& in, output_buffer<out_t>& out)
{
    using acc_t  = typename ConfigT::acc_scalar_t;
    using vecmm  = aie::mmul<P * MT_OUTER, MT_INNER, MT_INNER, int8, int8>;
    using codemm = aie::mmul<MT_OUTER, MT_INNER, MT_INNER, uint8, uint8>;
    constexpr int VB    = P * BLK;
    constexpr int RB    = MT_INNER * MT_INNER;
    constexpr int BANDS = ROWS / MT_OUTER;
    constexpr int NV    = NB / P;
    // uint8 out: scale = 255 * code; int16 out: scale = code (CLB); the exact scale is stored whole.
    constexpr uint8_t CLB_UNIT = std::is_same_v<out_t, int16_t> ? 1 : 255;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();
    int8_t*   __restrict row_max   = scratch_;                                     // BLK int8 per band
    uint32_t* __restrict row_scale = (uint32_t*)(scratch_ + ROWS * MT_INNER);      // BLK int16 (or uint8 code) per band

    const int32_t B    = B_param;
    const int8_t  S    = S_param;
    const uint8_t DMAX = DMAX_param;
    const aie::mask<RB> row0 = aie::mask<RB>::from_uint64((1ull << MT_INNER) - 1);
    // R = CLB_UNIT e_0 1^T copies a row's code across it; R = -S 1 1^T sums the row times -S; R = -S I scales each element.
    const aie::vector<uint8, RB> clb_col    = aie::select(aie::zeros<uint8, RB>(), aie::broadcast<uint8, RB>(CLB_UNIT), row0);
    const aie::vector<int8, RB>  neg_s_ones = aie::broadcast<int8, RB>(-S);
    const aie::vector<int8, RB>  neg_s_diag = aie::select(aie::zeros<int8, RB>(), aie::broadcast<int8, RB>(-S),
                                                          aie::mask<RB>::from_uint64(diagonal_lanes()));
    const aie::accum<acc_t, VB> b_acc(aie::broadcast<int32, VB>(B));
    const aie::accum<acc_t, VB> z0_acc(aie::broadcast<int32, VB>(P == 1 ? COLS * B : 0));

    // ---- pass 1: each row's max, in every lane of its row ----
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        *aie::begin_vector<BLK>(row_max + bm * BLK) = band_row_max<ConfigT, P>(in_ptr + bm * NB * BLK);
    }

    // ---- pass 2: d = min(max - x, DMAX) (exact in uint8), Z = COLS*B - S*sum(d) in every lane of its row, and the
    //      row's scale on the scalar unit ----
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t* __restrict band = in_ptr + bm * NB * BLK;
        const aie::vector<uint8, VB> max_u = repeat<P>(aie::load_v<BLK>(row_max + bm * BLK)).template cast_to<uint8>();
        vecmm z(z0_acc);
        for (int bv = 0; bv < NV; ++bv) {
            const aie::vector<uint8, VB> x =
                load_microtiles<ConfigT, P, BLK, MT_INNER, MT_OUTER>(band + bv * VB).template cast_to<uint8>();
            z.mac(aie::min(aie::sub(max_u, x), DMAX).template cast_to<int8>(), neg_s_ones);
        }
        aie::accum<acc_t, BLK> zb = fold_add<P>(z.to_accum());
        if constexpr (P > 1) zb = aie::add(zb, aie::broadcast<int32, BLK>(COLS * B));
        if constexpr (ConfigT::USE_CLB) {
            // The row's code at byte m*MT_INNER, as a word: byte stores issue far apart, and pass 3 reads only that byte.
            const aie::vector<uint8, BLK> q = zb.template to_vector<uint8>(8);
            for (int m = 0; m < MT_OUTER; ++m)
                row_scale[(bm * BLK + m * MT_INNER) / 4] = clb_code(q.get(m * MT_INNER));
        } else {
            // The exact scale across the row's lanes, two lanes a word.
            const aie::vector<int16, BLK> zv = zb.template to_vector<int16>(0);   // Z <= COLS*B <= 32767
            for (int m = 0; m < MT_OUTER; ++m) {
                const uint32_t inv = (uint16_t)exact_scale(zv.get(m * MT_INNER));
                for (int w = 0; w < MT_INNER / 2; ++w)
                    row_scale[(bm * BLK + m * MT_INNER) / 2 + w] = inv | (inv << 16);
            }
        }
    }

    // ---- pass 3: recompute the score B - S*d and normalise ----
    for (int bm = 0; bm < BANDS; ++bm)
        chess_prepare_for_pipelining chess_loop_range(1, )
    {
        const in_t*  __restrict band = in_ptr  + bm * NB * BLK;
              out_t* __restrict dst  = out_ptr + bm * NB * BLK;
        const aie::vector<uint8, VB> max_u = repeat<P>(aie::load_v<BLK>(row_max + bm * BLK)).template cast_to<uint8>();
        aie::vector<int16, BLK> scale;
        if constexpr (ConfigT::USE_CLB) {
            codemm spread;
            spread.mul(aie::load_v<BLK>((const uint8_t*)row_scale + bm * BLK), clb_col);
            scale = spread.template to_vector<int16>(0);
        } else {
            scale = aie::load_v<BLK>((const int16_t*)row_scale + bm * BLK);
        }
        const aie::vector<int16, VB> scale_v = repeat<P>(scale);
        for (int bv = 0; bv < NV; ++bv) {
            const aie::vector<uint8, VB> x =
                load_microtiles<ConfigT, P, BLK, MT_INNER, MT_OUTER>(band + bv * VB).template cast_to<uint8>();
            vecmm score(b_acc);
            score.mac(aie::min(aie::sub(max_u, x), DMAX).template cast_to<int8>(), neg_s_diag);
            *aie::begin_vector<VB>(dst + bv * VB) =
                aie::mul(score.template to_vector<int16>(0), scale_v).template to_vector<out_t>(OUT_SHIFT);
        }
    }
}
#endif
