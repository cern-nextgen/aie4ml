// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Kernel-side helpers shared by the HCCS and accurate Softmax kernels; the graph does not include this header.

#pragma once

#include "softmax.h"

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


template <typename ConfigT, int N, int MTI, int MTO, typename T>
static inline aie::vector<T, N> load_microtile(const T* p)
{
    const aie::vector<T, N> v = *aie::cbegin_vector<N>(p);
    if constexpr (ConfigT::TRANSPOSE_INPUT) return aie::transpose(v, MTI, MTO);
    return v;
}

// P consecutive microtiles of BLK lanes as one vector, each transposed on its own when the input is.
template <typename ConfigT, int P, int BLK, int MTI, int MTO, typename T>
static inline aie::vector<T, P * BLK> load_microtiles(const T* p)
{
    if constexpr (!ConfigT::TRANSPOSE_INPUT || P == 1)
        return load_microtile<ConfigT, P * BLK, MTI, MTO>(p);
    else if constexpr (P == 2)
        return aie::concat(load_microtile<ConfigT, BLK, MTI, MTO>(p), load_microtile<ConfigT, BLK, MTI, MTO>(p + BLK));
    else
        return aie::concat(load_microtile<ConfigT, BLK, MTI, MTO>(p), load_microtile<ConfigT, BLK, MTI, MTO>(p + BLK),
                           load_microtile<ConfigT, BLK, MTI, MTO>(p + 2 * BLK),
                           load_microtile<ConfigT, BLK, MTI, MTO>(p + 3 * BLK));
}

// P copies of v side by side, and the lane-wise max / sum of v's P equal parts.
template <int P, typename T, unsigned N>
static inline aie::vector<T, P * N> repeat(const aie::vector<T, N>& v)
{
    if constexpr (P == 1) return v;
    else if constexpr (P == 2) return aie::concat(v, v);
    else return aie::concat(v, v, v, v);
}

template <int P, typename T, unsigned N>
static inline aie::vector<T, N / P> fold_max(const aie::vector<T, N>& v)
{
    if constexpr (P == 1) return v;
    else return fold_max<P / 2>(aie::max(v.template extract<N / 2>(0), v.template extract<N / 2>(1)));
}

template <int P, typename A, unsigned N>
static inline aie::accum<A, N / P> fold_add(const aie::accum<A, N>& v)
{
    if constexpr (P == 1) return v;
    else return fold_add<P / 2>(aie::add(v.template extract<N / 2>(0), v.template extract<N / 2>(1)));
}


// Each row's max over a band of microtiles, in every lane of its row: reduced into lane m*MT_INNER, then copied
// across the row by the matrix unit as A * (e_0 1^T).
template <typename ConfigT, int P>
static inline aie::vector<int8, ConfigT::MICROTILE_OUTER * ConfigT::MICROTILE_INNER> band_row_max(const int8* band)
{
    constexpr int MTO = ConfigT::MICROTILE_OUTER, MTI = ConfigT::MICROTILE_INNER, BLK = MTO * MTI;
    constexpr int VB = P * BLK, NV = ConfigT::COLS / MTI / P;
    const aie::vector<int8, MTI * MTI> first_col = aie::select(
        aie::zeros<int8, MTI * MTI>(), aie::broadcast<int8, MTI * MTI>(1), aie::mask<MTI * MTI>::from_uint64((1ull << MTI) - 1));
    aie::vector<int8, VB> vmax = load_microtiles<ConfigT, P, BLK, MTI, MTO>(band);
    for (int bv = 1; bv < NV; ++bv)
        vmax = aie::max(vmax, load_microtiles<ConfigT, P, BLK, MTI, MTO>(band + bv * VB));
    aie::vector<int8, BLK> bmax = fold_max<P>(vmax);
    for (int step = MTI / 2; step >= 1; step >>= 1)
        bmax = aie::max(bmax, aie::shuffle_down(bmax, step));
    aie::mmul<MTO, MTI, MTI, int8, int8> spread;
    spread.mul(bmax, first_col);
    return spread.template to_vector<int8>(0);
}
