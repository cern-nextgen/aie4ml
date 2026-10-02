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


// Fill a BLK-lane vector where lane group m is broadcast(s[m]): the per-row statistics spread
// to their lane groups. Concat of broadcasts, one shot instead of MT_OUTER inserts.
template <unsigned R, unsigned W>
static inline aie::vector<int16, R * W> spread_rows(const int16_t* s)
{
    if constexpr (R == 1)
        return aie::broadcast<int16, W>(s[0]);
    else if constexpr (R == 2)
        return aie::concat(aie::broadcast<int16, W>(s[0]), aie::broadcast<int16, W>(s[1]));
    else if constexpr (R == 4)
        return aie::concat(aie::broadcast<int16, W>(s[0]), aie::broadcast<int16, W>(s[1]),
                           aie::broadcast<int16, W>(s[2]), aie::broadcast<int16, W>(s[3]));
    else
        return aie::concat(aie::concat(aie::broadcast<int16, W>(s[0]), aie::broadcast<int16, W>(s[1]),
                                       aie::broadcast<int16, W>(s[2]), aie::broadcast<int16, W>(s[3])),
                           aie::concat(aie::broadcast<int16, W>(s[4]), aie::broadcast<int16, W>(s[5]),
                                       aie::broadcast<int16, W>(s[6]), aie::broadcast<int16, W>(s[7])));
}


template <typename ConfigT>
void layernorm_i8_tiled<ConfigT>::run(input_buffer<in_t>&    in,
                                        const int16_t (&gamma)[COLS * MT_OUTER],
                                        const int16_t (&beta)[COLS * MT_OUTER],
                                        output_buffer<out_t>&  out)
{
    using acc_t = typename ConfigT::acc_scalar_t;
    const in_t*  __restrict in_ptr  = in.data();
          out_t* __restrict out_ptr = out.data();

    const aie::vector<int8, BLK> ones8 = aie::broadcast<int8, BLK>(1);

    for (int bg = 0; bg < BANDS; bg += GROUP) {
        // Padded to the vector width: lanes past the group's rows stay zero and are never stored.
        int32_t sum_x[STAT_LANES]  = {};
        int32_t sum_sq[STAT_LANES] = {};

        for (int g = 0; g < GROUP; ++g) {
            const in_t* __restrict band = in_ptr + (bg + g) * NB * BLK;

            // Lane group m of every microtile belongs to row m, so accumulating along the
            // feature axis keeps the rows' partial sums separate with no shuffling.
            aie::accum<acc_t, BLK> acc_sum = aie::zeros<acc_t, BLK>();
            aie::accum<acc_t, BLK> acc_sq  = aie::zeros<acc_t, BLK>();

            for (int bn = 0; bn < NB; ++bn)
                chess_prepare_for_pipelining
                chess_loop_range(NB, NB)
            {
                aie::vector<int8, BLK> vx = *aie::cbegin_vector<BLK>(band + bn * BLK);
                if constexpr (ConfigT::TRANSPOSE_INPUT) vx = aie::transpose(vx, MT_INNER, MT_OUTER);
                acc_sum = aie::mac(acc_sum, vx, ones8);
                acc_sq  = aie::mac_square(acc_sq, vx);
            }

            // Segmented reduce: halving rounds collapse each MT_INNER lane group and leave row m's
            // total in lane m * MT_INNER (a full reduce would sum the rows together). Done in
            // <=32-lane pieces because a shuffle_down across a 64-lane int32 vector (2048-bit) does
            // not compose into a full logical shift, but a 32-lane (1024-bit) one does.
            const aie::vector<int32, BLK> s_full = acc_sum.template to_vector<int32>(0);
            const aie::vector<int32, BLK> q_full = acc_sq.template to_vector<int32>(0);
            constexpr int SUB = (BLK < 32) ? BLK : 32;
            for (int sb = 0; sb < BLK / SUB; ++sb) {
                aie::vector<int32, SUB> s = s_full.template extract<SUB>(sb);
                aie::vector<int32, SUB> q = q_full.template extract<SUB>(sb);
                for (unsigned step = MT_INNER / 2; step >= 1; step >>= 1) {
                    s = aie::add(s, aie::shuffle_down(s, step));
                    q = aie::add(q, aie::shuffle_down(q, step));
                }
                for (int k = 0; k < SUB / MT_INNER; ++k) {
                    const int m = g * MT_OUTER + sb * (SUB / MT_INNER) + k;
                    sum_x[m]  = s.get(k * MT_INNER);
                    sum_sq[m] = q.get(k * MT_INNER);
                }
            }
        }

        int16_t mu16[STAT_LANES];
        int16_t inv_std16[STAT_LANES];
        base::template row_statistics<STAT_LANES>(sum_x, sum_sq, mu16, inv_std16);

        for (int g = 0; g < GROUP; ++g) {
            const in_t*  __restrict band = in_ptr  + (bg + g) * NB * BLK;
                  out_t* __restrict dst  = out_ptr + (bg + g) * NB * BLK;

            // Widen the per-row statistics to the microtile's lane groups.
            const aie::vector<int16, BLK> mu_v = spread_rows<MT_OUTER, MT_INNER>(mu16 + g * MT_OUTER);
            const aie::vector<int16, BLK> is_v = spread_rows<MT_OUTER, MT_INNER>(inv_std16 + g * MT_OUTER);

            for (int bn = 0; bn < NB; ++bn)
                chess_prepare_for_pipelining
                chess_loop_range(NB, NB)
            {
                aie::vector<int8, BLK> vx = *aie::cbegin_vector<BLK>(band + bn * BLK);
                if constexpr (ConfigT::TRANSPOSE_INPUT) vx = aie::transpose(vx, MT_INNER, MT_OUTER);

                // Already widened in ROM: one block load, no broadcast or concat in the loop.
                const aie::vector<int16, BLK> gamma_blk = *aie::cbegin_vector<BLK>((const int16_t*)gamma + bn * BLK);
                const aie::vector<int16, BLK> beta_blk  = *aie::cbegin_vector<BLK>((const int16_t*)beta + bn * BLK);

                const aie::vector<int16, BLK> vd16 =
                    aie::sub(aie::from_vector<acc_t>(vx), mu_v).template to_vector<int16>(0);

                const aie::accum<acc_t, BLK> acc_fs = aie::mul(is_v, gamma_blk);
                const aie::vector<int16, BLK> fscale = acc_fs.template to_vector<int16>(GAMMA_SHIFT);

                aie::accum<acc_t, BLK> acc_out = aie::mul(vd16, fscale);
                acc_out = aie::add(acc_out, beta_blk);

                auto vout_it = aie::begin_vector<BLK>(dst + bn * BLK);
                *vout_it = acc_out.template to_vector<out_t>(NORM_SHIFT - OUT_SHIFT);
            }
        }
    }
}
