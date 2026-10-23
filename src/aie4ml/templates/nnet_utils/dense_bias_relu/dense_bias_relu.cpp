// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "dense_bias_relu.h"
using namespace adf;

template<typename ConfigT>
dense_base<ConfigT>::dense_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(
        ConfigT::OUT_FEAT_SLICE * ConfigT::IN_FEAT_SLICE * sizeof(weight_t) <= ConfigT::BANK_BYTES,
        "Weight size per tile must not exceed one device memory bank");
  static_assert(
        ConfigT::IN_FEAT_SLICE % (2 * ConfigT::N) == 0,
        "IN_FEAT_SLICE must be divisible by 2*K");
  static_assert(
        ConfigT::OUT_FEAT_SLICE % (2 * ConfigT::N) == 0,
        "OUT_FEAT_SLICE must be divisible by 2*N");
static_assert(
    ConfigT::padded_independent_extent % (2 * ConfigT::M) == 0,
    "padded_independent_extent must be divisible by 2*M");
  static_assert(
        ConfigT::padded_IN_FEAT == ConfigT::IN_FEAT_SLICE * ConfigT::CAS_LENGTH,
        "padded_IN_FEAT must equal IN_FEAT_SLICE * CAS_LENGTH");
  static_assert(
        ConfigT::PARALLELISM_CONTRACT_OUTER
            ? (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE)
            : (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE * ConfigT::CAS_NUM),
        "padded_OUT_FEAT must equal OUT_FEAT_SLICE * CAS_NUM ('inner') or OUT_FEAT_SLICE ('outer')");
}

template<int M, typename VT, int N>
struct row_replicator;

template<typename VT, int N>
struct row_replicator<2, VT, N> {
  static inline aie::vector<VT, 2 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row);
  }
};

template<typename VT, int N>
struct row_replicator<4, VT, N> {
  static inline aie::vector<VT, 4 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row);
  }
};

template<typename VT, int N>
struct row_replicator<8, VT, N> {
  static inline aie::vector<VT, 8 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row, row, row, row, row);
  }
};

template<int M, typename VT, int N>
static inline aie::vector<VT, M * N>
replicate_rows(const aie::vector<VT, N>& row) {
  static_assert(M == 2 || M == 4 || M == 8, "Unsupported M; add more specializations.");
  return row_replicator<M, VT, N>::run(row);
}


// How a kernel's accumulators start, and how its sums leave: from the bias (or zero) or from the cascade; stored as the
// layer's codes or sent down the cascade. A chain's kernels differ in these only: single (bias, store), first (bias,
// cascade), middle (cascade, cascade), last (cascade, store).
enum class dense_in { bias, cascade };
enum class dense_out { store, cascade };

// Two row microtiles by two column tiles a step, over the tile's K slice.
template<typename ConfigT, dense_in IN, dense_out OUT>
static inline void dense_tile(const typename ConfigT::data_t* pA, const typename ConfigT::weight_t* wts,
                              const typename ConfigT::bias_t* pBias, typename ConfigT::result_t* pC,
                              input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                              output_cascade<typename ConfigT::acc_scalar_t>* outCascade)
{
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;
  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, typename ConfigT::acc_scalar_t>;

  const weight_t __aie_dm_resource_a* pB = (const weight_t __aie_dm_resource_a*)wts;

  for (unsigned z = 0; z < rowA / M; z += 2) {
    result_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    result_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const weight_t __aie_dm_resource_a* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const weight_t __aie_dm_resource_a* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      MMUL C00, C01, C10, C11;
      if constexpr (IN == dense_in::cascade) {
        C00 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        C01 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        C10 = MMUL(readincr_v<MMUL::size_C>(inCascade));
        C11 = MMUL(readincr_v<MMUL::size_C>(inCascade));
      }

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;
      aie::vector<weight_t, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (colB / N);
      aie::vector<weight_t, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (colB / N);

      // The first K step starts the accumulators: on the bias, from zero, or on the cascade's partial sums.
      if constexpr (IN == dense_in::bias && ConfigT::USE_BIAS) {
        auto bias_block_0 = replicate_rows<M, bias_t, N>(aie::load_v<N>(pBias + j * N));
        auto bias_block_1 = replicate_rows<M, bias_t, N>(aie::load_v<N>(pBias + (j + 1) * N));
        C00 = bias_block_0; C00.mac(A0, B0);
        C01 = bias_block_1; C01.mac(A0, B1);
        C10 = bias_block_0; C10.mac(A1, B0);
        C11 = bias_block_1; C11.mac(A1, B1);
      } else if constexpr (IN == dense_in::bias) {
        C00.mul(A0, B0);
        C01.mul(A0, B1);
        C10.mul(A1, B0);
        C11.mul(A1, B1);
      } else {
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_INPUT) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A;
        pA2 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (colB / N);
        B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (colB / N);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      if constexpr (OUT == dense_out::cascade) {
        writeincr(outCascade, C00.to_accum());
        writeincr(outCascade, C01.to_accum());
        writeincr(outCascade, C10.to_accum());
        writeincr(outCascade, C11.to_accum());
      } else if constexpr (ConfigT::USE_RELU) {
        aie::store_v(pC1, aie::max(C00.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC1, aie::max(C01.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C10.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C11.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
      } else {
        aie::store_v(pC1, C00.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
      }
    }
  }
}


template<typename ConfigT>
void dense_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                output_buffer<result_t>& ofm)
{
  dense_tile<ConfigT, dense_in::bias, dense_out::store>(ifm.data(), wts, bias, ofm.data(), nullptr, nullptr);
}

template<typename ConfigT>
void dense_first<ConfigT>::run(input_buffer<data_t>& ifm,
                               const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                               const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                               output_cascade<acc_scalar_t>* outCascade)
{
  dense_tile<ConfigT, dense_in::bias, dense_out::cascade>(ifm.data(), wts, bias, nullptr, nullptr, outCascade);
}

template<typename ConfigT>
void dense_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                input_cascade<acc_scalar_t>* inCascade,
                                output_cascade<acc_scalar_t>* outCascade)
{
  dense_tile<ConfigT, dense_in::cascade, dense_out::cascade>(ifm.data(), wts, nullptr, nullptr, inCascade, outCascade);
}

template<typename ConfigT>
void dense_last<ConfigT>::run(input_buffer<data_t>& ifm,
                              const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                              input_cascade<acc_scalar_t>* inCascade,
                              output_buffer<result_t>& ofm)
{
  dense_tile<ConfigT, dense_in::cascade, dense_out::store>(ifm.data(), wts, nullptr, ofm.data(), inCascade, nullptr);
}
