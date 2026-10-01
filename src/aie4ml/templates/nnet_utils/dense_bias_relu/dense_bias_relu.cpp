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


template<typename ConfigT>
void dense_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                output_buffer<result_t>& ofm)
{

  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;

  const data_t*      pA    = ifm.data();
  const weight_t __aie_dm_resource_a* pB = (const weight_t __aie_dm_resource_a*)wts;
  const bias_t* pBias = bias;
  result_t*          pC    = ofm.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    result_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    result_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const weight_t __aie_dm_resource_a* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const weight_t __aie_dm_resource_a* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      }else{
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;

      aie::vector<weight_t, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * (colB / N);
      aie::vector<weight_t, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * (colB / N);

      MMUL C00, C01, C10, C11;

      if constexpr (ConfigT::USE_BIAS) {
        aie::vector<bias_t, N> bias_v_0 = aie::load_v<N>(pBias + j * N);
        aie::vector<bias_t, N> bias_v_1 = aie::load_v<N>(pBias + (j + 1) * N);

        auto bias_block_0 = replicate_rows<M, bias_t, N>(bias_v_0);
        auto bias_block_1 = replicate_rows<M, bias_t, N>(bias_v_1);

        C00 = bias_block_0; C00.mac(A0, B0);
        C01 = bias_block_1; C01.mac(A0, B1);
        C10 = bias_block_0; C10.mac(A1, B0);
        C11 = bias_block_1; C11.mac(A1, B1);
      } else {
        C00.mul(A0, B0);
        C01.mul(A0, B1);
        C10.mul(A1, B0);
        C11.mul(A1, B1);
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

      if constexpr (ConfigT::USE_RELU) {
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
void dense_first<ConfigT>::run(input_buffer<data_t>& ifm,
                               const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                               const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                               output_cascade<acc_scalar_t>* outCascade)
{
  static constexpr int rowA = ConfigT::padded_independent_extent;
  static constexpr int colA = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;

  const data_t*   pA = ifm.data();
  const weight_t __aie_dm_resource_a* pB = (const weight_t __aie_dm_resource_a*)wts;
  const bias_t* pBias = bias;

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const weight_t __aie_dm_resource_a* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const weight_t __aie_dm_resource_a* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

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

      MMUL C00, C01, C10, C11;
      if constexpr (ConfigT::USE_BIAS) {
        auto bias_block_0 = replicate_rows<M, bias_t, N>(aie::load_v<N>(pBias + j * N));
        auto bias_block_1 = replicate_rows<M, bias_t, N>(aie::load_v<N>(pBias + (j + 1) * N));
        C00 = bias_block_0; C00.mac(A0, B0);
        C01 = bias_block_1; C01.mac(A0, B1);
        C10 = bias_block_0; C10.mac(A1, B0);
        C11 = bias_block_1; C11.mac(A1, B1);
      } else {
        C00.mul(A0, B0);
        C01.mul(A0, B1);
        C10.mul(A1, B0);
        C11.mul(A1, B1);
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

      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}


template<typename ConfigT>
void dense_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                input_cascade<acc_scalar_t>* inCascade,
                                output_cascade<acc_scalar_t>* outCascade)
{
  static constexpr int rowA = ConfigT::padded_independent_extent;
  static constexpr int colA = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;

  const data_t*   pA = ifm.data();
  const weight_t __aie_dm_resource_a* pB = (const weight_t __aie_dm_resource_a*)wts;

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      auto acc00 = readincr_v<MMUL::size_C>(inCascade);
      auto acc01 = readincr_v<MMUL::size_C>(inCascade);
      auto acc10 = readincr_v<MMUL::size_C>(inCascade);
      auto acc11 = readincr_v<MMUL::size_C>(inCascade);

      MMUL C00; C00 = acc00;
      MMUL C01; C01 = acc01;
      MMUL C10; C10 = acc10;
      MMUL C11; C11 = acc11;

      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const weight_t __aie_dm_resource_a* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const weight_t __aie_dm_resource_a* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

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

      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

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

      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}


template<typename ConfigT>
void dense_last<ConfigT>::run(input_buffer<data_t>& ifm,
                              const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                              input_cascade<acc_scalar_t>* inCascade,
                              output_buffer<result_t>& ofm)
{
  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;

  const data_t*      pA    = ifm.data();
  const weight_t __aie_dm_resource_a* pB = (const weight_t __aie_dm_resource_a*)wts;
  result_t*          pC    = ofm.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    result_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    result_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const weight_t __aie_dm_resource_a* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const weight_t __aie_dm_resource_a* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      MMUL C00(readincr_v<MMUL::size_C>(inCascade));
      MMUL C01(readincr_v<MMUL::size_C>(inCascade));
      MMUL C10(readincr_v<MMUL::size_C>(inCascade));
      MMUL C11(readincr_v<MMUL::size_C>(inCascade));

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

      C00.mac(A0, B0);  C01.mac(A0, B1);
      C10.mac(A1, B0);  C11.mac(A1, B1);

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

      if constexpr (ConfigT::USE_RELU) {
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
