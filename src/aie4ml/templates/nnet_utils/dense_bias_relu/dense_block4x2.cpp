// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// dense_bias_relu.cpp with four row microtiles per step instead of two, for AIE-MLv2: its eight accumulator registers
// hold the 4x2 block of 8x8 int32 microtiles, so each block's fixed cost (pipeline fill and drain, bias, stores) is
// paid once per eight output microtiles instead of four. AIE-ML spills at four rows.

#include "dense_block4x2.h"
using namespace adf;

template<typename ConfigT>
dense_block4x2_base<ConfigT>::dense_block4x2_base() {
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
        ConfigT::padded_independent_extent % (4 * ConfigT::M) == 0,
        "padded_independent_extent must be divisible by 4*M");
  static_assert(
        ConfigT::padded_IN_FEAT == ConfigT::IN_FEAT_SLICE * ConfigT::CAS_LENGTH,
        "padded_IN_FEAT must equal IN_FEAT_SLICE * CAS_LENGTH");
  static_assert(
        ConfigT::PARALLELISM_CONTRACT_OUTER
            ? (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE)
            : (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE * ConfigT::CAS_NUM),
        "padded_OUT_FEAT must equal OUT_FEAT_SLICE * CAS_NUM ('inner') or OUT_FEAT_SLICE ('outer')");
}

// The M rows of a microtile, each the bias row.
template<int M, typename VT, int N>
static inline aie::vector<VT, M * N> block4x2_bias(const aie::vector<VT, N>& row) {
  static_assert(M == 2 || M == 4 || M == 8, "Unsupported M");
  if constexpr (M == 2) return aie::concat(row, row);
  else if constexpr (M == 4) return aie::concat(row, row, row, row);
  else return aie::concat(row, row, row, row, row, row, row, row);
}

enum class block4x2_in { bias, cascade };
enum class block4x2_out { buffer, cascade };

// One K step of the block: the four LHS microtiles of its rows and the two RHS microtiles of its columns.
template<typename ConfigT, typename MMUL>
static inline void block4x2_load(aie::vector<typename ConfigT::data_t, MMUL::size_A> (&A)[4],
                              aie::vector<typename ConfigT::weight_t, MMUL::size_B> (&W)[2],
                              const typename ConfigT::data_t* __restrict (&pa)[4],
                              const typename ConfigT::weight_t __aie_dm_resource_a* __restrict (&pb)[2])
{
  static constexpr int NB = ConfigT::OUT_FEAT_SLICE / ConfigT::N;
  for (int r = 0; r < 4; ++r) chess_unroll_loop(*) {
    if constexpr (ConfigT::TRANSPOSE_INPUT)
      A[r] = aie::transpose(aie::load_v<MMUL::size_A>(pa[r]), ConfigT::K, ConfigT::M);
    else
      A[r] = aie::load_v<MMUL::size_A>(pa[r]);
    pa[r] += MMUL::size_A;
  }
  for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
    W[c] = aie::load_v<MMUL::size_B>(pb[c]);
    pb[c] += MMUL::size_B * NB;
  }
}

// Every 4x2 block of output microtiles over this tile's K slice, row blocks outer. Its accumulators start from the
// bias or the cascade and leave as stored results or down the cascade, both in block order, row-major within.
template<typename ConfigT, block4x2_in IN, block4x2_out OUT>
static inline void block4x2_tile(const typename ConfigT::data_t* pA, const typename ConfigT::weight_t* pB,
                              const typename ConfigT::bias_t* pBias, typename ConfigT::result_t* pC,
                              input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                              output_cascade<typename ConfigT::acc_scalar_t>* outCascade)
{
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using bias_t   = typename ConfigT::bias_t;
  using result_t = typename ConfigT::result_t;
  static constexpr int M = ConfigT::M, K = ConfigT::K, N = ConfigT::N;
  static constexpr int MB = ConfigT::padded_independent_extent / M;
  static constexpr int KS = ConfigT::IN_FEAT_SLICE / K, NB = ConfigT::OUT_FEAT_SLICE / N;
  using MMUL = aie::mmul<M, K, N, data_t, weight_t, typename ConfigT::acc_scalar_t>;
  using acc_t = aie::accum<typename ConfigT::acc_scalar_t, MMUL::size_C>;

  for (unsigned z = 0; z < MB; z += 4)
    for (unsigned j = 0; j < NB; j += 2) {
      MMUL C[4][2];
      if constexpr (IN == block4x2_in::cascade) {
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*)
            C[r][c] = MMUL(readincr_v<MMUL::size_C>(inCascade));
      }

      const data_t* __restrict pa[4];
      const weight_t __aie_dm_resource_a* __restrict pb[2];
      for (int r = 0; r < 4; ++r) chess_unroll_loop(*) pa[r] = pA + (z + r) * KS * MMUL::size_A;
      for (int c = 0; c < 2; ++c) chess_unroll_loop(*) pb[c] = (const weight_t __aie_dm_resource_a*)pB + (j + c) * MMUL::size_B;
      aie::vector<data_t, MMUL::size_A> A[4];
      aie::vector<weight_t, MMUL::size_B> W[2];

      // The first K step starts the accumulators: from the bias (or zero), or on the cascade's partial sums.
      block4x2_load<ConfigT, MMUL>(A, W, pa, pb);
      if constexpr (IN == block4x2_in::bias && ConfigT::USE_BIAS) {
        acc_t b[2];
        for (int c = 0; c < 2; ++c) chess_unroll_loop(*)
          b[c] = acc_t(block4x2_bias<M, bias_t, N>(aie::load_v<N>(pBias + (j + c) * N)));
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
            C[r][c] = MMUL(b[c]);
            C[r][c].mac(A[r], W[c]);
          }
      } else {
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
            if constexpr (IN == block4x2_in::bias) C[r][c].mul(A[r], W[c]);
            else C[r][c].mac(A[r], W[c]);
          }
      }

      for (int i = 1; i < KS; ++i)
        chess_prepare_for_pipelining
      {
        block4x2_load<ConfigT, MMUL>(A, W, pa, pb);
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*)
            C[r][c].mac(A[r], W[c]);
      }

      for (int r = 0; r < 4; ++r) chess_unroll_loop(*) {
        result_t* __restrict pc = pC + ((z + r) * NB + j) * MMUL::size_C;
        for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
          if constexpr (OUT == block4x2_out::cascade) {
            writeincr(outCascade, C[r][c].to_accum());
          } else {
            auto y = C[r][c].template to_vector<result_t>(ConfigT::SHIFT);
            if constexpr (ConfigT::USE_RELU) y = aie::max(y, result_t(0));
            aie::store_v(pc, y);
            pc += MMUL::size_C;
          }
        }
      }
    }
}


template<typename ConfigT>
void dense_block4x2_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                      const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                      const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                      output_buffer<result_t>& ofm)
{
  block4x2_tile<ConfigT, block4x2_in::bias, block4x2_out::buffer>(ifm.data(), wts, bias, ofm.data(), nullptr, nullptr);
}

template<typename ConfigT>
void dense_block4x2_first<ConfigT>::run(input_buffer<data_t>& ifm,
                                     const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                     const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                     output_cascade<acc_scalar_t>* outCascade)
{
  block4x2_tile<ConfigT, block4x2_in::bias, block4x2_out::cascade>(ifm.data(), wts, bias, nullptr, nullptr, outCascade);
}

template<typename ConfigT>
void dense_block4x2_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                      const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                      input_cascade<acc_scalar_t>* inCascade,
                                      output_cascade<acc_scalar_t>* outCascade)
{
  block4x2_tile<ConfigT, block4x2_in::cascade, block4x2_out::cascade>(ifm.data(), wts, nullptr, nullptr, inCascade, outCascade);
}

template<typename ConfigT>
void dense_block4x2_last<ConfigT>::run(input_buffer<data_t>& ifm,
                                    const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                    input_cascade<acc_scalar_t>* inCascade,
                                    output_buffer<result_t>& ofm)
{
  block4x2_tile<ConfigT, block4x2_in::cascade, block4x2_out::buffer>(ifm.data(), wts, nullptr, ofm.data(), inCascade, nullptr);
}
