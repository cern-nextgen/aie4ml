// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// A Dense fused with a pool over runs of its rows (passes/fuse_ops.py): POOL_GROUPS groups of POOL_ROWS rows
// in, each group whole kernel steps, and one row per group out, the group's sum or average. A step computes ROW_BLOCKS row
// microtiles of CB column tiles, as dense_bias_relu.cpp or dense_block4x2.cpp do, and converts them to the codes those
// would store. The last kernel of a cascade sums a group's codes on the multiplier: an SR x 8 x N product of 8 code rows
// (zero past a step's own) with a selector whose row c is all ones, so one 16-lane accumulator holds the sums of the
// step's column tiles beside its own. A group converts its sums once, at POOL_SHIFT, into its row of each column's
// output microtile. Every kernel of a cascade walks column tiles, groups, steps.

#include "dense_pool.h"
using namespace adf;

template<typename ConfigT>
struct pool_geometry {
  static constexpr int M = ConfigT::M, K = ConfigT::K, N = ConfigT::N, RB = ConfigT::ROW_BLOCKS;
  static constexpr int KS = ConfigT::IN_FEAT_SLICE / K, NB = ConfigT::OUT_FEAT_SLICE / N;
  static constexpr int CB = __AIE_ARCH__ == 10 ? 1 : 2;  // column tiles a step computes: AIE1 holds fewer accumulators
  static constexpr int SEL = 8;                            // code rows a selector product sums
  static constexpr int SR = 16 / N;                        // its rows, one a column tile
  static constexpr int MP = SEL / M < RB ? SEL / M : RB;   // the row microtiles in them
  static constexpr int GROUPS = ConfigT::POOL_GROUPS;
  static constexpr int GROUP_STEPS = ConfigT::POOL_ROWS / (RB * M);
};

// Selector c: an SR x SEL matrix whose row c is all ones.
template<int SR, int SEL>
struct pool_selectors {
  alignas(32) int8 v[2 * SR * SEL];
  constexpr pool_selectors() : v{} {
    for (int c = 0; c < 2; ++c)
      for (int k = 0; k < SEL; ++k) v[(c * SR + c) * SEL + k] = 1;
  }
};

template<int SR, int SEL>
inline constexpr pool_selectors<SR, SEL> POOL_SELECTORS{};

template<typename ConfigT>
dense_pool_base<ConfigT>::dense_pool_base() {
  using G = pool_geometry<ConfigT>;
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(ConfigT::OUT_FEAT_SLICE * ConfigT::IN_FEAT_SLICE * sizeof(weight_t) <= ConfigT::BANK_BYTES,
                "Weight size per tile must not exceed one device memory bank");
  static_assert(ConfigT::IN_FEAT_SLICE % (2 * ConfigT::K) == 0 && ConfigT::OUT_FEAT_SLICE % (2 * ConfigT::N) == 0,
                "a tile's K and N slices hold whole pairs of microtiles");
  static_assert(!ConfigT::PARALLELISM_CONTRACT_OUTER
                && ConfigT::padded_IN_FEAT == ConfigT::IN_FEAT_SLICE * ConfigT::CAS_LENGTH
                && ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE * ConfigT::CAS_NUM,
                "a pool splits K over the cascade and N over the chains ('inner')");
  static_assert(G::SEL % G::M == 0 && G::RB % G::MP == 0 && ConfigT::POOL_ROWS % (G::RB * G::M) == 0
                && ConfigT::padded_independent_extent == G::GROUPS * ConfigT::POOL_ROWS,
                "each group is whole steps of whole selector products");
  static_assert(sizeof(result_t) == 1 && ConfigT::POOL_SHIFT >= 0 && ConfigT::POOL_EXTENT % G::M == 0
                && ConfigT::POOL_EXTENT >= G::GROUPS && (G::N == 4 || G::N == 8),
                "the groups leave as 8-bit codes in whole microtiles of 4 or 8 columns, at a right shift");
}

// The codes a tile's rows would leave as unreduced: requantized at SHIFT, then the ReLU.
template<typename ConfigT, typename MMUL>
static inline aie::vector<typename ConfigT::result_t, MMUL::size_C> pool_codes(MMUL& C) {
  using result_t = typename ConfigT::result_t;
  auto y = C.template to_vector<result_t>(ConfigT::SHIFT);
  if constexpr (ConfigT::USE_RELU) y = aie::max(y, result_t(0));
  return y;
}

// The SEL code rows of a selector product: MP row microtiles of column tile c from r, then zero rows.
template<typename ConfigT, int MP, int SEL, typename MMUL, int RB, int CB>
static inline aie::vector<typename ConfigT::result_t, SEL * ConfigT::N> pool_rows(MMUL (&C)[RB][CB], int r, int c) {
  aie::vector<typename ConfigT::result_t, SEL * ConfigT::N> rows = aie::zeros<typename ConfigT::result_t, SEL * ConfigT::N>();
  for (int i = 0; i < MP; ++i) chess_unroll_loop(*) rows.insert(i, pool_codes<ConfigT>(C[r + i][c]));
  return rows;
}

// The M rows of a microtile, each the bias row.
template<int M, typename VT, int N>
static inline aie::vector<VT, M * N> pool_bias(const aie::vector<VT, N>& row) {
  aie::vector<VT, M * N> rows;
  for (int i = 0; i < M; ++i) chess_unroll_loop(*) rows.insert(i, row);
  return rows;
}

// Row `row` of the output microtiles of the step's column tiles from j, `codes` holding them in turn.
template<typename ConfigT, int CB>
static inline void pool_store_rows(typename ConfigT::result_t* pC, unsigned row, unsigned j,
                                   aie::vector<typename ConfigT::result_t, 16> codes)
{
  static constexpr int M = ConfigT::M, N = ConfigT::N, NB = ConfigT::OUT_FEAT_SLICE / N;
  const auto words = codes.template cast_to<int32>();
  for (int c = 0; c < CB; ++c) chess_unroll_loop(*) {
    int32* p = (int32*)(pC + ((row / M) * NB + j + c) * (M * N) + (row % M) * N);
    for (int w = 0; w < N / 4; ++w) chess_unroll_loop(*) p[w] = words[c * (N / 4) + w];
  }
}

// One K step of a step: the ROW_BLOCKS LHS microtiles of its rows and the CB RHS microtiles of its columns.
template<typename ConfigT, typename MMUL, int CB>
static inline void pool_load(aie::vector<typename ConfigT::data_t, MMUL::size_A> (&A)[ConfigT::ROW_BLOCKS],
                                   aie::vector<typename ConfigT::weight_t, MMUL::size_B> (&W)[CB],
                                   const typename ConfigT::data_t* __restrict (&pa)[ConfigT::ROW_BLOCKS],
                                   const typename ConfigT::weight_t __aie_dm_resource_a* __restrict (&pb)[CB])
{
  static constexpr int NB = ConfigT::OUT_FEAT_SLICE / ConfigT::N;
  for (int r = 0; r < ConfigT::ROW_BLOCKS; ++r) chess_unroll_loop(*) {
    if constexpr (ConfigT::TRANSPOSE_INPUT)
      A[r] = aie::transpose(aie::load_v<MMUL::size_A>(pa[r]), ConfigT::K, ConfigT::M);
    else
      A[r] = aie::load_v<MMUL::size_A>(pa[r]);
    pa[r] += MMUL::size_A;
  }
  for (int c = 0; c < CB; ++c) chess_unroll_loop(*) {
    W[c] = aie::load_v<MMUL::size_B>(pb[c]);
    pb[c] += MMUL::size_B * NB;
  }
}

enum class pool_in { bias, cascade };
enum class pool_out { sums, cascade };

template<typename ConfigT, pool_in IN, pool_out OUT>
static inline void pool_tile(const typename ConfigT::data_t* pA, const typename ConfigT::weight_t* pB,
                                   const typename ConfigT::bias_t* pBias, typename ConfigT::result_t* pC,
                                   input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                   output_cascade<typename ConfigT::acc_scalar_t>* outCascade)
{
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using bias_t   = typename ConfigT::bias_t;
  using result_t = typename ConfigT::result_t;
  using G = pool_geometry<ConfigT>;
  static constexpr int M = G::M, K = G::K, N = G::N, RB = G::RB, CB = G::CB, KS = G::KS, NB = G::NB;
  using MMUL = aie::mmul<M, K, N, data_t, weight_t, typename ConfigT::acc_scalar_t>;
  using SUM = aie::mmul<G::SR, G::SEL, N, int8, result_t, typename ConfigT::acc_scalar_t>;
  using acc_t = aie::accum<typename ConfigT::acc_scalar_t, MMUL::size_C>;

  for (unsigned j = 0; j < NB; j += CB) {
    if constexpr (OUT == pool_out::sums) {
      // The output microtiles holding rows past the last group are zero there; the groups' rows then go in.
      for (int b = G::GROUPS / M; b < ConfigT::POOL_EXTENT / M; ++b) chess_unroll_loop(*)
        for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
          aie::store_v(pC + (b * NB + j + c) * MMUL::size_C, aie::zeros<result_t, MMUL::size_C>());
    }
    for (unsigned g = 0; g < G::GROUPS; ++g) {
      SUM P(aie::zeros<typename ConfigT::acc_scalar_t, SUM::size_C>());
      for (unsigned s = 0; s < G::GROUP_STEPS; ++s) {
        const unsigned z = (g * G::GROUP_STEPS + s) * RB;  // the step's first row microtile
        MMUL C[RB][CB];
        if constexpr (IN == pool_in::cascade) {
          for (int r = 0; r < RB; ++r) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
              C[r][c] = MMUL(readincr_v<MMUL::size_C>(inCascade));
        }

        const data_t* __restrict pa[RB];
        const weight_t __aie_dm_resource_a* __restrict pb[CB];
        for (int r = 0; r < RB; ++r) chess_unroll_loop(*) pa[r] = pA + (z + r) * KS * MMUL::size_A;
        for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
          pb[c] = (const weight_t __aie_dm_resource_a*)pB + (j + c) * MMUL::size_B;
        aie::vector<data_t, MMUL::size_A> A[RB];
        aie::vector<weight_t, MMUL::size_B> W[CB];

        // The first K step starts the accumulators: from the bias (or zero), or on the cascade's partial sums.
        pool_load<ConfigT, MMUL, CB>(A, W, pa, pb);
        if constexpr (IN == pool_in::bias && ConfigT::USE_BIAS) {
          acc_t b[CB];
          for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
            b[c] = acc_t(pool_bias<M, bias_t, N>(aie::load_v<N>(pBias + (j + c) * N)));
          for (int r = 0; r < RB; ++r) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*) {
              C[r][c] = MMUL(b[c]);
              C[r][c].mac(A[r], W[c]);
            }
        } else {
          for (int r = 0; r < RB; ++r) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*) {
              if constexpr (IN == pool_in::bias) C[r][c].mul(A[r], W[c]);
              else C[r][c].mac(A[r], W[c]);
            }
        }

        for (int i = 1; i < KS; ++i)
          chess_prepare_for_pipelining
        {
          pool_load<ConfigT, MMUL, CB>(A, W, pa, pb);
          for (int r = 0; r < RB; ++r) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
              C[r][c].mac(A[r], W[c]);
        }

        if constexpr (OUT == pool_out::cascade) {
          for (int r = 0; r < RB; ++r) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*)
              writeincr(outCascade, C[r][c].to_accum());
        } else {
          for (int r = 0; r < RB; r += G::MP) chess_unroll_loop(*)
            for (int c = 0; c < CB; ++c) chess_unroll_loop(*) {
              const auto sel = aie::load_v<SUM::size_A>(POOL_SELECTORS<G::SR, G::SEL>.v + c * SUM::size_A);
              P.mac(sel, pool_rows<ConfigT, G::MP, G::SEL>(C, r, c));
            }
        }
      }
      if constexpr (OUT == pool_out::sums)
        pool_store_rows<ConfigT, CB>(pC, g, j, P.template to_vector<result_t>(ConfigT::POOL_SHIFT));
    }
  }
}


template<typename ConfigT>
void dense_pool_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                           output_buffer<result_t>& ofm)
{
  pool_tile<ConfigT, pool_in::bias, pool_out::sums>(ifm.data(), wts, bias, ofm.data(), nullptr, nullptr);
}

template<typename ConfigT>
void dense_pool_first<ConfigT>::run(input_buffer<data_t>& ifm,
                                          const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                          const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                          output_cascade<acc_scalar_t>* outCascade)
{
  pool_tile<ConfigT, pool_in::bias, pool_out::cascade>(ifm.data(), wts, bias, nullptr, nullptr, outCascade);
}

template<typename ConfigT>
void dense_pool_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                           input_cascade<acc_scalar_t>* inCascade,
                                           output_cascade<acc_scalar_t>* outCascade)
{
  pool_tile<ConfigT, pool_in::cascade, pool_out::cascade>(ifm.data(), wts, nullptr, nullptr, inCascade, outCascade);
}

template<typename ConfigT>
void dense_pool_last<ConfigT>::run(input_buffer<data_t>& ifm,
                                         const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
                                         input_cascade<acc_scalar_t>* inCascade,
                                         output_buffer<result_t>& ofm)
{
  pool_tile<ConfigT, pool_in::cascade, pool_out::sums>(ifm.data(), wts, nullptr, ofm.data(), inCascade, nullptr);
}
