// matmul.cpp with four row microtiles per step instead of two, for AIE-MLv2: its eight accumulator registers hold the
// 4x2 block of 8x8 int32 microtiles, so each block's fixed cost (pipeline fill and drain, stores) is paid once per
// eight output microtiles instead of four. AIE-ML spills at four rows.

#include "matmul_block4x2.h"
using namespace adf;

// B as stored: the tensor itself (B, or B transposed under TRANSPOSE_B), row-major in microtiles, each transposed in
// registers under TRANSPOSE_B.
template<typename ConfigT>
struct matmul_block4x2_b_operand {
  using b_t = typename ConfigT::b_t;
  static constexpr bool TRANSPOSED = ConfigT::TRANSPOSE_B;
  static constexpr int K = ConfigT::K, N = ConfigT::N;
  static constexpr int WIDTH = TRANSPOSED ? ConfigT::K_SLICE : ConfigT::N_SLICE;
  static constexpr int K_STEP = TRANSPOSED ? N * K : K * WIDTH;  // to the next microtile along K
  static constexpr int N_STEP = TRANSPOSED ? N * WIDTH : K * N;  // to the next microtile along N

  __attribute__((always_inline)) static aie::vector<b_t, K * N> load(const b_t* __restrict p) {
    if constexpr (TRANSPOSED) return aie::transpose(aie::load_v<K * N>(p), N, K);
    else return aie::load_v<K * N>(p);
  }
};

template<typename ConfigT>
matmul_block4x2_base<ConfigT>::matmul_block4x2_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(
      ConfigT::K_SLICE * ConfigT::N_SLICE * sizeof(typename ConfigT::b_t) <= 16384,
      "B tile size per kernel must fit in one AIE-ML bank (16 KiB)");
  static_assert(
      ConfigT::K_SLICE % (2 * ConfigT::K) == 0,
      "K_SLICE must be divisible by 2*K");
  static_assert(
      ConfigT::N_SLICE % (2 * ConfigT::N) == 0,
      "N_SLICE must be divisible by 2*N");
  static_assert(
      ConfigT::padded_M % (4 * ConfigT::M) == 0,
      "padded_M must be divisible by 4*M");
  static_assert(
      ConfigT::padded_K == ConfigT::K_SLICE * ConfigT::CAS_LENGTH,
      "padded_K must equal K_SLICE * CAS_LENGTH");
  static_assert(
      ConfigT::PARALLELISM_CONTRACT_OUTER
          ? (ConfigT::padded_N == ConfigT::N_SLICE)
          : (ConfigT::padded_N == ConfigT::N_SLICE * ConfigT::CAS_NUM),
      "padded_N must equal N_SLICE ('outer' keeps N whole) or N_SLICE * CAS_NUM ('inner' slices it)");
}

enum class block4x2_in { none, cascade };
enum class block4x2_out { buffer, cascade };

// One K step of the block: the four A microtiles of its rows and the two B microtiles of its columns.
template<typename ConfigT, typename MMUL>
static inline void block4x2_load(aie::vector<typename ConfigT::a_t, MMUL::size_A> (&A)[4],
                                 aie::vector<typename ConfigT::b_t, MMUL::size_B> (&B)[2],
                                 const typename ConfigT::a_t* __restrict (&pa)[4],
                                 const typename ConfigT::b_t* __restrict (&pb)[2])
{
  using BOp = matmul_block4x2_b_operand<ConfigT>;
  for (int r = 0; r < 4; ++r) chess_unroll_loop(*) {
    if constexpr (ConfigT::TRANSPOSE_A)
      A[r] = aie::transpose(aie::load_v<MMUL::size_A>(pa[r]), ConfigT::K, ConfigT::M);
    else
      A[r] = aie::load_v<MMUL::size_A>(pa[r]);
    pa[r] += MMUL::size_A;
  }
  for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
    B[c] = BOp::load(pb[c]);
    pb[c] += BOp::K_STEP;
  }
}

// Every 4x2 block of output microtiles over this tile's K slice, row blocks outer. Its accumulators start from zero or
// the cascade and leave as stored results or down the cascade, both in block order, row-major within.
template<typename ConfigT, block4x2_in IN, block4x2_out OUT>
static inline void block4x2_tile(const typename ConfigT::a_t* pA, const typename ConfigT::b_t* pB,
                                 typename ConfigT::c_t* pC,
                                 input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                 output_cascade<typename ConfigT::acc_scalar_t>* outCascade)
{
  using a_t = typename ConfigT::a_t;
  using b_t = typename ConfigT::b_t;
  using c_t = typename ConfigT::c_t;
  using BOp = matmul_block4x2_b_operand<ConfigT>;
  static constexpr int M = ConfigT::M, K = ConfigT::K, N = ConfigT::N;
  static constexpr int MB = ConfigT::padded_M / M;
  static constexpr int KS = ConfigT::K_SLICE / K, NB = ConfigT::N_SLICE / N;
  using MMUL = aie::mmul<M, K, N, a_t, b_t, typename ConfigT::acc_scalar_t>;

  for (unsigned z = 0; z < MB; z += 4)
    for (unsigned j = 0; j < NB; j += 2) {
      MMUL C[4][2];
      if constexpr (IN == block4x2_in::cascade) {
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*)
            C[r][c] = MMUL(readincr_v<MMUL::size_C>(inCascade));
      }

      const a_t* __restrict pa[4];
      const b_t* __restrict pb[2];
      for (int r = 0; r < 4; ++r) chess_unroll_loop(*) pa[r] = pA + (z + r) * KS * MMUL::size_A;
      for (int c = 0; c < 2; ++c) chess_unroll_loop(*) pb[c] = pB + (j + c) * BOp::N_STEP;
      aie::vector<a_t, MMUL::size_A> A[4];
      aie::vector<b_t, MMUL::size_B> B[2];

      // The first K step starts the accumulators, or adds on the cascade's partial sums.
      block4x2_load<ConfigT, MMUL>(A, B, pa, pb);
      for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
        for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
          if constexpr (IN == block4x2_in::cascade) C[r][c].mac(A[r], B[c]);
          else C[r][c].mul(A[r], B[c]);
        }

      for (int i = 1; i < KS; ++i)
        chess_prepare_for_pipelining
      {
        block4x2_load<ConfigT, MMUL>(A, B, pa, pb);
        for (int r = 0; r < 4; ++r) chess_unroll_loop(*)
          for (int c = 0; c < 2; ++c) chess_unroll_loop(*)
            C[r][c].mac(A[r], B[c]);
      }

      for (int r = 0; r < 4; ++r) chess_unroll_loop(*) {
        c_t* __restrict pc = pC + ((z + r) * NB + j) * MMUL::size_C;
        for (int c = 0; c < 2; ++c) chess_unroll_loop(*) {
          if constexpr (OUT == block4x2_out::cascade) {
            writeincr(outCascade, C[r][c].to_accum());
          } else {
            aie::store_v(pc, C[r][c].template to_vector<c_t>(ConfigT::SHIFT));
            pc += MMUL::size_C;
          }
        }
      }
    }
}


template<typename ConfigT>
void matmul_block4x2_single<ConfigT>::run(input_buffer<a_t>& A,
                                          input_buffer<b_t>& B,
                                          output_buffer<c_t>& C) {
  block4x2_tile<ConfigT, block4x2_in::none, block4x2_out::buffer>(A.data(), B.data(), C.data(), nullptr, nullptr);
}

template<typename ConfigT>
void matmul_block4x2_first<ConfigT>::run(input_buffer<a_t>& A,
                                         input_buffer<b_t>& B,
                                         output_cascade<acc_scalar_t>* outCascade) {
  block4x2_tile<ConfigT, block4x2_in::none, block4x2_out::cascade>(A.data(), B.data(), nullptr, nullptr, outCascade);
}

template<typename ConfigT>
void matmul_block4x2_middle<ConfigT>::run(input_buffer<a_t>& A,
                                          input_buffer<b_t>& B,
                                          input_cascade<acc_scalar_t>* inCascade,
                                          output_cascade<acc_scalar_t>* outCascade) {
  block4x2_tile<ConfigT, block4x2_in::cascade, block4x2_out::cascade>(A.data(), B.data(), nullptr, inCascade,
                                                                       outCascade);
}

template<typename ConfigT>
void matmul_block4x2_last<ConfigT>::run(input_buffer<a_t>& A,
                                        input_buffer<b_t>& B,
                                        input_cascade<acc_scalar_t>* inCascade,
                                        output_buffer<c_t>& C) {
  block4x2_tile<ConfigT, block4x2_in::cascade, block4x2_out::buffer>(A.data(), B.data(), C.data(), inCascade, nullptr);
}
