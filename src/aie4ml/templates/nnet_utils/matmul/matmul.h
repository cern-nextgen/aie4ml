#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

// B as stored: the tensor itself (B, or B transposed under TRANSPOSE_B), row-major in blocks of B_ROWS rows by one
// microtile's columns. An mmul operand stacks a microtile's rows of blocks, transposed in registers under TRANSPOSE_B.
template<typename ConfigT>
struct matmul_b_operand {
  using b_t = typename ConfigT::b_t;
  static constexpr bool TRANSPOSED = ConfigT::TRANSPOSE_B;
  static constexpr int K = ConfigT::K, N = ConfigT::N, ROWS = ConfigT::B_ROWS;
  static constexpr int COLS = TRANSPOSED ? K : N;
  static constexpr int WIDTH = TRANSPOSED ? ConfigT::K_SLICE : ConfigT::N_SLICE;
  static constexpr int STACK = (TRANSPOSED ? N : K) / ROWS;
  static constexpr int K_STEP = TRANSPOSED ? ROWS * K : K * WIDTH;  // to the next microtile along K
  static constexpr int N_STEP = TRANSPOSED ? N * WIDTH : ROWS * N;  // to the next microtile along N
  static_assert(STACK * ROWS == (TRANSPOSED ? N : K), "B_ROWS must divide a microtile's rows of the stored B");

  __attribute__((always_inline)) static aie::vector<b_t, K * N> load(const b_t* __restrict p) {
    aie::vector<b_t, K * N> v;
    if constexpr (STACK == 1) {
      v = aie::load_v<K * N>(p);
    } else {
      for (int s = 0; s < STACK; ++s)
        chess_flatten_loop
      {
        v.template insert<ROWS * COLS>(s, aie::load_v<ROWS * COLS>(p + s * ROWS * WIDTH));
      }
    }
    if constexpr (TRANSPOSED) return aie::transpose(v, N, K);
    else return v;
  }
};

template<typename ConfigT>
class matmul_base {
public:
  using a_t          = typename ConfigT::a_t;
  using b_t          = typename ConfigT::b_t;
  using c_t          = typename ConfigT::c_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;

  matmul_base();
};

template<typename ConfigT>
class matmul_single : public matmul_base<ConfigT> {
public:
  using a_t          = typename ConfigT::a_t;
  using b_t          = typename ConfigT::b_t;
  using c_t          = typename ConfigT::c_t;
  using acc_scalar_t = typename matmul_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<a_t>& A,
           input_buffer<b_t>& B,
           output_buffer<c_t>& C);

  static void registerKernelClass() {
    REGISTER_FUNCTION(matmul_single::run);
  }
};

template<typename ConfigT>
class matmul_first : public matmul_base<ConfigT> {
public:
  using a_t          = typename ConfigT::a_t;
  using b_t          = typename ConfigT::b_t;
  using acc_scalar_t = typename matmul_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<a_t>& A,
           input_buffer<b_t>& B,
           output_cascade<acc_scalar_t>* outCascade);

  static void registerKernelClass() {
    REGISTER_FUNCTION(matmul_first::run);
  }
};

template<typename ConfigT>
class matmul_middle : public matmul_base<ConfigT> {
public:
  using a_t          = typename ConfigT::a_t;
  using b_t          = typename ConfigT::b_t;
  using acc_scalar_t = typename matmul_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<a_t>& A,
           input_buffer<b_t>& B,
           input_cascade<acc_scalar_t>* inCascade,
           output_cascade<acc_scalar_t>* outCascade);

  static void registerKernelClass() {
    REGISTER_FUNCTION(matmul_middle::run);
  }
};

template<typename ConfigT>
class matmul_last : public matmul_base<ConfigT> {
public:
  using a_t          = typename ConfigT::a_t;
  using b_t          = typename ConfigT::b_t;
  using c_t          = typename ConfigT::c_t;
  using acc_scalar_t = typename matmul_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<a_t>& A,
           input_buffer<b_t>& B,
           input_cascade<acc_scalar_t>* inCascade,
           output_buffer<c_t>& C);

  static void registerKernelClass() {
    REGISTER_FUNCTION(matmul_last::run);
  }
};
