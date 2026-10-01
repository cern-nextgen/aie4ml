#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

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
