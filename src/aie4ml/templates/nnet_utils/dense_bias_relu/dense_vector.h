// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class dense_vector_base {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  dense_vector_base();
};

template<typename ConfigT>
class dense_vector_single : public dense_vector_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename dense_vector_base<ConfigT>::acc_scalar_t;
  using bias_t   = typename dense_vector_base<ConfigT>::bias_t;

  void run(input_buffer<data_t>&       ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
           output_buffer<result_t>&      ofm);

  static void registerKernelClass() {
    REGISTER_FUNCTION(dense_vector_single::run);
  }
};

template<typename ConfigT>
class dense_vector_first : public dense_vector_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename dense_vector_base<ConfigT>::acc_scalar_t;
  using bias_t   = typename dense_vector_base<ConfigT>::bias_t;

  void run(input_buffer<data_t>&          ifm,
            const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
            const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
            output_cascade<acc_scalar_t>* outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_vector_first::run); }
};

template<typename ConfigT>
class dense_vector_middle : public dense_vector_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename dense_vector_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<data_t>&          ifm,
            const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
            input_cascade<acc_scalar_t>*  inCascade,
            output_cascade<acc_scalar_t>* outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_vector_middle::run); }
};

template<typename ConfigT>
class dense_vector_last : public dense_vector_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename dense_vector_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<data_t>&         ifm,
            const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
            input_cascade<acc_scalar_t>* inCascade,
            output_buffer<result_t>&        ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_vector_last::run); }
};
