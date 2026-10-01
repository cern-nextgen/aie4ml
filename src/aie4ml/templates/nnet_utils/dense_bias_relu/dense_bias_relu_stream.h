// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class dense_stream_base {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using bias_t        = typename ConfigT::bias_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  dense_stream_base();
};

template<typename ConfigT>
class dense_single_stream : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;
  using bias_t        = typename ConfigT::bias_t;

  void run(input_stream<data_t>*         ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
           output_stream<result_t>*       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_single_stream::run); }
};

template<typename ConfigT>
class dense_first_stream : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;
  using bias_t        = typename ConfigT::bias_t;

  void run(input_stream<data_t>*          ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
           output_cascade<acc_scalar_t>*  outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_first_stream::run); }
};

template<typename ConfigT>
class dense_middle_stream : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  void run(input_stream<data_t>*          ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           input_cascade<acc_scalar_t>*   inCascade,
           output_cascade<acc_scalar_t>*  outCascade);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_middle_stream::run); }
};

template<typename ConfigT>
class dense_last_stream : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  void run(input_stream<data_t>*          ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           input_cascade<acc_scalar_t>*   inCascade,
           output_stream<result_t>*       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_last_stream::run); }
};

// One stream end and one buffer end: a buffer holds microtiles as dense_bias_relu.cpp's do, and a chain with a
// buffer input runs that file's first and middle kernels.
template<typename ConfigT>
class dense_single_stream_in : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;
  using bias_t        = typename ConfigT::bias_t;

  void run(input_stream<data_t>*         ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
           output_buffer<result_t>&       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_single_stream_in::run); }
};

template<typename ConfigT>
class dense_last_stream_in : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  void run(input_stream<data_t>*          ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           input_cascade<acc_scalar_t>*   inCascade,
           output_buffer<result_t>&       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_last_stream_in::run); }
};

template<typename ConfigT>
class dense_single_stream_out : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;
  using bias_t        = typename ConfigT::bias_t;

  void run(input_buffer<data_t>&         ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
           output_stream<result_t>*       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_single_stream_out::run); }
};

template<typename ConfigT>
class dense_last_stream_out : public dense_stream_base<ConfigT> {
public:
  using data_t        = typename ConfigT::data_t;
  using weight_t      = typename ConfigT::weight_t;
  using result_t      = typename ConfigT::result_t;
  using acc_scalar_t  = typename ConfigT::acc_scalar_t;

  void run(input_buffer<data_t>&          ifm,
           const weight_t (&wts)[ConfigT::IN_FEAT_SLICE * ConfigT::OUT_FEAT_SLICE],
           input_cascade<acc_scalar_t>*   inCascade,
           output_stream<result_t>*       ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(dense_last_stream_out::run); }
};
