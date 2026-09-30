// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Conv2D row bands, one tile each, handing their neighbours the rows the neighbours' windows read. Band b's window
// reads HALO_TOP rows of band b - 1 and HALO_BOTTOM rows of band b + 1 (the zero border where the image ends), and
// it writes, beside its own rows, its first SEND_FIRST rows for band b - 1 and its last SEND_LAST rows for band
// b + 1 of the next layer. Each of those is a buffer of its own with one reader, shared wherever both ends reach it.
// A band's own rows arrive in the middle of its window, where its producer band writes them; the band fills the halo
// rows around them in place, and the unchanged compute core reads the window.

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

// The ports a band has: which neighbours it reads and which it writes for.
template<typename ConfigT, int B>
struct conv2d_halo_role {
  static constexpr bool TOP = B > 0 && ConfigT::HALO_TOP > 0;
  static constexpr bool BOTTOM = B + 1 < ConfigT::CAS_NUM && ConfigT::HALO_BOTTOM > 0;
  static constexpr bool FIRST = B > 0 && ConfigT::SEND_FIRST > 0;
  static constexpr bool LAST = B + 1 < ConfigT::CAS_NUM && ConfigT::SEND_LAST > 0;
  static constexpr int HALOS = TOP + BOTTOM, EDGES = FIRST + LAST;
};

template<typename ConfigT, int B>
class conv2d_halo_base {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;
  using role     = conv2d_halo_role<ConfigT, B>;

  conv2d_halo_base();

protected:
  // `halo`: its top then its bottom halo, as it has them; `edge`: its first then its last rows, as it sends them.
  void compute(const data_t* own, const data_t* const* halo, const weight_t (&wts)[ConfigT::WN],
               const bias_t (&bias)[ConfigT::BN], result_t* out, result_t* const* edge);

private:
  static constexpr bool READS = ConfigT::HALO_TOP + ConfigT::HALO_BOTTOM > 0;
};

// ADF reads a kernel's ports from its run() signature, so each count of halo inputs and sent edges is a class:
// own rows, [halos], weights, bias -> own rows, [edges].
template<typename ConfigT, int B, int HALOS = conv2d_halo_role<ConfigT, B>::HALOS,
         int EDGES = conv2d_halo_role<ConfigT, B>::EDGES>
class conv2d_halo;

// Port lists by count, which conv2d_halo.cpp defines each run() with.
#define CONV2D_HALO_IN_0
#define CONV2D_HALO_IN_1 , input_buffer<data_t>& h0
#define CONV2D_HALO_IN_2 , input_buffer<data_t>& h0, input_buffer<data_t>& h1
#define CONV2D_HALO_OUT_0
#define CONV2D_HALO_OUT_1 , output_buffer<result_t>& e0
#define CONV2D_HALO_OUT_2 , output_buffer<result_t>& e0, output_buffer<result_t>& e1
#define CONV2D_HALO_INS_0
#define CONV2D_HALO_INS_1 h0.data(),
#define CONV2D_HALO_INS_2 h0.data(), h1.data(),
#define CONV2D_HALO_OUTS_0
#define CONV2D_HALO_OUTS_1 e0.data(),
#define CONV2D_HALO_OUTS_2 e0.data(), e1.data(),
#define CONV2D_HALO_KERNEL(HALOS, EDGES)                                                                          \
  template<typename ConfigT, int B>                                                                               \
  class conv2d_halo<ConfigT, B, HALOS, EDGES> : public conv2d_halo_base<ConfigT, B> {                             \
  public:                                                                                                         \
    using typename conv2d_halo_base<ConfigT, B>::data_t;                                                          \
    using typename conv2d_halo_base<ConfigT, B>::weight_t;                                                        \
    using typename conv2d_halo_base<ConfigT, B>::result_t;                                                        \
    using typename conv2d_halo_base<ConfigT, B>::bias_t;                                                          \
    void run(input_buffer<data_t>& own CONV2D_HALO_IN_##HALOS, const weight_t (&wts)[ConfigT::WN],                \
             const bias_t (&bias)[ConfigT::BN], output_buffer<result_t>& ofm CONV2D_HALO_OUT_##EDGES);            \
    static void registerKernelClass() { REGISTER_FUNCTION(conv2d_halo::run); }                                    \
  };
CONV2D_HALO_KERNEL(0, 0)
CONV2D_HALO_KERNEL(0, 1)
CONV2D_HALO_KERNEL(0, 2)
CONV2D_HALO_KERNEL(1, 0)
CONV2D_HALO_KERNEL(1, 1)
CONV2D_HALO_KERNEL(1, 2)
CONV2D_HALO_KERNEL(2, 0)
CONV2D_HALO_KERNEL(2, 1)
CONV2D_HALO_KERNEL(2, 2)
#undef CONV2D_HALO_KERNEL
