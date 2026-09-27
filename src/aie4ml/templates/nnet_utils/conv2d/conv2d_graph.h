// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include "buffer_location.h"
#include "conv2d.h"
#include "conv2d_stream.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class conv2d_graph : public graph {
public:
  static constexpr unsigned CAS_NUM = ConfigT::CAS_NUM;
  static constexpr unsigned CAS_LENGTH = ConfigT::CAS_LENGTH;

  static constexpr bool OUTER = ConfigT::PARALLELISM_CONTRACT_OUTER;
  static constexpr unsigned IN_PORTS = OUTER ? CAS_NUM * CAS_LENGTH : CAS_LENGTH;

  input_port  in1[IN_PORTS];
  adf::port<adf::direction::in> wts[CAS_NUM * CAS_LENGTH];
  adf::port<adf::direction::in> bias[CAS_NUM];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH];

  static constexpr bool STREAM_IO = ConfigT::STREAM_IO;

  void place_graph(int COL_START, int ROW_START)
  {
    for (unsigned idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx) {
      const unsigned pos = idx % CAS_LENGTH;
      const unsigned chain = idx / CAS_LENGTH;
      const bool reverse = ConfigT::ALTERNATING_HORIZONTAL && ((ROW_START + chain) % 2 != 0);
      const int tileCol = COL_START + (reverse ? CAS_LENGTH - 1 - pos : pos);
      const int tileRow = ROW_START + chain;
      adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);
      if constexpr (!STREAM_IO) {
        // The Dense bank schedule: stack and bias in bank 1, weights in bank 2, and the activations
        // where IN1/OUT1_BUFFER_LOCATIONS put them -- one copy per bank.
        pin_buffer(kk[idx].in[0], ConfigT::IN1_BUFFER_LOCATIONS[idx], COL_START, ROW_START);
        adf::location<adf::stack>(kk[idx]) = adf::bank(tileCol, tileRow, 1);
        adf::location<adf::buffer>(kk[idx].in[1]) = adf::bank(tileCol, tileRow, 2);
        if (pos == 0) {  // the chain's first kernel seeds its accumulators with the bias
          adf::location<adf::buffer>(kk[idx].in[2]) = adf::bank(tileCol, tileRow, 1);
        }
        if (pos == CAS_LENGTH - 1) {
          pin_buffer(kk[idx].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[chain], COL_START, ROW_START);
        }
      }
    }
  }

  conv2d_graph( void )
  {
    for (unsigned chain = 0; chain < CAS_NUM; ++chain) {
      const unsigned base = chain * CAS_LENGTH;
      if constexpr (STREAM_IO) {
        kk[base] = kernel::create_object<conv2d_stream<ConfigT>>();
      } else if constexpr (CAS_LENGTH == 1) {
        kk[base] = kernel::create_object<conv2d_single<ConfigT>>();
      } else {
        kk[base] = kernel::create_object<conv2d_first<ConfigT>>();
        if constexpr (CAS_LENGTH > 2) {
          for (unsigned c = 1; c + 1 < CAS_LENGTH; ++c) {
            kk[base + c] = kernel::create_object<conv2d_middle<ConfigT>>();
          }
        }
        kk[base + CAS_LENGTH - 1] = kernel::create_object<conv2d_last<ConfigT>>();
      }
    }

    for (unsigned idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx) {
      const unsigned col = idx % CAS_LENGTH;
      const unsigned chain = idx / CAS_LENGTH;
      source(kk[idx]) = STREAM_IO ? "conv2d_stream.cpp" : "conv2d.cpp";
      runtime<ratio>(kk[idx]) = 1.0;
      single_buffer(kk[idx].in[1]);
      connect<parameter>(wts[idx], async(kk[idx].in[1]));
      connect<>(in1[OUTER ? idx : col], kk[idx].in[0]);
      if constexpr (!STREAM_IO) {
        dimensions(kk[idx].in[0]) = { ConfigT::IN_ELEMENTS };
      }
      if (col == 0) {
        connect<parameter>(bias[chain], async(kk[idx].in[2]));
        single_buffer(kk[idx].in[2]);
      }
      if (col == CAS_LENGTH - 1) {
        connect<>(kk[idx].out[0], out1[chain]);
        if constexpr (!STREAM_IO) {
          dimensions(kk[idx].out[0]) = { ConfigT::OUT_ELEMENTS };
        }
      }
    }

    if constexpr (CAS_LENGTH > 1) {
      for (unsigned chain = 0; chain < CAS_NUM; ++chain) {
        for (unsigned c = 0; c + 1 < CAS_LENGTH; ++c) {
          connect<cascade>(kk[chain * CAS_LENGTH + c].out[0], kk[chain * CAS_LENGTH + c + 1].in[2]);
        }
      }
    }
  }
};
