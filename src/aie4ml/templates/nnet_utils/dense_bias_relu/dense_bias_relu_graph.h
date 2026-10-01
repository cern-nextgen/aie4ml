// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <vector>
#include "dense_bias_relu.h"
#include "dense_bias_relu_stream.h"
#include "dense_vector.h"
#include "parameters.h"

using namespace adf;

// The buffer kernels by row microtiles per step: 2 (dense_bias_relu.cpp) or 1 (dense_vector.cpp).
template<typename ConfigT, int ROW_BLOCKS>
struct dense_buffer_kernels {
  using single = dense_single<ConfigT>;
  using first = dense_first<ConfigT>;
  using middle = dense_middle<ConfigT>;
  using last = dense_last<ConfigT>;
  static constexpr const char* source = "dense_bias_relu.cpp";
};

template<typename ConfigT>
struct dense_buffer_kernels<ConfigT, 1> {
  using single = dense_vector_single<ConfigT>;
  using first = dense_vector_first<ConfigT>;
  using middle = dense_vector_middle<ConfigT>;
  using last = dense_vector_last<ConfigT>;
  static constexpr const char* source = "dense_vector.cpp";
};

template<typename ConfigT>
class dense_bias_relu_graph : public graph {
public:
  static constexpr unsigned IN_FEAT  = ConfigT::IN_FEAT;
  static constexpr unsigned OUT_FEAT = ConfigT::OUT_FEAT;
  static constexpr unsigned CAS_NUM  = ConfigT::CAS_NUM;
  static constexpr unsigned CAS_LENGTH = ConfigT::CAS_LENGTH;
  static constexpr unsigned IN_FEAT_SLICE  = ConfigT::IN_FEAT_SLICE;
  static constexpr unsigned OUT_FEAT_SLICE = ConfigT::OUT_FEAT_SLICE;
  static constexpr unsigned padded_independent_extent = ConfigT::padded_independent_extent;
  static constexpr int M  = ConfigT::M;
  static constexpr int K  = ConfigT::K;
  static constexpr int N  = ConfigT::N;

  // 'inner' multicasts one LHS slice per column to every chain; 'outer' gives each
  // (chain, column) tile its own row slice, so the port array is per-tile.
  static constexpr bool PARALLELISM_CONTRACT_OUTER = ConfigT::PARALLELISM_CONTRACT_OUTER;
  static constexpr unsigned LHS_PORTS = PARALLELISM_CONTRACT_OUTER ? CAS_NUM * CAS_LENGTH : CAS_LENGTH;
  // The LHS arrives, and the output leaves, on a stream (in row order) or in a buffer (in microtiles).
  static constexpr bool STREAM_IN = ConfigT::STREAM_IN;
  static constexpr bool STREAM_OUT = ConfigT::STREAM_OUT;
  using BufferKernels = dense_buffer_kernels<ConfigT, ConfigT::ROW_BLOCKS>;

  input_port  in1[LHS_PORTS];
  adf::port<adf::direction::in> wts[CAS_NUM * CAS_LENGTH];
  adf::port<adf::direction::in> bias[CAS_NUM];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH]; // row wise

void place_graph(int COL_START, int ROW_START)
{
  for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx)
  {
    const int pos = idx % CAS_LENGTH;
    const int chain = idx / CAS_LENGTH;
    const bool is_last = pos == CAS_LENGTH - 1;
    const bool reverse = ConfigT::ALTERNATING_HORIZONTAL && ((ROW_START + chain) % 2 != 0);
    const int tileCol = COL_START + (reverse ? CAS_LENGTH - 1 - pos : pos);
    const int tileRow = ROW_START + chain;

    adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);

    if constexpr (!STREAM_IN) {
      const auto inputLocation = ConfigT::IN1_BUFFER_LOCATIONS[idx];
      if (inputLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[idx].in[0]) = adf::bank(
          COL_START + inputLocation.col, ROW_START + inputLocation.row, inputLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[idx].in[0]) = {
          adf::bank(COL_START + inputLocation.col, ROW_START + inputLocation.row, inputLocation.bank0),
          adf::bank(COL_START + inputLocation.col, ROW_START + inputLocation.row, inputLocation.bank1)
        };
      }
    }

    adf::location<adf::stack>(kk[idx]) = adf::bank(tileCol, tileRow, 1);
    adf::location<adf::buffer>(kk[idx].in[1]) = adf::bank(tileCol, tileRow, 2);
    if (pos == 0) {  // the chain's first kernel seeds its accumulators with the bias
      adf::location<adf::buffer>(kk[idx].in[2]) = adf::bank(tileCol, tileRow, 1);
    }

    if (is_last) {
      if constexpr (!STREAM_OUT) {
        const auto outputLocation = ConfigT::OUT1_BUFFER_LOCATIONS[idx / CAS_LENGTH];
        if (outputLocation.bank_count == 1) {
          adf::location<adf::buffer>(kk[idx].out[0]) = adf::bank(
            COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank0);
        } else {
          adf::location<adf::buffer>(kk[idx].out[0]) = {
            adf::bank(COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank0),
            adf::bank(COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank1)
          };
        }
      }
    }
  }
}

  dense_bias_relu_graph( void )
  {

    // A stream end at either side runs dense_bias_relu_stream.cpp; a chain's first and middle kernels see only its
    // input. Every create_object stays braced and guarded: the graph front end instantiates each one it sees.
    for (int chain = 0; chain < CAS_NUM; ++chain) {
        const int last = chain * CAS_LENGTH + (CAS_LENGTH - 1);
        if constexpr (CAS_LENGTH == 1) {
            if constexpr (STREAM_IN && STREAM_OUT) {
                kk[last] = kernel::create_object<dense_single_stream<ConfigT>>();
            } else if constexpr (STREAM_IN) {
                kk[last] = kernel::create_object<dense_single_stream_in<ConfigT>>();
            } else if constexpr (STREAM_OUT) {
                kk[last] = kernel::create_object<dense_single_stream_out<ConfigT>>();
            } else {
                kk[last] = kernel::create_object<typename BufferKernels::single>();
            }
        } else {
            if constexpr (STREAM_IN) {
                kk[chain * CAS_LENGTH + 0] = kernel::create_object<dense_first_stream<ConfigT>>();
                if constexpr (CAS_LENGTH > 2) {
                    for (int c = 1; c < CAS_LENGTH - 1; ++c) {
                        kk[chain * CAS_LENGTH + c] = kernel::create_object<dense_middle_stream<ConfigT>>();
                    }
                }
            } else {
                kk[chain * CAS_LENGTH + 0] = kernel::create_object<typename BufferKernels::first>();
                if constexpr (CAS_LENGTH > 2) {
                    for (int c = 1; c < CAS_LENGTH - 1; ++c) {
                        kk[chain * CAS_LENGTH + c] = kernel::create_object<typename BufferKernels::middle>();
                    }
                }
            }
            if constexpr (STREAM_IN && STREAM_OUT) {
                kk[last] = kernel::create_object<dense_last_stream<ConfigT>>();
            } else if constexpr (STREAM_IN) {
                kk[last] = kernel::create_object<dense_last_stream_in<ConfigT>>();
            } else if constexpr (STREAM_OUT) {
                kk[last] = kernel::create_object<dense_last_stream_out<ConfigT>>();
            } else {
                kk[last] = kernel::create_object<typename BufferKernels::last>();
            }
        }
    }

    for (int idx = 0; idx < CAS_LENGTH * CAS_NUM; ++idx) {
        int col = idx % CAS_LENGTH;
        int row = idx / CAS_LENGTH;
        const bool stream_core = STREAM_IN || (STREAM_OUT && col == CAS_LENGTH - 1);
        source(kk[idx])        = stream_core ? "dense_bias_relu_stream.cpp" : BufferKernels::source;
        runtime<ratio>(kk[idx]) = 1.0;
        single_buffer(kk[idx].in[1]);
        connect<parameter>(wts[idx], async(kk[idx].in[1]));
        if (col == 0) {
          connect<parameter>(bias[row], async(kk[idx].in[2]));
          single_buffer(kk[idx].in[2]);
        }

    }

    for (unsigned col = 0; col < CAS_LENGTH; ++col) {
      for (unsigned ch = 0; ch < CAS_NUM; ++ch) {
        int idx = ch*CAS_LENGTH + col;
        connect<>( in1[PARALLELISM_CONTRACT_OUTER ? idx : col], kk[idx].in[0] );
        if constexpr (!STREAM_IN) {
          dimensions( kk[idx].in[0] ) = { padded_independent_extent * IN_FEAT_SLICE };
        }
      }
    }

    for (int chain = 0; chain < CAS_NUM; ++chain) {
        const int last_idx = chain * CAS_LENGTH + (CAS_LENGTH - 1);
        connect<>( kk[last_idx].out[0], out1[chain] );
        if constexpr (!STREAM_OUT) {
          dimensions( kk[last_idx].out[0] ) = { padded_independent_extent * OUT_FEAT_SLICE };
        }
    }

    if constexpr (CAS_LENGTH > 1) {
      for (int chain = 0; chain < CAS_NUM; ++chain) {
        for (int c = 0; c < CAS_LENGTH - 1; ++c) {
          connect<cascade>(
            kk[chain * CAS_LENGTH + c].out[0],
            kk[chain * CAS_LENGTH + c + 1].in[2]
          );
        }
      }
    }

  }

};
