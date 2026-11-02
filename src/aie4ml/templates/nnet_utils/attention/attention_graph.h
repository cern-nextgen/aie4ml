// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// One head's fused attention (attention.h): CAS_NUM chains of CAS_LENGTH tiles, chain c on row c, west to east (east
// to west on AIE odd rows). A chain of one tile does it all; a chain of two is a tile that scores the keys and one that
// weighs the values and divides.
// Ports: in1 a chain's queries and in2 every key, to its first tile; in3 every value, to its last tile; out1 a chain's
// O, from its last tile. The exp and reciprocal tables (ConfigT) are compiled into each kernel.
#pragma once
#include <adf.h>
#include <vector>
#include "attention.h"
#include "buffer_location.h"
#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class attention_graph : public graph {
public:
  static constexpr int CAS_NUM = ConfigT::CAS_NUM;
  static constexpr int CAS_LENGTH = ConfigT::CAS_LENGTH;
  static_assert(CAS_LENGTH == 1 || CAS_LENGTH == 2, "a chain is one tile, or a key tile and a dividing one");
  static constexpr int NJ = ConfigT::HEAD_DIM / 8;
  static constexpr int Q_BYTES = ConfigT::QUERY_BLOCKS * NJ * 64;
  static constexpr int KV_BYTES = ConfigT::KEY_BLOCKS * NJ * 64;
  static constexpr int V_PORT = CAS_LENGTH == 1 ? 2 : 1;  // the last tile's input of the values

  input_port in1[CAS_NUM];
  input_port in2[1];
  input_port in3[1];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH];

  void place_graph(int COL_START, int ROW_START)
  {
    for (int c = 0; c < CAS_NUM; ++c) {
      const int row = ROW_START + c;
      const bool reverse = ConfigT::ALTERNATING_HORIZONTAL && (row % 2 != 0);
      for (int p = 0; p < CAS_LENGTH; ++p) {
        const int idx = c * CAS_LENGTH + p;
        const int col = COL_START + (reverse ? CAS_LENGTH - 1 - p : p);
        adf::location<adf::kernel>(kk[idx]) = adf::tile(col, row);
        adf::location<adf::stack>(kk[idx]) = adf::bank(col, row, AIE4ML_STACK_BANK);
        for (int t = 0; t < (CAS_LENGTH == 1 ? 4 : 2); ++t)  // each table's two copies in two banks
          adf::location<adf::parameter>(kk[idx].param[t]) =
              adf::bank(col, row, t % 2 ? AIE4ML_WEIGHT_BANK : AIE4ML_STACK_BANK);
        if (p == 0) {
          pin_buffer(kk[idx].in[0], ConfigT::IN1_BUFFER_LOCATIONS[c], COL_START, ROW_START);
          pin_buffer(kk[idx].in[1], ConfigT::IN2_BUFFER_LOCATIONS[c], COL_START, ROW_START);
        }
        if (p + 1 == CAS_LENGTH) {
          pin_buffer(kk[idx].in[V_PORT], ConfigT::IN3_BUFFER_LOCATIONS[c], COL_START, ROW_START);
          pin_buffer(kk[idx].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[c], COL_START, ROW_START);
        }
      }
    }
  }

  attention_graph()
  {
    for (int c = 0; c < CAS_NUM; ++c) {
      const int first = c * CAS_LENGTH, last = first + CAS_LENGTH - 1;
      for (int p = 0; p < CAS_LENGTH; ++p) {
        const int idx = first + p;
        // every create_object stays braced and guarded: the graph front end appends a statement to each
        if constexpr (CAS_LENGTH == 1) {
          kk[idx] = kernel::create_object<attention_single<ConfigT>>(table(ConfigT::EXP_TABLE), table(ConfigT::EXP_TABLE),
                                                                     table(ConfigT::RECIP_TABLE), table(ConfigT::RECIP_TABLE));
        } else {
          if (p == 0) {
            kk[idx] = kernel::create_object<attention_scores<ConfigT>>(table(ConfigT::EXP_TABLE), table(ConfigT::EXP_TABLE));
          } else {
            kk[idx] = kernel::create_object<attention_values<ConfigT>>(table(ConfigT::RECIP_TABLE),
                                                                       table(ConfigT::RECIP_TABLE));
          }
        }
        source(kk[idx]) = "attention.cpp";
        runtime<ratio>(kk[idx]) = 1.0;
        stack_size(kk[idx]) = ConfigT::STACK_BYTES;
      }
      connect<>(in1[c], kk[first].in[0]);
      dimensions(kk[first].in[0]) = {Q_BYTES};
      connect<>(in2[0], kk[first].in[1]);
      dimensions(kk[first].in[1]) = {KV_BYTES};
      connect<>(in3[0], kk[last].in[V_PORT]);
      dimensions(kk[last].in[V_PORT]) = {KV_BYTES};
      if (CAS_LENGTH == 2) connect<cascade>(kk[first].out[0], kk[last].in[0]);
      connect<>(kk[last].out[0], out1[c]);
      dimensions(kk[last].out[0]) = {Q_BYTES};
    }
  }

private:
  static std::vector<int16> table(const int16 (&t)[ConfigT::TABLE_SIZE])
  {
    return std::vector<int16>(t, t + ConfigT::TABLE_SIZE);
  }
};
