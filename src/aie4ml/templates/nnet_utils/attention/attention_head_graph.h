// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// One fused attention head (attention_head.h): CAS_NUM chains of CAS_LENGTH tiles, placed as attention_graph.h.
// Ports: in1 x, the two halves of its columns for the chains' first tiles, then for their last (a broadcast each,
// one input DMA channel each); out1 a chain's O, from its last tile. The tables, weights and biases (ConfigT) are
// compiled into each kernel.
#pragma once
#include <adf.h>
#include <vector>
#include "attention_head.h"
#include "buffer_location.h"
#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class attention_head_graph : public graph {
public:
  static constexpr int CAS_NUM = ConfigT::CAS_NUM;
  static constexpr int CAS_LENGTH = ConfigT::CAS_LENGTH;
  static_assert(CAS_LENGTH == 1 || CAS_LENGTH == 2, "a chain is one tile, or a key tile and a dividing one");
  static constexpr int NJ = ConfigT::HEAD_DIM / 8;
  static constexpr int X_HALVES = 2;
  static constexpr int X_BYTES = ConfigT::KEY_BLOCKS * ConfigT::MODEL_DIM / X_HALVES * 8;
  static constexpr int O_BYTES = ConfigT::QUERY_BLOCKS * ConfigT::OUT_DIM * 8;

  input_port in1[CAS_LENGTH * X_HALVES];
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
        const int tables = CAS_LENGTH == 1 || p == 1 ? 4 : 2, projected = CAS_LENGTH == 1 ? 6 : (p == 0 ? 4 : 2);
        for (int t = 0; t < tables; ++t)  // each table's two copies in two banks
          adf::location<adf::parameter>(kk[idx].param[t]) =
              adf::bank(col, row, t % 2 ? AIE4ML_WEIGHT_BANK : AIE4ML_STACK_BANK);
        for (int t = tables; t < tables + projected; ++t)  // the weights and biases beside them
          adf::location<adf::parameter>(kk[idx].param[t]) = adf::bank(col, row, AIE4ML_WEIGHT_BANK);
        for (int h = 0; h < X_HALVES; ++h)
          pin_buffer(kk[idx].in[x_port(p) + h], ConfigT::IN1_BUFFER_LOCATIONS[X_HALVES * idx + h], COL_START,
                     ROW_START);
        if (p + 1 == CAS_LENGTH) pin_buffer(kk[idx].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[c], COL_START, ROW_START);
      }
    }
  }

  attention_head_graph()
  {
    for (int c = 0; c < CAS_NUM; ++c) {
      const int first = c * CAS_LENGTH, last = first + CAS_LENGTH - 1;
      const int block = c * ConfigT::QUERY_BLOCKS;  // the chain's first query block
      for (int p = 0; p < CAS_LENGTH; ++p) {
        const int idx = first + p;
        // every create_object stays braced and guarded: the graph front end appends a statement to each
        if constexpr (CAS_LENGTH == 1) {
          kk[idx] = kernel::create_object<attention_head_single<ConfigT>>(
              table(ConfigT::EXP_TABLE), table(ConfigT::EXP_TABLE), table(ConfigT::RECIP_TABLE),
              table(ConfigT::RECIP_TABLE), weights(ConfigT::WQ), weights(ConfigT::WK), weights(ConfigT::WV),
              bias(ConfigT::BQ), bias(ConfigT::BK), bias(ConfigT::BV), block);
        } else {
          if (p == 0) {
            kk[idx] = kernel::create_object<attention_head_scores<ConfigT>>(
                table(ConfigT::EXP_TABLE), table(ConfigT::EXP_TABLE), weights(ConfigT::WQ), weights(ConfigT::WK),
                bias(ConfigT::BQ), bias(ConfigT::BK), block);
          } else {
            kk[idx] = kernel::create_object<attention_head_values<ConfigT>>(
                table(ConfigT::EXP_TABLE), table(ConfigT::EXP_TABLE), table(ConfigT::RECIP_TABLE),
                table(ConfigT::RECIP_TABLE), weights(ConfigT::WV), bias(ConfigT::BV));
          }
        }
        source(kk[idx]) = "attention_head.cpp";
        runtime<ratio>(kk[idx]) = 1.0;
        stack_size(kk[idx]) = ConfigT::STACK_BYTES;
        for (int h = 0; h < X_HALVES; ++h) {
          connect<>(in1[X_HALVES * p + h], kk[idx].in[x_port(p) + h]);
          dimensions(kk[idx].in[x_port(p) + h]) = {X_BYTES};
        }
      }
      if (CAS_LENGTH == 2) connect<cascade>(kk[first].out[0], kk[last].in[0]);
      connect<>(kk[last].out[0], out1[c]);
      dimensions(kk[last].out[0]) = {O_BYTES};
    }
  }

private:
  // the port of x's first half: after the values' cascade
  static constexpr int x_port(int p) { return CAS_LENGTH == 2 && p == 1 ? 1 : 0; }

  static std::vector<int16> table(const int16 (&t)[ConfigT::TABLE_SIZE])
  {
    return std::vector<int16>(t, t + ConfigT::TABLE_SIZE);
  }
  static std::vector<int8> weights(const int8 (&w)[ConfigT::MODEL_DIM * ConfigT::HEAD_DIM])
  {
    return std::vector<int8>(w, w + ConfigT::MODEL_DIM * ConfigT::HEAD_DIM);
  }
  static std::vector<int32> bias(const int32 (&b)[ConfigT::HEAD_DIM])
  {
    return std::vector<int32>(b, b + ConfigT::HEAD_DIM);
  }
};
