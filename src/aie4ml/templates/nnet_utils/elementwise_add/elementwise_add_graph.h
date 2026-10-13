#pragma once
#include <adf.h>
#include "elementwise_add.h"
#include "buffer_location.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class elementwise_add_graph : public graph {
public:
  static constexpr unsigned CAS_NUM = ConfigT::CAS_NUM;
  static constexpr unsigned TILE_ELEMENTS = ConfigT::TILE_ELEMENTS;

  input_port in1[CAS_NUM];
  input_port in2[CAS_NUM];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM];

  void place_graph(int COL_START, int ROW_START)
  {
    for (int row = 0; row < CAS_NUM; ++row)
    {
      const int tileCol = COL_START;
      const int tileRow = ROW_START + row;
      adf::location<adf::kernel>(kk[row]) = adf::tile(tileCol, tileRow);
      pin_buffer(kk[row].in[0], ConfigT::IN1_BUFFER_LOCATIONS[row], COL_START, ROW_START);
      adf::location<adf::stack>(kk[row]) = adf::bank(tileCol, tileRow, AIE4ML_STACK_BANK);
      pin_buffer(kk[row].in[1], ConfigT::IN2_BUFFER_LOCATIONS[row], COL_START, ROW_START);
      pin_buffer(kk[row].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[row], COL_START, ROW_START);
    }
  }

  elementwise_add_graph()
  {
    for (int row = 0; row < CAS_NUM; ++row) {
      kk[row] = kernel::create_object<elementwise_add_kernel<ConfigT>>();
      source(kk[row]) = "elementwise_add.cpp";
      runtime<ratio>(kk[row]) = 1.0;
    }

    for (int row = 0; row < CAS_NUM; ++row) {
      connect<>(in1[row], kk[row].in[0]);
      connect<>(in2[row], kk[row].in[1]);
      connect<>(kk[row].out[0], out1[row]);
      dimensions(kk[row].in[0]) = {TILE_ELEMENTS};
      dimensions(kk[row].in[1]) = {TILE_ELEMENTS};
      dimensions(kk[row].out[0]) = {TILE_ELEMENTS};
    }
  }
};
