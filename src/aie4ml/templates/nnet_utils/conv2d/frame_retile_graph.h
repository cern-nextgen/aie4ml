// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <utility>
#include "buffer_location.h"
#include "frame_fold.h"
#include "frame_retile.h"
#include "parameters.h"

using namespace adf;

// A retiler kernel per window -- CAS_NUM of them, no cascade -- window `w` on row `w`, on the Dense bank contract: its input and frame
// where IN1/OUT1_BUFFER_LOCATIONS put them -- one copy per bank -- and its stack in bank 1. Each
// window's kernel is configured by `ConfigT::window<w>::type`; FOLD picks the kernel that folds the frame.
template<typename ConfigT>
class frame_retile_graph : public graph {
public:
  input_port in1[ConfigT::CAS_NUM];
  output_port out1[ConfigT::CAS_NUM];
  kernel kk[ConfigT::CAS_NUM];

  void place_graph(int COL_START, int ROW_START)
  {
    for (int w = 0; w < ConfigT::CAS_NUM; ++w) {
      adf::location<adf::kernel>(kk[w]) = adf::tile(COL_START, ROW_START + w);
      adf::location<adf::stack>(kk[w]) = adf::bank(COL_START, ROW_START + w, 1);
      pin_buffer(kk[w].in[0], ConfigT::IN1_BUFFER_LOCATIONS[w], COL_START, ROW_START);
      pin_buffer(kk[w].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[w], COL_START, ROW_START);
    }
  }

  frame_retile_graph( void ) { build(std::make_integer_sequence<int, ConfigT::CAS_NUM>{}); }

private:
  template<int... W>
  void build(std::integer_sequence<int, W...>) { (build_window<W>(), ...); }

  template<int W>
  void build_window()
  {
    using WindowT = typename ConfigT::template window<W>::type;
    if constexpr (ConfigT::FOLD) {
      kk[W] = kernel::create_object<frame_fold<WindowT>>();
      source(kk[W]) = "frame_fold.cpp";
    } else {
      kk[W] = kernel::create_object<frame_retile<WindowT>>();
      source(kk[W]) = "frame_retile.cpp";
    }
    runtime<ratio>(kk[W]) = 1.0;
    dimensions(kk[W].in[0]) = { WindowT::SRC_BYTES };
    dimensions(kk[W].out[0]) = { ConfigT::FRAME_BYTES };
    connect<>(in1[W], kk[W].in[0]);
    connect<>(kk[W].out[0], out1[W]);
  }
};
