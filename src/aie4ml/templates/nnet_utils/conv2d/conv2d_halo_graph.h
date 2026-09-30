// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <utility>
#include "buffer_location.h"
#include "conv2d_halo.h"
#include "parameters.h"

using namespace adf;

// Conv2D row bands handing their neighbours the rows their windows read: band b on row b, one tile each. A port
// array holds the bands' own rows, then per pair of neighbours (b, b + 1) band b's last rows (band b + 1's top halo)
// and band b + 1's first rows (band b's bottom halo) -- in1 the rows this op's window reads, out1 the rows its
// consumer's does (halo_ports in common.py). Every port has one kernel on each end, pinned where the op contract
// lists it, so each is one buffer.
template<typename ConfigT>
class conv2d_halo_graph : public graph {
public:
  static constexpr int BANDS = ConfigT::CAS_NUM;
  static_assert(BANDS >= 2 && ConfigT::CAS_LENGTH == 1, "halo bands: two or more single-tile row bands");
  static constexpr int HT = ConfigT::HALO_TOP, HB = ConfigT::HALO_BOTTOM;
  static constexpr int SF = ConfigT::SEND_FIRST, SL = ConfigT::SEND_LAST;
  static constexpr int IN_PAIR = (HT > 0) + (HB > 0), OUT_PAIR = (SL > 0) + (SF > 0);
  static constexpr int IN_PORTS = BANDS + (BANDS - 1) * IN_PAIR, OUT_PORTS = BANDS + (BANDS - 1) * OUT_PAIR;

  // The ports of band b's top and bottom halo, and of the first and last rows it sends.
  static constexpr int top_port(int b) { return BANDS + (b - 1) * IN_PAIR; }
  static constexpr int bottom_port(int b) { return BANDS + b * IN_PAIR + (HT > 0); }
  static constexpr int first_port(int b) { return BANDS + (b - 1) * OUT_PAIR + (SL > 0); }
  static constexpr int last_port(int b) { return BANDS + b * OUT_PAIR; }

  // Elements of the rows of one channel-blocked frame row, times rows.
  static constexpr int IN_ROW = ConfigT::IN_ELEMENTS / ConfigT::IN_ROWS;
  static constexpr int OUT_ROW = ConfigT::OUT_ELEMENTS / ConfigT::OUT_ROWS;

  input_port in1[IN_PORTS];
  adf::port<adf::direction::in> wts[BANDS];
  adf::port<adf::direction::in> bias[BANDS];
  output_port out1[OUT_PORTS];
  kernel kk[BANDS];

  void place_graph(int COL_START, int ROW_START) { place(COL_START, ROW_START, std::make_integer_sequence<int, BANDS>{}); }

  conv2d_halo_graph( void ) { build(std::make_integer_sequence<int, BANDS>{}); }

private:
  template<int... B>
  void build(std::integer_sequence<int, B...>) { (build_band<B>(), ...); }

  template<int... B>
  void place(int COL_START, int ROW_START, std::integer_sequence<int, B...>) { (place_band<B>(COL_START, ROW_START), ...); }

  template<int B>
  void build_band()
  {
    using role = conv2d_halo_role<ConfigT, B>;
    constexpr int WTS = 1 + role::TOP + role::BOTTOM;
    kk[B] = kernel::create_object<conv2d_halo<ConfigT, B>>();
    source(kk[B]) = "conv2d_halo.cpp";
    runtime<ratio>(kk[B]) = 1.0;
    connect<>(in1[B], kk[B].in[0]);
    dimensions(kk[B].in[0]) = { ConfigT::IN_ELEMENTS };
    if constexpr (role::TOP) {
      connect<>(in1[top_port(B)], kk[B].in[1]);
      dimensions(kk[B].in[1]) = { HT * IN_ROW };
    }
    if constexpr (role::BOTTOM) {
      connect<>(in1[bottom_port(B)], kk[B].in[1 + role::TOP]);
      dimensions(kk[B].in[1 + role::TOP]) = { HB * IN_ROW };
    }
    single_buffer(kk[B].in[WTS]);
    connect<parameter>(wts[B], async(kk[B].in[WTS]));
    single_buffer(kk[B].in[WTS + 1]);
    connect<parameter>(bias[B], async(kk[B].in[WTS + 1]));
    connect<>(kk[B].out[0], out1[B]);
    dimensions(kk[B].out[0]) = { ConfigT::OUT_ELEMENTS };
    if constexpr (role::FIRST) {
      connect<>(kk[B].out[1], out1[first_port(B)]);
      dimensions(kk[B].out[1]) = { SF * OUT_ROW };
    }
    if constexpr (role::LAST) {
      connect<>(kk[B].out[1 + role::FIRST], out1[last_port(B)]);
      dimensions(kk[B].out[1 + role::FIRST]) = { SL * OUT_ROW };
    }
  }

  template<int B>
  void place_band(int COL_START, int ROW_START)
  {
    using role = conv2d_halo_role<ConfigT, B>;
    constexpr int WTS = 1 + role::TOP + role::BOTTOM;
    adf::location<adf::kernel>(kk[B]) = adf::tile(COL_START, ROW_START + B);
    adf::location<adf::stack>(kk[B]) = adf::bank(COL_START, ROW_START + B, 1);
    adf::location<adf::buffer>(kk[B].in[WTS]) = adf::bank(COL_START, ROW_START + B, 2);
    adf::location<adf::buffer>(kk[B].in[WTS + 1]) = adf::bank(COL_START, ROW_START + B, 1);
    pin_buffer(kk[B].in[0], ConfigT::IN1_BUFFER_LOCATIONS[B], COL_START, ROW_START);
    if constexpr (role::TOP) pin_buffer(kk[B].in[1], ConfigT::IN1_BUFFER_LOCATIONS[top_port(B)], COL_START, ROW_START);
    if constexpr (role::BOTTOM)
      pin_buffer(kk[B].in[1 + role::TOP], ConfigT::IN1_BUFFER_LOCATIONS[bottom_port(B)], COL_START, ROW_START);
    pin_buffer(kk[B].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[B], COL_START, ROW_START);
    if constexpr (role::FIRST)
      pin_buffer(kk[B].out[1], ConfigT::OUT1_BUFFER_LOCATIONS[first_port(B)], COL_START, ROW_START);
    if constexpr (role::LAST)
      pin_buffer(kk[B].out[1 + role::FIRST], ConfigT::OUT1_BUFFER_LOCATIONS[last_port(B)], COL_START, ROW_START);
  }
};
