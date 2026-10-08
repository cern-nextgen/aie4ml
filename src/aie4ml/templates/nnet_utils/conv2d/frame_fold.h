// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Builds a folded conv's input frame [CB][ROWS][COLS][8]: per output pixel its whole window, unit n = (ky, kx) at
// byte n * UNIT, read from the input window in rows of UNIT-byte pixels, the border zero.
// ConfigT declares:
//   data_t
//   UNIT                  bytes per source pixel: 1, 2 or 4
//   KH, KW, STRIDE_H, STRIDE_W   the conv's window
//   SRC_COLS, SRC_BYTES   source pixels per row (rows on a 32-byte boundary), and bytes per inference: its rows and
//                         the slack loads read past them
//   ROWS, COLS, CB        the frame this kernel writes: output rows, columns, and its channel blocks
//   FIRST_BLOCK           the first of the conv's blocks this kernel writes
//   FRAME_BYTES           = CB * ROWS * COLS * 8

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class frame_fold {
public:
  using data_t = typename ConfigT::data_t;

  frame_fold();
  void run(input_buffer<data_t>& src, output_buffer<data_t>& frame);

  static void registerKernelClass() { REGISTER_FUNCTION(frame_fold::run); }
};
