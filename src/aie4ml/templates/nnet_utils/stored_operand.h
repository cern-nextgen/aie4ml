// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <aie_api/aie.hpp>

// An mmul operand read as its producer stores it. The kernel sees R x C microtiles of a VR x VC slice; the buffer
// holds that slice, or its transpose when TRANSPOSED, row-major in blocks of BLOCK_ROWS rows by one microtile's
// columns. A microtile stacks a microtile's rows of blocks, transposed in registers when TRANSPOSED.
// Kernel sources only: the graph front end compiles without the AIE API.
template<typename T, int R, int C, int VR, int VC, bool TRANSPOSED, int BLOCK_ROWS>
struct stored_operand {
  static constexpr int COLS = TRANSPOSED ? R : C;
  static constexpr int WIDTH = TRANSPOSED ? VR : VC;
  static constexpr int STACK = (TRANSPOSED ? C : R) / BLOCK_ROWS;
  static constexpr int OUTER_STEP = TRANSPOSED ? BLOCK_ROWS * R : R * VC;  // to the next microtile down the view
  static constexpr int INNER_STEP = TRANSPOSED ? C * VR : BLOCK_ROWS * C;  // to the next microtile across it
  static_assert(STACK * BLOCK_ROWS == (TRANSPOSED ? C : R), "BLOCK_ROWS must divide a microtile's stored rows");

  __attribute__((always_inline)) static aie::vector<T, R * C> load(const T* __restrict p) {
    aie::vector<T, R * C> v;
    if constexpr (STACK == 1) {
      v = aie::load_v<R * C>(p);
    } else {
      for (int s = 0; s < STACK; ++s)
        chess_flatten_loop
      {
        v.template insert<BLOCK_ROWS * COLS>(s, aie::load_v<BLOCK_ROWS * COLS>(p + s * BLOCK_ROWS * WIDTH));
      }
    }
    if constexpr (TRANSPOSED) return aie::transpose(v, C, R);
    else return v;
  }
};
