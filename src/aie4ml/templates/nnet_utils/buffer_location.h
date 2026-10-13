#pragma once
#include <adf.h>

// Pins one buffer port where the op contract lists it (see BufferLocation), relative to the op's anchor: one
// buffer in one bank, or its ping and pong one per bank.
template<typename PortT, typename LocationT>
void pin_buffer(PortT& port, const LocationT& at, int COL_START, int ROW_START)
{
  const int col = COL_START + at.col;
  const int row = ROW_START + at.row;
  if (at.bank_count == 1) {
    adf::location<adf::buffer>(port) = adf::bank(col, row, at.bank0);
  } else {
    adf::location<adf::buffer>(port) = {adf::bank(col, row, at.bank0), adf::bank(col, row, at.bank1)};
  }
}
