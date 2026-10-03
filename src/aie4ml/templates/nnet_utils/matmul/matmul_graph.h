#pragma once
#include <adf.h>
#include <cstdint>
#include "matmul.h"
#include "matmul_vector.h"
#include "parameters.h"

using namespace adf;

// The kernels by row microtiles per step: 2 (matmul.cpp) or 1 (matmul_vector.cpp).
template<typename ConfigT, int ROW_BLOCKS>
struct matmul_kernels {
  using single = matmul_single<ConfigT>;
  using first = matmul_first<ConfigT>;
  using middle = matmul_middle<ConfigT>;
  using last = matmul_last<ConfigT>;
  static constexpr const char* source = "matmul.cpp";
};

template<typename ConfigT>
struct matmul_kernels<ConfigT, 1> {
  using single = matmul_vector_single<ConfigT>;
  using first = matmul_vector_first<ConfigT>;
  using middle = matmul_vector_middle<ConfigT>;
  using last = matmul_vector_last<ConfigT>;
  static constexpr const char* source = "matmul_vector.cpp";
};

template<typename ConfigT>
class matmul_graph : public graph {
public:
  static constexpr unsigned CAS_NUM    = ConfigT::CAS_NUM;
  static constexpr unsigned CAS_LENGTH = ConfigT::CAS_LENGTH;
  static constexpr unsigned padded_M   = ConfigT::padded_M;
  static constexpr unsigned K_SLICE    = ConfigT::K_SLICE;
  static constexpr unsigned N_SLICE    = ConfigT::N_SLICE;

  // 'outer' splits the rows across the chains, so each tile reads its own lhs slice rather
  // than sharing one per cascade column. padded_M is already the per-tile row extent.
  static constexpr bool PARALLELISM_CONTRACT_OUTER = ConfigT::PARALLELISM_CONTRACT_OUTER;
  static constexpr unsigned LHS_PORTS = PARALLELISM_CONTRACT_OUTER ? CAS_NUM * CAS_LENGTH : CAS_LENGTH;

  using a_t = typename ConfigT::a_t;
  using b_t = typename ConfigT::b_t;
  using c_t = typename ConfigT::c_t;
  using Kernels = matmul_kernels<ConfigT, ConfigT::ROW_BLOCKS>;

  static constexpr std::uint32_t BANK_BYTES        = 16 * 1024;
  static constexpr std::uint32_t STACK_BYTES       = 1024; // reserve 1 KiB for compiler stack
  static constexpr std::uint32_t STACK_BYTES_ALIGN = 1024; // keep B tile starts simple and bank-aligned enough

  static constexpr std::uint32_t A_TILE_BYTES = padded_M * K_SLICE * sizeof(a_t);
  static constexpr std::uint32_t B_TILE_BYTES = K_SLICE * N_SLICE * sizeof(b_t);
  static constexpr std::uint32_t C_TILE_BYTES = padded_M * N_SLICE * sizeof(c_t);

  static_assert(A_TILE_BYTES <= BANK_BYTES,
                "One A ping/pong tile must fit in one 16 KiB bank");
  static_assert(B_TILE_BYTES + STACK_BYTES_ALIGN <= BANK_BYTES,
                "B tile plus reserved stack slack must fit in one 16 KiB bank");
  static_assert(C_TILE_BYTES <= BANK_BYTES,
                "One C ping/pong tile must fit in one 16 KiB bank");

  input_port inA[LHS_PORTS];
  input_port inB[CAS_NUM * CAS_LENGTH];
  output_port outC[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH];

  static constexpr std::uint32_t bank_base(int bank) {
    return static_cast<std::uint32_t>(bank) * BANK_BYTES;
  }

  void place_graph(int COL_START, int ROW_START)
  {
    for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx)
    {
      const int pos = idx % CAS_LENGTH;
      const int chain = idx / CAS_LENGTH;
      const bool is_last = pos == CAS_LENGTH - 1;
      const bool reverse = ConfigT::ALTERNATING_HORIZONTAL && ((ROW_START + chain) % 2 != 0);
      const auto lhsLocation = ConfigT::INA_BUFFER_LOCATIONS[idx];
      const auto rhsLocation = ConfigT::INB_BUFFER_LOCATIONS[idx];
      const int tileCol = COL_START + (reverse ? CAS_LENGTH - 1 - pos : pos);
      const int tileRow = ROW_START + chain;

      adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);

      if (lhsLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[idx].in[0]) = adf::bank(
          COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[idx].in[0]) = {
          adf::bank(COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank0),
          adf::bank(COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank1)
        };
      }

      if (rhsLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[idx].in[1]) = adf::bank(
          COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[idx].in[1]) = {
          adf::bank(COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank0),
          adf::bank(COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank1)
        };
      }

      // Stack shares one B bank, but let compiler choose the offset.
      adf::location<adf::stack>(kk[idx]) = adf::bank(tileCol, tileRow, 1);

      if (is_last) {
        const auto outputLocation = ConfigT::OUTC_BUFFER_LOCATIONS[idx / CAS_LENGTH];
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

  matmul_graph() {
    for (int row = 0; row < CAS_NUM; ++row) {
      if constexpr (CAS_LENGTH == 1) {
        kk[row * CAS_LENGTH] = kernel::create_object<typename Kernels::single>();
      } else {
        kk[row * CAS_LENGTH] = kernel::create_object<typename Kernels::first>();
        if constexpr (CAS_LENGTH > 2) {
          for (int c = 1; c < CAS_LENGTH - 1; ++c) {
            kk[row * CAS_LENGTH + c] = kernel::create_object<typename Kernels::middle>();
          }
        }
        kk[row * CAS_LENGTH + (CAS_LENGTH - 1)] = kernel::create_object<typename Kernels::last>();
      }
    }

    for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx) {
      source(kk[idx]) = Kernels::source;
      runtime<ratio>(kk[idx]) = 1.0;
    }

    for (unsigned col = 0; col < CAS_LENGTH; ++col) {
      for (unsigned row = 0; row < CAS_NUM; ++row) {
        const int idx = row * CAS_LENGTH + col;
        connect<>(inA[PARALLELISM_CONTRACT_OUTER ? idx : col], kk[idx].in[0]);
        dimensions(kk[idx].in[0]) = {padded_M * K_SLICE};
      }
    }

    for (unsigned row = 0; row < CAS_NUM; ++row) {
      for (unsigned col = 0; col < CAS_LENGTH; ++col) {
        const int idx = row * CAS_LENGTH + col;
        connect<>(inB[idx], kk[idx].in[1]);
        dimensions(kk[idx].in[1]) = {K_SLICE * N_SLICE};
      }
    }

    for (unsigned row = 0; row < CAS_NUM; ++row) {
      const int last_idx = row * CAS_LENGTH + (CAS_LENGTH - 1);
      connect<>(kk[last_idx].out[0], outC[row]);
      dimensions(kk[last_idx].out[0]) = {padded_M * N_SLICE};
    }

    if constexpr (CAS_LENGTH > 1) {
      for (unsigned row = 0; row < CAS_NUM; ++row) {
        for (unsigned col = 0; col < CAS_LENGTH - 1; ++col) {
          connect<cascade>(kk[row * CAS_LENGTH + col].out[0],
                           kk[row * CAS_LENGTH + col + 1].in[2]);
        }
      }
    }
  }
};
