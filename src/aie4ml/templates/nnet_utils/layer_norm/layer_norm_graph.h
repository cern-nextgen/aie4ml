#pragma once

#include <adf.h>
#include "layer_norm.h"
#include "buffer_location.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class layer_norm_graph : public graph {
public:
    static constexpr int CAS_NUM = ConfigT::CAS_NUM;
    static constexpr int ROWS    = ConfigT::ROWS;
    static constexpr int COLS    = ConfigT::COLS;

    input_port  in1[CAS_NUM];
    output_port out1[CAS_NUM];

    adf::port<adf::direction::in> gamma[CAS_NUM];
    adf::port<adf::direction::in> beta[CAS_NUM];

    kernel kk[CAS_NUM];

    void place_graph(int COL_START, int ROW_START)
    {
        for (int row = 0; row < CAS_NUM; ++row) {
            const int tileCol = COL_START;
            const int tileRow = ROW_START + row;

            adf::location<adf::kernel>(kk[row]) = adf::tile(tileCol, tileRow);

            pin_buffer(kk[row].in[0], ConfigT::IN1_BUFFER_LOCATIONS[row], COL_START, ROW_START);

            adf::location<adf::stack>(kk[row]) = adf::bank(tileCol, tileRow, AIE4ML_STACK_BANK);

            adf::location<adf::buffer>(kk[row].in[1]) = adf::bank(tileCol, tileRow, AIE4ML_STACK_BANK);
            adf::location<adf::buffer>(kk[row].in[2]) = adf::bank(tileCol, tileRow, AIE4ML_WEIGHT_BANK);

            pin_buffer(kk[row].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[row], COL_START, ROW_START);
        }
    }

    layer_norm_graph()
    {
        for (int i = 0; i < CAS_NUM; ++i) {
            // The layout the op declares picks the kernel: 'tiled' reads and writes
            // microtiles so a matmul on either side needs no memtile, 'linear' keeps rows
            // contiguous, which is what every other op reads.
            if constexpr (ConfigT::LAYOUT_TILED) {
                kk[i] = kernel::create_object<layernorm_i8_tiled<ConfigT>>();
            } else {
                kk[i] = kernel::create_object<layernorm_i8<ConfigT>>();
            }
            source(kk[i])         = "layer_norm.cpp";
            runtime<ratio>(kk[i]) = 1.0;

            connect<>(in1[i], kk[i].in[0]);
            dimensions(kk[i].in[0])  = {ROWS * COLS};
            dimensions(kk[i].out[0]) = {ROWS * COLS};
            connect<>(kk[i].out[0], out1[i]);

            single_buffer(kk[i].in[1]);
            connect<parameter>(gamma[i], async(kk[i].in[1]));

            single_buffer(kk[i].in[2]);
            connect<parameter>(beta[i], async(kk[i].in[2]));
        }
    }
};
