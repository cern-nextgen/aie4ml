// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Buffer-port Conv2D: the DMA lands the channel-blocked frame, so each kernel is the shared
// compute core plus its place in the cascade.

#include "conv2d.h"

#include "conv2d_core_one_block.h"  // implementations stay out of the header the ADF graph parses

using namespace adf;

template<typename ConfigT>
conv2d_base<ConfigT>::conv2d_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
  conv2d_check_contract<ConfigT>();
}

template<typename ConfigT>
void conv2d_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                 const weight_t (&wts)[ConfigT::WN],
                                 const bias_t (&bias)[ConfigT::BN],
                                 output_buffer<result_t>& ofm)
{
  conv2d_compute<ConfigT, false, false>(const_cast<data_t*>(ifm.data()), wts, bias, ofm.data(), nullptr, nullptr);
}

template<typename ConfigT>
void conv2d_first<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::WN],
                                const bias_t (&bias)[ConfigT::BN],
                                output_cascade<cascade_t>* outCascade)
{
  conv2d_compute<ConfigT, false, true>(
      const_cast<data_t*>(ifm.data()), wts, bias, nullptr, nullptr, outCascade);
}

template<typename ConfigT>
void conv2d_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                 const weight_t (&wts)[ConfigT::WN],
                                 input_cascade<cascade_t>* inCascade,
                                 output_cascade<cascade_t>* outCascade)
{
  conv2d_compute<ConfigT, true, true>(
      const_cast<data_t*>(ifm.data()), wts, nullptr, nullptr, inCascade, outCascade);
}

template<typename ConfigT>
void conv2d_last<ConfigT>::run(input_buffer<data_t>& ifm,
                               const weight_t (&wts)[ConfigT::WN],
                               input_cascade<cascade_t>* inCascade,
                               output_buffer<result_t>& ofm)
{
  conv2d_compute<ConfigT, true, false>(const_cast<data_t*>(ifm.data()), wts, nullptr, ofm.data(), inCascade, nullptr);
}
