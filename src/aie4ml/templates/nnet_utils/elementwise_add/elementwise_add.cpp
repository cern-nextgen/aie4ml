#include "elementwise_add.h"

using namespace adf;

template<typename ConfigT>
elementwise_add_base<ConfigT>::elementwise_add_base()
{
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
}

template<typename ConfigT>
void elementwise_add_kernel<ConfigT>::run(
    input_buffer<lhs_t>& lhs,
    input_buffer<rhs_t>& rhs,
    output_buffer<result_t>& out)
{
  const lhs_t* __restrict lhs_ptr = lhs.data();
  const rhs_t* __restrict rhs_ptr = rhs.data();
  result_t* __restrict out_ptr = out.data();

  constexpr int VEC = ConfigT::VEC_SIZE;
  constexpr int iters = ConfigT::TILE_ELEMENTS / VEC;

  static_assert(!(ConfigT::TRANSPOSE_LHS || ConfigT::TRANSPOSE_RHS)
                    || VEC == MT_OUTER * MT_INNER,
                "transposed elementwise operands require one microtile per vector load");

  for (int i = 0; i < iters; ++i)
    chess_prepare_for_pipelining
  {
    aie::vector<lhs_t, VEC> va = aie::load_v<VEC>(lhs_ptr);
    if constexpr (ConfigT::TRANSPOSE_LHS)
      va = aie::transpose(va, MT_INNER, MT_OUTER);
    lhs_ptr += VEC;
    aie::vector<rhs_t, VEC> vb = aie::load_v<VEC>(rhs_ptr);
    if constexpr (ConfigT::TRANSPOSE_RHS)
      vb = aie::transpose(vb, MT_INNER, MT_OUTER);
    rhs_ptr += VEC;
    auto acc = aie::add(aie::from_vector<acc_scalar_t>(va), vb);
    auto sum = aie::to_vector<result_t>(acc, ConfigT::SHIFT);
    if constexpr (ConfigT::USE_RELU)
      sum = aie::max(sum, aie::zeros<result_t, VEC>());

    aie::store_v(out_ptr, sum);
    out_ptr += VEC;
  }
}
