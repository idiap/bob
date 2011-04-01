/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 23:24:51 2011 
 *
 * @brief Reductions 
 */

#include "core/python/array_reduction.h"

namespace tp = Torch::python;

void bind_array_reductions_4 () {
  tp::bind_reductions(tp::bool_4);
  tp::bind_reductions(tp::int8_4);
  tp::bind_reductions(tp::int16_4);
  tp::bind_reductions(tp::int32_4);
  tp::bind_reductions(tp::int64_4);
  tp::bind_reductions(tp::uint8_4);
  tp::bind_reductions(tp::uint16_4);
  tp::bind_reductions(tp::uint32_4);
  tp::bind_reductions(tp::uint64_4);
  tp::bind_reductions(tp::float32_4);
  tp::bind_reductions(tp::float64_4);
  //tp::bind_reductions(tp::float128_4);
  tp::bind_common_reductions(tp::complex64_4);
  tp::bind_common_reductions(tp::complex128_4);
  //tp::bind_common_reductions(tp::complex256_4);
}
