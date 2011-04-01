/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 23:24:51 2011 
 *
 * @brief Reductions 
 */

#include "core/python/array_reduction.h"

namespace tp = Torch::python;

void bind_array_reductions_1 () {
  tp::bind_reductions(tp::bool_1);
  tp::bind_reductions(tp::int8_1);
  tp::bind_reductions(tp::int16_1);
  tp::bind_reductions(tp::int32_1);
  tp::bind_reductions(tp::int64_1);
  tp::bind_reductions(tp::uint8_1);
  tp::bind_reductions(tp::uint16_1);
  tp::bind_reductions(tp::uint32_1);
  tp::bind_reductions(tp::uint64_1);
  tp::bind_reductions(tp::float32_1);
  tp::bind_reductions(tp::float64_1);
  //tp::bind_reductions(tp::float128_1);
  tp::bind_common_reductions(tp::complex64_1);
  tp::bind_common_reductions(tp::complex128_1);
  //tp::bind_common_reductions(tp::complex256_1);
}
