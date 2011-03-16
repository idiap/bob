/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 23:49:43 2011 
 *
 * @brief Mathematics operator on arrays 
 */

#include "core/python/array_math.h"

namespace tp = Torch::python;

void bind_array_math_1 () {
  tp::bind_bool_math(tp::bool_1);
  tp::bind_int_math(tp::int8_1);
  tp::bind_int_math(tp::int16_1);
  tp::bind_int_math(tp::int32_1);
  tp::bind_int_math(tp::int64_1);
  tp::bind_int_math(tp::uint8_1);
  tp::bind_int_math(tp::uint16_1);
  tp::bind_int_math(tp::uint32_1);
  tp::bind_int_math(tp::uint64_1);
  tp::bind_float_math(tp::float32_1);
  tp::bind_float_math(tp::float64_1);
  //tp::bind_float_math(tp::float128_1);
  tp::bind_complex_math(tp::complex64_1);
  tp::bind_complex_math(tp::complex128_1);
  //tp::bind_complex_math(tp::complex256_1);
}
