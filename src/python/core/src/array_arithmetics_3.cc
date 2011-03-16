/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:17:53 2011 
 *
 * @brief All sorts of array arithmetic operations 
 */

#include "core/python/array_arithmetics.h"

namespace tp = Torch::python;

void bind_array_arithmetics_3 () {
  tp::bind_bool_arith(tp::bool_3);
  tp::bind_int_arith(tp::int8_3);
  tp::bind_int_arith(tp::int16_3);
  tp::bind_int_arith(tp::int32_3);
  tp::bind_int_arith(tp::int64_3);
  tp::bind_uint_arith(tp::uint8_3);
  tp::bind_uint_arith(tp::uint16_3);
  tp::bind_uint_arith(tp::uint32_3);
  tp::bind_uint_arith(tp::uint64_3);
  tp::bind_float_arith(tp::float32_3);
  tp::bind_float_arith(tp::float64_3);
  //tp::bind_float_arith(tp::float128_3);
  tp::bind_complex_arith(tp::complex64_3);
  tp::bind_complex_arith(tp::complex128_3);
  //tp::bind_complex_arith(tp::complex256_3);
}
