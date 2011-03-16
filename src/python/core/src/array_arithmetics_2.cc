/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:17:53 2011 
 *
 * @brief All sorts of array arithmetic operations 
 */

#include "core/python/array_arithmetics.h"

namespace tp = Torch::python;

void bind_array_arithmetics_2 () {
  tp::bind_bool_arith(tp::bool_2);
  tp::bind_int_arith(tp::int8_2);
  tp::bind_int_arith(tp::int16_2);
  tp::bind_int_arith(tp::int32_2);
  tp::bind_int_arith(tp::int64_2);
  tp::bind_uint_arith(tp::uint8_2);
  tp::bind_uint_arith(tp::uint16_2);
  tp::bind_uint_arith(tp::uint32_2);
  tp::bind_uint_arith(tp::uint64_2);
  tp::bind_float_arith(tp::float32_2);
  tp::bind_float_arith(tp::float64_2);
  //tp::bind_float_arith(tp::float128_2);
  tp::bind_complex_arith(tp::complex64_2);
  tp::bind_complex_arith(tp::complex128_2);
  //tp::bind_complex_arith(tp::complex256_2);
}
