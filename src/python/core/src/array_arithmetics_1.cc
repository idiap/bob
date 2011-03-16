/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:17:53 2011 
 *
 * @brief All sorts of array arithmetic operations 
 */

#include "core/python/array_arithmetics.h"

namespace tp = Torch::python;

void bind_array_arithmetics_1 () {
  tp::bind_bool_arith(tp::bool_1);
  tp::bind_int_arith(tp::int8_1);
  tp::bind_int_arith(tp::int16_1);
  tp::bind_int_arith(tp::int32_1);
  tp::bind_int_arith(tp::int64_1);
  tp::bind_uint_arith(tp::uint8_1);
  tp::bind_uint_arith(tp::uint16_1);
  tp::bind_uint_arith(tp::uint32_1);
  tp::bind_uint_arith(tp::uint64_1);
  tp::bind_float_arith(tp::float32_1);
  tp::bind_float_arith(tp::float64_1);
  //tp::bind_float_arith(tp::float128_1);
  tp::bind_complex_arith(tp::complex64_1);
  tp::bind_complex_arith(tp::complex128_1);
  //tp::bind_complex_arith(tp::complex256_1);
}
