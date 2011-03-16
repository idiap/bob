/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:17:53 2011 
 *
 * @brief All sorts of array arithmetic operations 
 */

#include "core/python/array_arithmetics.h"

namespace tp = Torch::python;

void bind_array_arithmetics_4 () {
  tp::bind_bool_arith(tp::bool_4);
  tp::bind_int_arith(tp::int8_4);
  tp::bind_int_arith(tp::int16_4);
  tp::bind_int_arith(tp::int32_4);
  tp::bind_int_arith(tp::int64_4);
  tp::bind_uint_arith(tp::uint8_4);
  tp::bind_uint_arith(tp::uint16_4);
  tp::bind_uint_arith(tp::uint32_4);
  tp::bind_uint_arith(tp::uint64_4);
  tp::bind_float_arith(tp::float32_4);
  tp::bind_float_arith(tp::float64_4);
  //tp::bind_float_arith(tp::float128_4);
  tp::bind_complex_arith(tp::complex64_4);
  tp::bind_complex_arith(tp::complex128_4);
  //tp::bind_complex_arith(tp::complex256_4);
}
