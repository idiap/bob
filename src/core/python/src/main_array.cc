/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_core_array_tinyvector();
void bind_core_array();

// START: generate.py
void bind_core_array_bool_1();
void bind_core_array_int8_1();
void bind_core_array_uint8_1();
void bind_core_array_int64_1();
void bind_core_array_uint64_1();
void bind_core_array_float32_1();
void bind_core_array_complex64_1();
void bind_core_array_int16_1();
void bind_core_array_uint16_1();
void bind_core_array_float64_1();
void bind_core_array_int32_1();
void bind_core_array_uint32_1();
void bind_core_array_float128_1();
void bind_core_array_complex128_1();
void bind_core_array_complex256_1();
void bind_core_array_bool_2();
void bind_core_array_int8_2();
void bind_core_array_uint8_2();
void bind_core_array_int64_2();
void bind_core_array_uint64_2();
void bind_core_array_float32_2();
void bind_core_array_complex64_2();
void bind_core_array_int16_2();
void bind_core_array_uint16_2();
void bind_core_array_float64_2();
void bind_core_array_int32_2();
void bind_core_array_uint32_2();
void bind_core_array_float128_2();
void bind_core_array_complex128_2();
void bind_core_array_complex256_2();
void bind_core_array_bool_3();
void bind_core_array_int8_3();
void bind_core_array_uint8_3();
void bind_core_array_int64_3();
void bind_core_array_uint64_3();
void bind_core_array_float32_3();
void bind_core_array_complex64_3();
void bind_core_array_int16_3();
void bind_core_array_uint16_3();
void bind_core_array_float64_3();
void bind_core_array_int32_3();
void bind_core_array_uint32_3();
void bind_core_array_float128_3();
void bind_core_array_complex128_3();
void bind_core_array_complex256_3();
void bind_core_array_bool_4();
void bind_core_array_int8_4();
void bind_core_array_uint8_4();
void bind_core_array_int64_4();
void bind_core_array_uint64_4();
void bind_core_array_float32_4();
void bind_core_array_complex64_4();
void bind_core_array_int16_4();
void bind_core_array_uint16_4();
void bind_core_array_float64_4();
void bind_core_array_int32_4();
void bind_core_array_uint32_4();
void bind_core_array_float128_4();
void bind_core_array_complex128_4();
void bind_core_array_complex256_4();
// END: generate.py

BOOST_PYTHON_MODULE(libpytorch_core_array) {
  scope().attr("__doc__") = "Torch core classes and sub-classes for array manipulation";
  bind_core_array_tinyvector();
  bind_core_array();

  // START: generate.py
  bind_core_array_bool_1();
  bind_core_array_int8_1();
  bind_core_array_uint8_1();
  bind_core_array_int64_1();
  bind_core_array_uint64_1();
  bind_core_array_float32_1();
  bind_core_array_complex64_1();
  bind_core_array_int16_1();
  bind_core_array_uint16_1();
  bind_core_array_float64_1();
  bind_core_array_int32_1();
  bind_core_array_uint32_1();
  bind_core_array_float128_1();
  bind_core_array_complex128_1();
  bind_core_array_complex256_1();
  bind_core_array_bool_2();
  bind_core_array_int8_2();
  bind_core_array_uint8_2();
  bind_core_array_int64_2();
  bind_core_array_uint64_2();
  bind_core_array_float32_2();
  bind_core_array_complex64_2();
  bind_core_array_int16_2();
  bind_core_array_uint16_2();
  bind_core_array_float64_2();
  bind_core_array_int32_2();
  bind_core_array_uint32_2();
  bind_core_array_float128_2();
  bind_core_array_complex128_2();
  bind_core_array_complex256_2();
  bind_core_array_bool_3();
  bind_core_array_int8_3();
  bind_core_array_uint8_3();
  bind_core_array_int64_3();
  bind_core_array_uint64_3();
  bind_core_array_float32_3();
  bind_core_array_complex64_3();
  bind_core_array_int16_3();
  bind_core_array_uint16_3();
  bind_core_array_float64_3();
  bind_core_array_int32_3();
  bind_core_array_uint32_3();
  bind_core_array_float128_3();
  bind_core_array_complex128_3();
  bind_core_array_complex256_3();
  bind_core_array_bool_4();
  bind_core_array_int8_4();
  bind_core_array_uint8_4();
  bind_core_array_int64_4();
  bind_core_array_uint64_4();
  bind_core_array_float32_4();
  bind_core_array_complex64_4();
  bind_core_array_int16_4();
  bind_core_array_uint16_4();
  bind_core_array_float64_4();
  bind_core_array_int32_4();
  bind_core_array_uint32_4();
  bind_core_array_float128_4();
  bind_core_array_complex128_4();
  bind_core_array_complex256_4();
  // END: generate.py

}
