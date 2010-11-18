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
void bind_core_array_bool();
void bind_core_array_int8();
void bind_core_array_uint8();
void bind_core_array_int16();
void bind_core_array_uint16();
void bind_core_array_int32();
void bind_core_array_uint32();
void bind_core_array_int64();
void bind_core_array_uint64();
void bind_core_array_float32();
void bind_core_array_float64();
void bind_core_array_float128();
void bind_core_array_complex64();
void bind_core_array_complex128();
void bind_core_array_complex256();

BOOST_PYTHON_MODULE(libpytorch_core_array) {
  scope().attr("__doc__") = "Torch core classes and sub-classes for array manipulation";
  bind_core_array_tinyvector();
  bind_core_array();
  bind_core_array_bool();
  bind_core_array_int8();
  bind_core_array_uint8();
  bind_core_array_int16();
  bind_core_array_uint16();
  bind_core_array_int32();
  bind_core_array_uint32();
  bind_core_array_int64();
  bind_core_array_uint64();
  bind_core_array_float32();
  bind_core_array_float64();
  bind_core_array_float128();
  bind_core_array_complex64();
  bind_core_array_complex128();
  bind_core_array_complex256();
}
