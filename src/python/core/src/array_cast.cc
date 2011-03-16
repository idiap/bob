/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 14 Mar 09:53:06 2011 
 *
 * @brief Several methods allowing conversion to/from numpy 
 */

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "core/cast.h"
#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"
#include "core/python/ndarray.h"

namespace tp = Torch::python;
namespace bp = boost::python;
namespace tca = Torch::core::array;

template <typename T, int N> static
bp::ndarray make_ndarray(const blitz::Array<T,N>& bzarray) {
  return bp::ndarray(bzarray);
}

template <typename T, int N> static
bp::ndarray cast_ndarray(const blitz::Array<T,N>& bzarray, NPY_TYPES cast_to) {
  return bp::ndarray(bzarray, cast_to);
}

/**
 * Load casts to different base types
 */
template <typename T, int N> static 
bp::object cast(blitz::Array<T,N>& array, const char* eltype_string) {
  tca::ElementType eltype = tca::unstringize(eltype_string);
  switch (eltype) {
    case tca::t_bool: 
      return bp::object(Torch::core::cast<bool,T>(array));
    case tca::t_int8: 
      return bp::object(Torch::core::cast<int8_t,T>(array));
    case tca::t_int16: 
      return bp::object(Torch::core::cast<int16_t,T>(array));
    case tca::t_int32: 
      return bp::object(Torch::core::cast<int32_t,T>(array));
    case tca::t_int64: 
      return bp::object(Torch::core::cast<int64_t,T>(array));
    case tca::t_uint8: 
      return bp::object(Torch::core::cast<uint8_t,T>(array));
    case tca::t_uint16: 
      return bp::object(Torch::core::cast<uint16_t,T>(array));
    case tca::t_uint32: 
      return bp::object(Torch::core::cast<uint32_t,T>(array));
    case tca::t_uint64: 
      return bp::object(Torch::core::cast<uint64_t,T>(array));
    case tca::t_float32: 
      return bp::object(Torch::core::cast<float,T>(array));
    case tca::t_float64: 
      return bp::object(Torch::core::cast<double,T>(array));
    //case tca::t_float128: 
    //  return bp::object(Torch::core::cast<long double,T>(array));
    case tca::t_complex64: 
      return bp::object(Torch::core::cast<std::complex<float>,T>(array));
    case tca::t_complex128: 
      return bp::object(Torch::core::cast<std::complex<double>,T>(array));
    //case tca::t_complex256: 
    //  return bp::object(Torch::core::cast<std::complex<long double>,T>(array));
    default:
      break;
  }
  boost::format msg("Unsupported cast type: %s (interpreted as %s)");
  msg % eltype_string;
  msg % tca::stringize(eltype);
  PyErr_SetString(PyExc_TypeError, msg.str().c_str());
  bp::throw_error_already_set();
  return bp::object(); //shuts up gcc
}

/**
 * Loads stuff for exporting the array
 */
template <typename T, int N> static void bind_cast (tp::array<T,N>& array) {
  array.object()->def("as_ndarray", &make_ndarray<T,N>, (bp::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");
  array.object()->def("__array__", &make_ndarray<T,N>, (bp::arg("self")), "Creates a copy of this array as a NumPy Array with the same dimensions and storage type");
  array.object()->def("as_ndarray", &cast_ndarray<T,N>, (bp::arg("self"), bp::arg("cast_to")), "Creates a copy of this array as a NumPy Array with the same dimensions, but casts the storage type to the (numpy) type defined");
  array.object()->def("cast", &cast<T,N>, (bp::arg("self"), bp::arg("cast_to")), "Creates a copy of this array as another blitz::Array<> with different element type. The cast_to parameter should be picked from one of the allowed values for Torch blitz Arrays");
}

void bind_array_cast () {
  bind_cast(tp::bool_1);
  bind_cast(tp::bool_2);
  bind_cast(tp::bool_3);
  bind_cast(tp::bool_4);
  
  bind_cast(tp::int8_1);
  bind_cast(tp::int8_2);
  bind_cast(tp::int8_3);
  bind_cast(tp::int8_4);
  
  bind_cast(tp::int16_1);
  bind_cast(tp::int16_2);
  bind_cast(tp::int16_3);
  bind_cast(tp::int16_4);
  
  bind_cast(tp::int32_1);
  bind_cast(tp::int32_2);
  bind_cast(tp::int32_3);
  bind_cast(tp::int32_4);
  
  bind_cast(tp::int64_1);
  bind_cast(tp::int64_2);
  bind_cast(tp::int64_3);
  bind_cast(tp::int64_4);
  
  bind_cast(tp::uint8_1);
  bind_cast(tp::uint8_2);
  bind_cast(tp::uint8_3);
  bind_cast(tp::uint8_4);
  
  bind_cast(tp::uint16_1);
  bind_cast(tp::uint16_2);
  bind_cast(tp::uint16_3);
  bind_cast(tp::uint16_4);
  
  bind_cast(tp::uint32_1);
  bind_cast(tp::uint32_2);
  bind_cast(tp::uint32_3);
  bind_cast(tp::uint32_4);
  
  bind_cast(tp::uint64_1);
  bind_cast(tp::uint64_2);
  bind_cast(tp::uint64_3);
  bind_cast(tp::uint64_4);
  
  bind_cast(tp::float32_1);
  bind_cast(tp::float32_2);
  bind_cast(tp::float32_3);
  bind_cast(tp::float32_4);
  
  bind_cast(tp::float64_1);
  bind_cast(tp::float64_2);
  bind_cast(tp::float64_3);
  bind_cast(tp::float64_4);
  
  //bind_cast(tp::float128_1);
  //bind_cast(tp::float128_2);
  //bind_cast(tp::float128_3);
  //bind_cast(tp::float128_4);
  
  bind_cast(tp::complex64_1);
  bind_cast(tp::complex64_2);
  bind_cast(tp::complex64_3);
  bind_cast(tp::complex64_4);
  
  bind_cast(tp::complex128_1);
  bind_cast(tp::complex128_2);
  bind_cast(tp::complex128_3);
  bind_cast(tp::complex128_4);
  
  //bind_cast(tp::complex256_1);
  //bind_cast(tp::complex256_2);
  //bind_cast(tp::complex256_3);
  //bind_cast(tp::complex256_4);
}
