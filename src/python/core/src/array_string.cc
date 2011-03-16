/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 13 Mar 08:36:07 2011 
 *
 * @brief Several stringyfied representations of arrays 
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

namespace tp = Torch::python;
namespace bp = boost::python;

/**
 * Prints to a string
 */
template<typename T, int N> static  bp::str str(const blitz::Array<T,N>& a) {
  std::ostringstream s;
  s << a;
  return boost::python::str(s.str());
}

template <typename T, int N> 
static void bind_strings (tp::array<T,N>& array) {
  array.object()->def("__str__", &str<T,N>, "String representation of this object");
}

void bind_array_string () {
  bind_strings(tp::bool_1);
  bind_strings(tp::bool_2);
  bind_strings(tp::bool_3);
  bind_strings(tp::bool_4);
  
  bind_strings(tp::int8_1);
  bind_strings(tp::int8_2);
  bind_strings(tp::int8_3);
  bind_strings(tp::int8_4);
  
  bind_strings(tp::int16_1);
  bind_strings(tp::int16_2);
  bind_strings(tp::int16_3);
  bind_strings(tp::int16_4);
  
  bind_strings(tp::int32_1);
  bind_strings(tp::int32_2);
  bind_strings(tp::int32_3);
  bind_strings(tp::int32_4);
  
  bind_strings(tp::int64_1);
  bind_strings(tp::int64_2);
  bind_strings(tp::int64_3);
  bind_strings(tp::int64_4);
  
  bind_strings(tp::uint8_1);
  bind_strings(tp::uint8_2);
  bind_strings(tp::uint8_3);
  bind_strings(tp::uint8_4);
  
  bind_strings(tp::uint16_1);
  bind_strings(tp::uint16_2);
  bind_strings(tp::uint16_3);
  bind_strings(tp::uint16_4);
  
  bind_strings(tp::uint32_1);
  bind_strings(tp::uint32_2);
  bind_strings(tp::uint32_3);
  bind_strings(tp::uint32_4);
  
  bind_strings(tp::uint64_1);
  bind_strings(tp::uint64_2);
  bind_strings(tp::uint64_3);
  bind_strings(tp::uint64_4);
  
  bind_strings(tp::float32_1);
  bind_strings(tp::float32_2);
  bind_strings(tp::float32_3);
  bind_strings(tp::float32_4);
  
  bind_strings(tp::float64_1);
  bind_strings(tp::float64_2);
  bind_strings(tp::float64_3);
  bind_strings(tp::float64_4);
  
  //bind_strings(tp::float128_1);
  //bind_strings(tp::float128_2);
  //bind_strings(tp::float128_3);
  //bind_strings(tp::float128_4);
  
  bind_strings(tp::complex64_1);
  bind_strings(tp::complex64_2);
  bind_strings(tp::complex64_3);
  bind_strings(tp::complex64_4);
  
  bind_strings(tp::complex128_1);
  bind_strings(tp::complex128_2);
  bind_strings(tp::complex128_3);
  bind_strings(tp::complex128_4);
  
  //bind_strings(tp::complex256_1);
  //bind_strings(tp::complex256_2);
  //bind_strings(tp::complex256_3);
  //bind_strings(tp::complex256_4);
}
