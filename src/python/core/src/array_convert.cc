/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for Torch::core::convert. 
 *    Types supported are uint8, uint16 and float64
 */

#include <boost/python.hpp>
#include "core/convert.h"
#include "core/python/array_base.h"

namespace tp = Torch::python;
namespace bp = boost::python;

static const char* ARRAY_CONVERT_DOC = "Return a blitz array of the specified type by converting the given blitz array.";
static const char* ARRAY_CONVERT_DOC_RANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified input and output ranges.";
static const char* ARRAY_CONVERT_DOC_TORANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified output range.";
static const char* ARRAY_CONVERT_DOC_FROMRANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified input range.";

template <typename U, typename T, int D>
static inline blitz::Array<U,D> convert1 (const blitz::Array<T,D>& a) {
  return Torch::core::convert<U>(a);
}

template <typename U, typename T, int D>
static inline blitz::Array<U,D> convert2 (const blitz::Array<T,D>& a, bp::tuple dst, bp::tuple src) {
  U dst_min = bp::extract<U>(dst[0]);
  U dst_max = bp::extract<U>(dst[1]);
  T src_min = bp::extract<T>(src[0]);
  T src_max = bp::extract<T>(src[1]);
  return Torch::core::convert<U>(a, dst_min, dst_max, src_min, src_max);
}

template <typename U, typename T, int D>
static inline blitz::Array<U,D> convert3 (const blitz::Array<T,D>& a, bp::tuple dst) {
  U dst_min = bp::extract<U>(dst[0]);
  U dst_max = bp::extract<U>(dst[1]);
  return Torch::core::convertToRange<U>(a, dst_min, dst_max);
}

template <typename U, typename T, int D>
static inline blitz::Array<U,D> convert4 (const blitz::Array<T,D>& a, bp::tuple src) {
  T src_min = bp::extract<T>(src[0]);
  T src_max = bp::extract<T>(src[1]);
  return Torch::core::convertFromRange<U>(a, src_min, src_max);
}

/**
 * Loads stuff for exporting the array
 */
template <typename T, int N> static void bind_convert (tp::array<T,N>& array) {
  array.object()->def("__convert_uint8__", &convert1<uint8_t,T,N>, (bp::arg("self")), ARRAY_CONVERT_DOC); 
  array.object()->def("__convert_uint8__", &convert2<uint8_t,T,N>, (bp::arg("self"), bp::arg("destRange"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_RANGE);
  array.object()->def("__convert_uint8__", &convert3<uint8_t,T,N>, (bp::arg("self"), bp::arg("destRange")), ARRAY_CONVERT_DOC_TORANGE); 
  array.object()->def("__convert_uint8__", &convert4<uint8_t,T,N>, (bp::arg("self"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_FROMRANGE); 
  array.object()->def("__convert_uint16__", &convert1<uint16_t,T,N>, (bp::arg("self")), ARRAY_CONVERT_DOC); 
  array.object()->def("__convert_uint16__", &convert2<uint16_t,T,N>, (bp::arg("self"), bp::arg("destRange"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_RANGE);
  array.object()->def("__convert_uint16__", &convert3<uint16_t,T,N>, (bp::arg("self"), bp::arg("destRange")), ARRAY_CONVERT_DOC_TORANGE); 
  array.object()->def("__convert_uint16__", &convert4<uint16_t,T,N>, (bp::arg("self"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_FROMRANGE); 
  array.object()->def("__convert_float64__", &convert1<double,T,N>, (bp::arg("self")), ARRAY_CONVERT_DOC); 
  array.object()->def("__convert_float64__", &convert2<double,T,N>, (bp::arg("self"), bp::arg("destRange"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_RANGE);
  array.object()->def("__convert_float64__", &convert3<double,T,N>, (bp::arg("self"), bp::arg("destRange")), ARRAY_CONVERT_DOC_TORANGE); 
  array.object()->def("__convert_float64__", &convert4<double,T,N>, (bp::arg("self"), bp::arg("sourceRange")), ARRAY_CONVERT_DOC_FROMRANGE); 
}

void bind_array_convert() {
  bind_convert(tp::bool_1);
  bind_convert(tp::bool_2);
  bind_convert(tp::bool_3);
  bind_convert(tp::bool_4);
  
  bind_convert(tp::int8_1);
  bind_convert(tp::int8_2);
  bind_convert(tp::int8_3);
  bind_convert(tp::int8_4);
  
  bind_convert(tp::int16_1);
  bind_convert(tp::int16_2);
  bind_convert(tp::int16_3);
  bind_convert(tp::int16_4);
  
  bind_convert(tp::int32_1);
  bind_convert(tp::int32_2);
  bind_convert(tp::int32_3);
  bind_convert(tp::int32_4);
  
  bind_convert(tp::int64_1);
  bind_convert(tp::int64_2);
  bind_convert(tp::int64_3);
  bind_convert(tp::int64_4);
  
  bind_convert(tp::uint8_1);
  bind_convert(tp::uint8_2);
  bind_convert(tp::uint8_3);
  bind_convert(tp::uint8_4);
  
  bind_convert(tp::uint16_1);
  bind_convert(tp::uint16_2);
  bind_convert(tp::uint16_3);
  bind_convert(tp::uint16_4);
  
  bind_convert(tp::uint32_1);
  bind_convert(tp::uint32_2);
  bind_convert(tp::uint32_3);
  bind_convert(tp::uint32_4);
  
  bind_convert(tp::uint64_1);
  bind_convert(tp::uint64_2);
  bind_convert(tp::uint64_3);
  bind_convert(tp::uint64_4);
  
  bind_convert(tp::float32_1);
  bind_convert(tp::float32_2);
  bind_convert(tp::float32_3);
  bind_convert(tp::float32_4);
  
  bind_convert(tp::float64_1);
  bind_convert(tp::float64_2);
  bind_convert(tp::float64_3);
  bind_convert(tp::float64_4);
  
  //bind_convert(tp::float128_1);
  //bind_convert(tp::float128_2);
  //bind_convert(tp::float128_3);
  //bind_convert(tp::float128_4);
} 
