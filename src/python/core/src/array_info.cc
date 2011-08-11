/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Fri 11 Mar 22:15:23 2011 
 *
 * @brief Information on the array contents and data 
 */

#include <boost/python.hpp>

#include "core/python/TypeMapper.h"
#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

#include "core/blitz_compat.h"

namespace tp = Torch::python;
namespace bp = boost::python;

/**
 * Returns the Numpy typecode, typename or C enum for this blitz Array
 */
template<typename T, int N> static const char* to_typecode
(const blitz::Array<T,N>& a) {
  return tp::TYPEMAP.type_to_typecode<T>().c_str();
}

template<typename T, int N> static const char* to_typename
(const blitz::Array<T,N>& a) {
  return tp::TYPEMAP.type_to_typename<T>().c_str();
}

template<typename T, int N> static NPY_TYPES to_enum
(const blitz::Array<T,N>& a) {
  return tp::TYPEMAP.type_to_enum<T>();
}

/**
 * Declares a bunch of informative methods for arrays
 */
template <typename T, int N> 
static void bind (tp::array<T,N>& array) {
  typedef typename tp::array<T,N>::array_type array_type;
  typedef typename tp::array<T,N>::shape_type shape_type;
  typedef typename tp::array<T,N>::stride_type stride_type;

  array.object()->def("extent", (int (array_type::*)(int) const)&array_type::extent, (boost::python::arg("self"), boost::python::arg("dimension")), "Returns the array size in one of the dimensions");

  array.object()->def("dimensions", &array_type::dimensions, "Total number of dimensions on this array");

  array.object()->def("rank", &array_type::dimensions, "Total number of dimensions on this array");

  array.object()->def("rows", &array_type::rows, "Equivalent to extent(firstDim)");

  array.object()->def("columns", &array_type::columns, "Equivalent to extent(secondDim)");

  array.object()->def("depth", &array_type::depth, "Equivalent to extent(thirdDim)");

  array.object()->def("size", &array_type::size, "Total number of elements in this array");

  array.object()->def("base", (const shape_type& (array_type::*)() const)(&array_type::base), boost::python::return_value_policy<boost::python::return_by_value>(), "The base of a dimension is the first valid index value. A typical C-style array will have base of zero; a Fortran-style array will have base of one. The base can be different for each dimension, but only if you deliberately use a Range-argument constructor or design a custom storage ordering.");

  array.object()->def("base", (int (array_type::*)(int) const)(&array_type::base), "The base of a dimension is the first valid index value. A typical C-style array will have base of zero; a Fortran-style array will have base of one. The base can be different for each dimension, but only if you deliberately use a Range-argument constructor or design a custom storage ordering.");

  array.object()->def("isMajorRank", &array_type::isMajorRank, boost::python::arg("dimension"), "Returns true if the dimension has the largest stride. For C-style arrays, the first dimension always has the largest stride. For Fortran-style arrays, the last dimension has the largest stride.");

  array.object()->def("isMinorRank", &array_type::isMinorRank, boost::python::arg("dimension"), "Returns true if the dimension does not have the largest stride. See also isMajorRank().");

  array.object()->def("isStorageContiguous", &array_type::isStorageContiguous, "Returns true if the array data is stored contiguously in memory. If you slice the array or work on subarrays, there can be skips -- the array data is interspersed with other data not part of the array. See also the various data..() functions. If you need to ensure that the storage is contiguous, try reference(copy()).");

  array.object()->def("numElements", &array_type::numElements, "The same as size()");

  array.object()->def("shape", &array_type::shape, boost::python::return_value_policy<boost::python::return_by_value>(), "Returns the vector of extents (lengths) of the array");

  array.object()->def("stride", (const stride_type& (array_type::*)() const)&array_type::stride, boost::python::return_value_policy<boost::python::return_by_value>(), "A stride is the distance between pointers to two array elements which are adjacent in a dimension. For example, A.stride(firstDim) is equal to &A(1,0,0) - &A(0,0,0). The stride for the second dimension, A.stride(secondDim), is equal to &A(0,1,0) - &A(0,0,0), and so on. For more information about strides, see the description of custom storage formats in Section 2.9 of the Blitz manual. See also the description of parameters like firstDim and secondDim in the previous section of the same manual.");

  array.object()->def("stride", (blitz::diffType (array_type::*)(int) const)&array_type::stride, "A stride is the distance between pointers to two array elements which are adjacent in a dimension. For example, A.stride(firstDim) is equal to &A(1,0,0) - &A(0,0,0). The stride for the second dimension, A.stride(secondDim), is equal to &A(0,1,0) - &A(0,0,0), and so on. For more information about strides, see the description of custom storage formats in Section 2.9 of the Blitz manual. See also the description of parameters like firstDim and secondDim in the previous section of the same manual.");

  //some type information that correlates the C++ type to the Numpy C-API
  //types.
  array.object()->def("numpy_typecode", &to_typecode<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typecode of this blitz::Array");

  array.object()->def("numpy_typename", &to_typename<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy typename of this blitz::Array");

  array.object()->def("numpy_enum", &to_enum<T, N>, (boost::python::arg("self")), "Describes the equivalent numpy C enumeration of this blitz::Array");

}

void bind_array_info () {
  bind(tp::bool_1);
  bind(tp::bool_2);
  bind(tp::bool_3);
  bind(tp::bool_4);
  
  bind(tp::int8_1);
  bind(tp::int8_2);
  bind(tp::int8_3);
  bind(tp::int8_4);
  
  bind(tp::int16_1);
  bind(tp::int16_2);
  bind(tp::int16_3);
  bind(tp::int16_4);
  
  bind(tp::int32_1);
  bind(tp::int32_2);
  bind(tp::int32_3);
  bind(tp::int32_4);
  
  bind(tp::int64_1);
  bind(tp::int64_2);
  bind(tp::int64_3);
  bind(tp::int64_4);
  
  bind(tp::uint8_1);
  bind(tp::uint8_2);
  bind(tp::uint8_3);
  bind(tp::uint8_4);
  
  bind(tp::uint16_1);
  bind(tp::uint16_2);
  bind(tp::uint16_3);
  bind(tp::uint16_4);
  
  bind(tp::uint32_1);
  bind(tp::uint32_2);
  bind(tp::uint32_3);
  bind(tp::uint32_4);
  
  bind(tp::uint64_1);
  bind(tp::uint64_2);
  bind(tp::uint64_3);
  bind(tp::uint64_4);
  
  bind(tp::float32_1);
  bind(tp::float32_2);
  bind(tp::float32_3);
  bind(tp::float32_4);
  
  bind(tp::float64_1);
  bind(tp::float64_2);
  bind(tp::float64_3);
  bind(tp::float64_4);
  
  //bind(tp::float128_1);
  //bind(tp::float128_2);
  //bind(tp::float128_3);
  //bind(tp::float128_4);
  
  bind(tp::complex64_1);
  bind(tp::complex64_2);
  bind(tp::complex64_3);
  bind(tp::complex64_4);
  
  bind(tp::complex128_1);
  bind(tp::complex128_2);
  bind(tp::complex128_3);
  bind(tp::complex128_4);
  
  //bind(tp::complex256_1);
  //bind(tp::complex256_2);
  //bind(tp::complex256_3);
  //bind(tp::complex256_4);
}
