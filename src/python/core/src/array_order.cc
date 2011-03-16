/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 13 Mar 11:18:05 2011 
 *
 * @brief Several methods to work on the array element ordering 
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

namespace tp = Torch::python;
namespace bp = boost::python;

template <typename T>
static void bind_transpose_2 (tp::array<T,2>& array) {
  typedef typename tp::array<T,2>::array_type array_type;

  array.object()->def("transpose", (array_type (array_type::*)(int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
  array.object()->def("transposeSelf", (void (array_type::*)(int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
}

template <typename T>
static void bind_transpose_3 (tp::array<T,3>& array) {
  typedef typename tp::array<T,3>::array_type array_type;

  array.object()->def("transpose", (array_type (array_type::*)(int,int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
  array.object()->def("transposeSelf", (void (array_type::*)(int,int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
}

template <typename T>
static void bind_transpose_4 (tp::array<T,4>& array) {
  typedef typename tp::array<T,4>::array_type array_type;

  array.object()->def("transpose", (array_type (array_type::*)(int,int,int,int))&array_type::transpose, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
  array.object()->def("transposeSelf", (void (array_type::*)(int,int,int,int))&array_type::transposeSelf, (boost::python::arg("self"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3"), boost::python::arg("dim4")), "The dimensions of the array are reordered so that the first dimension is dim1, the second is dim2, and so on. The arguments should be a permutation of the symbols firstDim, secondDim, .... Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
}

template <typename T, int N>
static void bind_reverse (tp::array<T,N>& array) {
  typedef typename tp::array<T,N>::array_type array_type;

  array.object()->def("reverse", &array_type::reverse, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
  array.object()->def("reverseSelf", &array_type::reverseSelf, (boost::python::arg("self"), boost::python::arg("dimension")), "This method reverses the array in the specified dimension. For example, if reverse(firstDim) is invoked on a 2-dimensional array, then the ordering of rows in the array will be reversed; reverse(secondDim) would reverse the order of the columns. Note that this is implemented by twiddling the strides of the array, and doesn't cause any data copying.");
}

template <typename T> static void bind_order_1 (tp::array<T,1>& array) {
  bind_reverse(array);
}

template <typename T> static void bind_order_2 (tp::array<T,2>& array) {
  bind_reverse(array);
  bind_transpose_2(array);
}

template <typename T> static void bind_order_3 (tp::array<T,3>& array) {
  bind_reverse(array);
  bind_transpose_3(array);
}

template <typename T> static void bind_order_4 (tp::array<T,4>& array) {
  bind_reverse(array);
  bind_transpose_4(array);
}

void bind_array_order () {
  bind_order_1(tp::bool_1);
  bind_order_2(tp::bool_2);
  bind_order_3(tp::bool_3);
  bind_order_4(tp::bool_4);
  
  bind_order_1(tp::int8_1);
  bind_order_2(tp::int8_2);
  bind_order_3(tp::int8_3);
  bind_order_4(tp::int8_4);
  
  bind_order_1(tp::int16_1);
  bind_order_2(tp::int16_2);
  bind_order_3(tp::int16_3);
  bind_order_4(tp::int16_4);
  
  bind_order_1(tp::int32_1);
  bind_order_2(tp::int32_2);
  bind_order_3(tp::int32_3);
  bind_order_4(tp::int32_4);
  
  bind_order_1(tp::int64_1);
  bind_order_2(tp::int64_2);
  bind_order_3(tp::int64_3);
  bind_order_4(tp::int64_4);
  
  bind_order_1(tp::uint8_1);
  bind_order_2(tp::uint8_2);
  bind_order_3(tp::uint8_3);
  bind_order_4(tp::uint8_4);
  
  bind_order_1(tp::uint16_1);
  bind_order_2(tp::uint16_2);
  bind_order_3(tp::uint16_3);
  bind_order_4(tp::uint16_4);
  
  bind_order_1(tp::uint32_1);
  bind_order_2(tp::uint32_2);
  bind_order_3(tp::uint32_3);
  bind_order_4(tp::uint32_4);
  
  bind_order_1(tp::uint64_1);
  bind_order_2(tp::uint64_2);
  bind_order_3(tp::uint64_3);
  bind_order_4(tp::uint64_4);
  
  bind_order_1(tp::float32_1);
  bind_order_2(tp::float32_2);
  bind_order_3(tp::float32_3);
  bind_order_4(tp::float32_4);
  
  bind_order_1(tp::float64_1);
  bind_order_2(tp::float64_2);
  bind_order_3(tp::float64_3);
  bind_order_4(tp::float64_4);
  
  //bind_order_1(tp::float128_1);
  //bind_order_2(tp::float128_2);
  //bind_order_3(tp::float128_3);
  //bind_order_4(tp::float128_4);
  
  bind_order_1(tp::complex64_1);
  bind_order_2(tp::complex64_2);
  bind_order_3(tp::complex64_3);
  bind_order_4(tp::complex64_4);
  
  bind_order_1(tp::complex128_1);
  bind_order_2(tp::complex128_2);
  bind_order_3(tp::complex128_3);
  bind_order_4(tp::complex128_4);
  
  //bind_order_1(tp::complex256_1);
  //bind_order_2(tp::complex256_2);
  //bind_order_3(tp::complex256_3);
  //bind_order_4(tp::complex256_4);
}
