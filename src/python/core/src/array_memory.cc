/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 21:00:57 2011 
 *
 * @brief Array operations that modify size or copy its data 
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

namespace tp = Torch::python;
namespace bp = boost::python;

template <typename T, int N>
static blitz::Array<T,N> sameAs(const blitz::Array<T,N>& original) {
	return blitz::Array<T,N>(original.shape());
}

template <typename T, int N>
static void bind_memory_common (tp::array<T,N>& array) {
  typedef typename tp::array<T,N>::array_type array_type;
  typedef typename tp::array<T,N>::shape_type shape_type;

  array.object()->def("free", &array_type::free, "This method resizes an array to zero size. If the array data is not being shared with another array object, then it is freed.");
  array.object()->def("resize", (void (array_type::*)(const shape_type&))(&array_type::resize), boost::python::arg("shape"), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
  array.object()->def("resizeAndPreserve", (void (array_type::*)(const shape_type&))(&array_type::resizeAndPreserve), boost::python::arg("shape"), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
  array.object()->def("copy", &array_type::copy, "This method creates a copy of the array's data, using the same storage ordering as the current array. The returned array is guaranteed to be stored contiguously in memory, and to be the only object referring to its memory block (i.e. the data isn't shared with any other array object).");
  array.object()->def("sameAs", &sameAs<T,N>, "This method creates a new array with the same basic type and shape of the current array. The returned array is guaranteed to be stored contiguously in memory, and to be the only object referring to its memory block (i.e. the data isn't shared with any other array object).");
  array.object()->def("makeUnique", &array_type::makeUnique, "If the array's data is being shared with another Blitz++ array object, this member function creates a copy so the array object has a unique view of the data.");
}


template <typename T>
static void bind_memory_1 (tp::array<T,1>& array) {
  typedef typename tp::array<T,1>::array_type array_type;
  typedef typename tp::array<T,1>::shape_type shape_type;

  bind_memory_common(array);

  array.object()->def("resize", (void (array_type::*)(int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
  //bogus blitz implementation:
  //array.object()->def("resizeAndPreserve", (void (array_type::*)(int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
}

template <typename T>
static void bind_memory_2 (tp::array<T,2>& array) {
  typedef typename tp::array<T,2>::array_type array_type;
  typedef typename tp::array<T,2>::shape_type shape_type;

  bind_memory_common(array);

  array.object()->def("resize", (void (array_type::*)(int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
  //bogus blitz implementation:
  //array.object()->def("resizeAndPreserve", (void (array_type::*)(int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
}

template <typename T>
static void bind_memory_3 (tp::array<T,3>& array) {
  typedef typename tp::array<T,3>::array_type array_type;
  typedef typename tp::array<T,3>::shape_type shape_type;

  bind_memory_common(array);

  array.object()->def("resize", (void (array_type::*)(int,int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
  //bogus blitz implementation:
  //array.object()->def("resizeAndPreserve", (void (array_type::*)(int,int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
}

template <typename T>
static void bind_memory_4 (tp::array<T,4>& array) {
  typedef typename tp::array<T,4>::array_type array_type;
  typedef typename tp::array<T,4>::shape_type shape_type;

  bind_memory_common(array);

  array.object()->def("resize", (void (array_type::*)(int,int,int,int))(&array_type::resize), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "If the array is already the size specified, then no memory is allocated. After resizing, the contents of the array are garbage. See also resizeAndPreserve().");
  //bogus blitz implementation:
  //array.object()->def("resizeAndPreserve", (void (array_type::*)(int,int,int,int))(&array_type::resizeAndPreserve), (boost::python::arg("self"), boost::python::arg("dim0"), boost::python::arg("dim1"), boost::python::arg("dim2"), boost::python::arg("dim3")), "If the array is already the size specified, then no change occurs (the array is not reallocated and copied). The contents of the array are preserved whenever possible; if the new array size is smaller, then some data will be lost. Any new elements created by resizing the array are left uninitialized.");
}

void bind_array_memory () {
  bind_memory_1(tp::bool_1);
  bind_memory_1(tp::int8_1);
  bind_memory_1(tp::int16_1);
  bind_memory_1(tp::int32_1);
  bind_memory_1(tp::int64_1);
  bind_memory_1(tp::uint8_1);
  bind_memory_1(tp::uint16_1);
  bind_memory_1(tp::uint32_1);
  bind_memory_1(tp::uint64_1);
  bind_memory_1(tp::float32_1);
  bind_memory_1(tp::float64_1);
  //bind_memory_1(tp::float128_1);
  bind_memory_1(tp::complex64_1);
  bind_memory_1(tp::complex128_1);
  //bind_memory_1(tp::complex256_1);
 
  bind_memory_2(tp::bool_2);
  bind_memory_2(tp::int8_2);
  bind_memory_2(tp::int16_2);
  bind_memory_2(tp::int32_2);
  bind_memory_2(tp::int64_2);
  bind_memory_2(tp::uint8_2);
  bind_memory_2(tp::uint16_2);
  bind_memory_2(tp::uint32_2);
  bind_memory_2(tp::uint64_2);
  bind_memory_2(tp::float32_2);
  bind_memory_2(tp::float64_2);
  //bind_memory_2(tp::float128_2);
  bind_memory_2(tp::complex64_2);
  bind_memory_2(tp::complex128_2);
  //bind_memory_2(tp::complex256_2);
 
  bind_memory_3(tp::bool_3);
  bind_memory_3(tp::int8_3);
  bind_memory_3(tp::int16_3);
  bind_memory_3(tp::int32_3);
  bind_memory_3(tp::int64_3);
  bind_memory_3(tp::uint8_3);
  bind_memory_3(tp::uint16_3);
  bind_memory_3(tp::uint32_3);
  bind_memory_3(tp::uint64_3);
  bind_memory_3(tp::float32_3);
  bind_memory_3(tp::float64_3);
  //bind_memory_3(tp::float128_3);
  bind_memory_3(tp::complex64_3);
  bind_memory_3(tp::complex128_3);
  //bind_memory_3(tp::complex256_3);
 
  bind_memory_4(tp::bool_4);
  bind_memory_4(tp::int8_4);
  bind_memory_4(tp::int16_4);
  bind_memory_4(tp::int32_4);
  bind_memory_4(tp::int64_4);
  bind_memory_4(tp::uint8_4);
  bind_memory_4(tp::uint16_4);
  bind_memory_4(tp::uint32_4);
  bind_memory_4(tp::uint64_4);
  bind_memory_4(tp::float32_4);
  bind_memory_4(tp::float64_4);
  //bind_memory_4(tp::float128_4);
  bind_memory_4(tp::complex64_4);
  bind_memory_4(tp::complex128_4);
  //bind_memory_4(tp::complex256_4);
}
