/**
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 * @brief Array operations to create a gray version of color images
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

namespace tp = Torch::python;
namespace bp = boost::python;

template <typename T>
blitz::Array<T,1> vectorOf(const blitz::Array<T,2>& original) {
	blitz::Array<T,1> dst(original.numElements());

	typename blitz::Array<T,2>::const_iterator src_it = original.begin();
	typename blitz::Array<T,1>::iterator       dst_it = dst.begin();

	for (; src_it != original.end(); )
		*dst_it++ = *src_it++;

	return dst;

	/*
	  boost::shared_ptr<blitz::Array<T,N> >retval(new blitz::Array<T,N>(shape, storage));
	  typename blitz::Array<T,N>::iterator j(retval->begin());
	  bp::handle<> obj_iter(PyObject_GetIter(o.ptr()));
	  for(Py_ssize_t i=0; i<length;++i,++j) {
	  bp::handle<> py_elem_hdl(
	  bp::allow_null(PyIter_Next(obj_iter.get())));
	  if (PyErr_Occurred()) {
	  PyErr_Clear();
	  boost::format s("element %d is not accessible?");
	  s % i;
	  PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
	  bp::throw_error_already_set();
	  }
	  if (!py_elem_hdl.get()) break; // end of iteration
	  bp::object py_elem_obj(py_elem_hdl);
	  (*j) = bp::extract<T>(py_elem_obj);
	  }
	  
	  return retval;
	*/
}

template <typename T>
static void create_vector_of(tp::array<T,2>&array) {
	array.object()->def("vectorOf", &vectorOf<T>, "This method will take a 2d blitz array and turn it into a 1d blitz array.");
}

template <typename T>
blitz::Array<T,2> grayAs(const blitz::Array<T,3>& original) {
	// WARNING: ignore the color dim / planes (original.extent(0))
	return blitz::Array<T,2>(original.extent(1), original.extent(2));
}

template <typename T> 
static void create_gray_as(tp::array<T,3>&array) {
	array.object()->def("grayAs", &grayAs<T>, "This method creates a new array with the same basic type and shape of the current array. Except it donegrades 3D (color) to 2D (gray). The returned array is guaranteed to be stored contiguously in memory, and to be the only object referring to its memory block (i.e. the data isn't shared with any other array object).");
}

void bind_gray_as()
{
  create_gray_as(tp::bool_3);
  create_gray_as(tp::int8_3);
  create_gray_as(tp::int16_3);
  create_gray_as(tp::int32_3);
  create_gray_as(tp::int64_3);
  create_gray_as(tp::uint8_3);
  create_gray_as(tp::uint16_3);
  create_gray_as(tp::uint32_3);
  create_gray_as(tp::uint64_3);
  create_gray_as(tp::float32_3);
  create_gray_as(tp::float64_3);
  create_gray_as(tp::complex64_3);
  create_gray_as(tp::complex128_3);
}

void bind_vector_of()
{
  create_vector_of(tp::bool_2);
  create_vector_of(tp::int8_2);
  create_vector_of(tp::int16_2);
  create_vector_of(tp::int32_2);
  create_vector_of(tp::int64_2);
  create_vector_of(tp::uint8_2);
  create_vector_of(tp::uint16_2);
  create_vector_of(tp::uint32_2);
  create_vector_of(tp::uint64_2);
  create_vector_of(tp::float32_2);
  create_vector_of(tp::float64_2);
  create_vector_of(tp::complex64_2);
  create_vector_of(tp::complex128_2);
}

