/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 27 Sep 11:59:11 2011 
 *
 * @brief Helpers for dealing with numpy.ndarrays and blitz::Array<>'s more
 * transparently
 */

#ifndef TORCH_PYTHON_BLITZ_WRAP_H 
#define TORCH_PYTHON_BLITZ_WRAP_H

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "core/python/ndarray.h"

namespace Torch { namespace python {

  /**
   * Use this method to check wrap-ability for blitz::Array<T,N>
   */
  template <typename T, int N>
  bool can_wrap_as_blitz (boost::python::numeric::array& a,
      bool writeable=true) {

    if (!PyArray_Check(a.ptr())) return false;

    PyArrayObject* arr = (PyArrayObject*)a.ptr();

    if (PyArray_NDIM(arr) != N) return false;

    if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
          arr->descr->elsize == 1)) return false;
        
    if (arr->descr->type_num != ctype_to_num<T>()) return false;

    //tests the following: NPY_ARRAY_C_CONTIGUOUS and NPY_ARRAY_ALIGNED
    if (!PyArray_ISCARRAY_RO(arr)) return false;

    if (writeable && !PyArray_ISWRITEABLE(arr)) return false;

    //if you get to this point, return true
    return true;
  }

  /**
   * Use this method to wrap non-const references to blitz::Array<>'s that will
   * **not** be re-allocated during usage - only element values will be
   * changed. By using this API you can convert any numpy array to a specific
   * blitz::Array<> and we will do our best to convey the information. If the
   * type cannot be converted w/o a cast, we will succeed. Otherwise a
   * TypeError will be raised on your script.
   */
  template <typename T, int N>
  blitz::Array<T,N> temporary_array (boost::python::numeric::array& a,
      bool writeable=true) {

    typedef blitz::Array<T,N> array_type;
    typedef blitz::TinyVector<int,N> shape_type;

    PyArrayObject* arr = (PyArrayObject*)a.ptr();

    if (PyArray_NDIM(arr) != N) {
      boost::format mesg("cannot wrap as blitz::Array<%s,%s>, ndarray %d dimensions.");
      mesg % Torch::core::array::stringize<T>() % N % PyArray_NDIM(arr);
      throw std::runtime_error(mesg.str().c_str());
    }

    if (!(PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE) ||
          arr->descr->elsize == 1)) 
      PYTHON_ERROR(RuntimeError, "can only wrap as blitz::Array<>, ndarrays with native byte-ordering");
        
    if (arr->descr->type_num != ctype_to_num<T>()) {
      boost::format mesg("cannot wrap blitz::Array<%s,%d> with ndarray having elements of type %s");
      mesg % Torch::core::array::stringize<T>() % N;
      mesg % Torch::core::array::stringize(num_to_type(arr->descr->type_num));
      throw std::runtime_error(mesg.str().c_str());
    }

    //tests the following: NPY_ARRAY_C_CONTIGUOUS and NPY_ARRAY_ALIGNED
    if (!PyArray_ISCARRAY_RO(arr)) 
      PYTHON_ERROR(RuntimeError, "can only wrap as blitz::Array<> C-style contiguous and properly aligned ndarray's");

    if (writeable && !PyArray_ISWRITEABLE(arr))
      PYTHON_ERROR(RuntimeError, "input ndarray does not pass write-ability requirement for method");

    shape_type shape;
    for (int k=0; k<arr->nd; ++k) shape[k] = arr->dimensions[k];
    shape_type stride;
    for (int k=0; k<arr->nd; ++k) stride[k] = (arr->strides[k] / sizeof(T));

    //finally, we return the wrapper.
    return array_type((T*)arr->data, shape, stride, blitz::neverDeleteData);

  }

}}

#endif /* TORCH_PYTHON_BLITZ_WRAP_H */
