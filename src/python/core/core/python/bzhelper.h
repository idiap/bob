/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 27 Sep 11:59:11 2011 
 *
 * @brief Helpers for dealing with numpy.ndarrays and blitz::Array<>'s more
 * transparently
 */

#ifndef TORCH_CORE_PYTHON_BZHELPER_H 
#define TORCH_CORE_PYTHON_BZHELPER_H

#include <boost/python.hpp>

// Note: Header files that are distributed and include numpy/arrayobject.h need
//       to have these protections. Be warned.
#if !defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#endif
#if !defined(torch_IMPORT_ARRAY) and !defined(NO_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <blitz/array.h>
#include <stdint.h>

#include "core/python/exception.h"

namespace Torch { namespace python {

  /**
   * Conversion from C++ type to Numpy C enum
   */
  template <typename T> int type_to_num(void) {
    PYTHON_ERROR(TypeError, "Unsupported type");
  }

  //implemented converters
  template <> int type_to_num<bool>(void); 
  template <> int type_to_num<signed char>(void);
  template <> int type_to_num<unsigned char>(void); 
  template <> int type_to_num<short>(void);
  template <> int type_to_num<unsigned short>(void); 
  template <> int type_to_num<int>(void);
  template <> int type_to_num<unsigned int>(void); 
  template <> int type_to_num<long>(void);
  template <> int type_to_num<unsigned long>(void);
  template <> int type_to_num<long long>(void);
  template <> int type_to_num<unsigned long long>(void);
  template <> int type_to_num<float>(void);
  template <> int type_to_num<double>(void);
  template <> int type_to_num<long double>(void);
  template <> int type_to_num<std::complex<float> >(void);
  template <> int type_to_num<std::complex<double> >(void);
  template <> int type_to_num<std::complex<long double> >(void);

  template <typename T, int N>
  blitz::Array<T,N> numpy_bz(boost::python::numeric::array& bp_arr) {

    typedef blitz::Array<T,N> array_type;
    typedef blitz::TinyVector<int,N> shape_type;

    PyObject* obj = bp_arr.ptr();
    if (!PyArray_Check(obj)) { //just make sure...
      PYTHON_ERROR(TypeError, "input object is not a numpy.ndarray");
    }

    PyArrayObject* arr = (PyArrayObject*)obj;

    //test exact convertibility -- can only support this mode currently
    if (arr->nd != N) {
      PYTHON_ERROR(TypeError, "wrong input ndarray number of dimensions");
    }

    if (type_to_num<T>() != arr->descr->type_num) {
      PYTHON_ERROR(TypeError, "ndarray dtype mismatch");
    }

    if (!PyArray_EquivByteorders(arr->descr->byteorder, NPY_NATIVE)) {
      PYTHON_ERROR(TypeError, "cannot digest non-native byte order");
    }

    if (!PyArray_ISWRITEABLE(arr)) {
      PYTHON_ERROR(TypeError, "cannot apply blitz layer on const ndarray");
    }

    if (!PyArray_ISBEHAVED(arr)) {
      PYTHON_ERROR(TypeError, "cannot apply blitz layer ndarray that does not behave (search for what this means with PyArray_ISBEHAVED)");
    }
    
    shape_type shape;
    for (int k=0; k<arr->nd; ++k) shape[k] = arr->dimensions[k];
    shape_type stride;
    for (int k=0; k<arr->nd; ++k) stride[k] = (arr->strides[k] / sizeof(T));

    //finally, we return the wrapper.
    return array_type((T*)arr->data, shape, stride, blitz::neverDeleteData);
    
  }

  /**
   * A wrapper for the dtype object in python.
   */
  class dtype {

    public: //api

      dtype(const boost::python::object& name);
      dtype(const dtype& other);

      virtual ~dtype();

      inline int type() const { return _m->type_num; }

    private: //representation

      PyArray_Descr* _m; ///< my description

  };

}}

#endif /* TORCH_CORE_PYTHON_BZHELPER_H */

