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
#include <boost/python/numeric.hpp>
#include <stdexcept>
#include <dlfcn.h>

// ============================================================================
// Note: Header files that are distributed and include numpy/arrayobject.h need
//       to have these protections. Be warned.

// Defines a unique symbol for the API
#if !defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#endif

// Normally, don't import_array(), except if torch_IMPORT_ARRAY is defined.
#if !defined(torch_IMPORT_ARRAY) and !defined(NO_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

// Finally, we include numpy's arrayobject header. Not before!
#include <numpy/arrayobject.h>
// ============================================================================

#include <blitz/array.h>
#include <stdint.h>

#include "core/python/exception.h"
#include "core/array_type.h"

namespace Torch { namespace python {

  /**
   * Initializes numpy and boost bindings. Should be called once per module.
   *
   * Pass to it the module doc string and it will also update the module
   * documentation string.
   */
  void setup_python(const char* module_docstring);

  /**
   * Conversion from Torch element type to Numpy C enum
   */
  int eltype_to_num(Torch::core::array::ElementType eltype);

  Torch::core::array::ElementType num_to_eltype(int num);

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

  //assertions for simple checks
  void assert_ndarray_shape(PyArrayObject* arr, int N);
  void assert_ndarray_type(PyArrayObject* arr, int type_num);
  void assert_ndarray_byteorder(PyArrayObject* arr);
  void assert_ndarray_writeable(PyArrayObject* arr);
  void assert_ndarray_behaved(PyArrayObject* arr);

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
    assert_ndarray_shape(arr, N);
    assert_ndarray_type(arr, type_to_num<T>());
    assert_ndarray_byteorder(arr);
    assert_ndarray_writeable(arr);
    assert_ndarray_behaved(arr);

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

  /**
   * Creates a new NumPy ndarray object with a given number of dimensions,
   * shape and dtype.
   */
  PyArrayObject* make_ndarray(int nd, npy_intp* dims, int type);

  /**
   * Gets the PyArray_Descr* for a given type
   */
  PyArray_Descr* describe_ndarray(int type);

  /**
   * Generates a well behaved copy of an array, starting from any possible
   * python type. The newly generated array will have the given type "dt" and a
   * number of dimensions as specified by "dims".
   */
  PyArrayObject* copy_ndarray(PyObject* any, PyArray_Descr* dt,
      int dims);

  /**
   * Checks convertibility. See the manual page for
   * PyArray_GetArrayParamsFromObject() for details on the return values.
   *
   *  @param any input object
   *  @param req_dtype requested dtype (if need to enforce) otherwise 0
   *  @param writeable check for write-ability
   *  @param dtype output assessement of the object
   *  @param ndim assessed number of dimensions
   *  @param dims assessed shape
   *  @param arr if obj_ptr is ndarray, return it here
   *
   *  @return 0, if it worked ok.
   */
  int check_ndarray(PyObject* any, PyArray_Descr* req_dtype,
      int writeable, PyArray_Descr*& dtype, int& ndim, npy_intp* dims,
      PyArrayObject*& arr);

}}

#endif /* TORCH_CORE_PYTHON_BZHELPER_H */
