/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sat 24 Sep 05:01:54 2011 CEST
 *
 * @brief Automatic converters to-from python for blitz::Array's
 */

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <dlfcn.h>

#include <blitz/array.h>
#include <stdint.h>

#include <boost/preprocessor.hpp>

#include "core/array_type.h"
#include "core/python/exception.h"
#include "core/python/TypeMapper.h"

namespace bp = boost::python;
namespace tp = Torch::python;

template<typename T, int N>
void npy_copy_cast(blitz::Array<T,N>& bz, PyArrayObject* arrobj) {
  PYTHON_ERROR(TypeError, "Unsupported number of dimensions"); 
}

template<typename T>
static void npy_copy_cast(blitz::Array<T,1>& bz, PyArrayObject* arrobj) {
  for (int i=0; i<PyArray_DIM(arrobj,0); ++i)
    bz(i) = *static_cast<T*>(PyArray_GETPTR1(arrobj, i));
}

template<typename T>
static void npy_copy_cast(blitz::Array<T,2>& bz, PyArrayObject* arrobj) {
  for (int i=0; i<PyArray_DIM(arrobj,0); ++i)
    for (int j=0; j<PyArray_DIM(arrobj,1); ++j)
      bz(i,j) = *static_cast<T*>(PyArray_GETPTR2(arrobj, i, j));
}

template<typename T>
static void npy_copy_cast(blitz::Array<T,3>& bz, PyArrayObject* arrobj) {
  for (int i=0; i<PyArray_DIM(arrobj,0); ++i)
    for (int j=0; j<PyArray_DIM(arrobj,1); ++j)
      for (int k=0; k<PyArray_DIM(arrobj,2); ++k)
        bz(i,j,k) = *static_cast<T*>(PyArray_GETPTR3(arrobj, i, j, k));
}

template<typename T>
static void npy_copy_cast(blitz::Array<T,4>& bz, PyArrayObject* arrobj) {
  for (int i=0; i<PyArray_DIM(arrobj,0); ++i)
    for (int j=0; j<PyArray_DIM(arrobj,1); ++j)
      for (int k=0; k<PyArray_DIM(arrobj,2); ++k)
        for (int l=0; l<PyArray_DIM(arrobj,3); ++l)
          bz(i,j,k,l) = *static_cast<T*>(PyArray_GETPTR4(arrobj, i, j, k, l));
}

/**
 * Objects of this type create a binding between blitz::Array<T,N> and
 * NumPy arrays. You can specify a NumPy array as a parameter to a
 * bound method that would normally receive a blitz::Array<T,N> or a const
 * blitz::Array<T,N>& and the conversion will just magically happen, as
 * efficiently as possible.
 *
 * Please note that passing by value should be avoided as much as possible. In
 * this mode, the underlying method will still be able to alter the underlying
 * array storage area w/o being able to modify the array itself, causing a
 * gigantic mess. If you want to make something close to pass-by-value, just
 * pass by non-const reference instead.
 */
template <typename T, int N> struct bz_from_npy {

  typedef typename blitz::Array<T,N> array_type;
  typedef typename blitz::TinyVector<int,N> shape_type;

  /**
   * Registers converter from numpy array into a blitz::Array<T,N>
   */
  bz_from_npy() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<array_type>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * a Array<T,N>
   */
  static void* convertible(PyObject* obj_ptr) {
    if (!PyArray_Check(obj_ptr)) return 0; //has to be a numpy.ndarray
    PyArrayObject* arrobj = reinterpret_cast<PyArrayObject*>(obj_ptr);
    if (arrobj->nd == N     //has to have the number of dimensions == N
        && tp::TYPEMAP.type_to_enum<T>() == arrobj->descr->type_num //good type
        && arrobj->descr->byteorder == '=' //native byte order only accepted!
        ) {
      return obj_ptr;
    }
    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the blitz::Array<> storage area
    void* storage = ((boost::python::converter::rvalue_from_python_storage<array_type>*)data)->storage.bytes;

    //now we proceed to the conversion.

    /**
     * We need to tackle the following only use-case:
     *    The NumPy array type T is the right type for the conversion, in which
     *    case we just take a pointer to the data and instantiate a
     *    blitz::Array<T,N> out of that
     *
     * We cannot support automatic casting because this will not work well with
     * overloaded methods. The user has to cast manually
     */
    PyArrayObject* arrobj = reinterpret_cast<PyArrayObject*>(obj_ptr);
    shape_type shape;
    for (int k=0; k<arrobj->nd; ++k) shape[k] = arrobj->dimensions[k];

    //perfect match, just grab the memory area -- fastest option
    shape_type stride;
    for (int k=0; k<arrobj->nd; ++k) stride[k] = (arrobj->strides[k] / 4);
    new (storage) array_type((T*)arrobj->data, 
        shape, stride, blitz::neverDeleteData); //place operator
    data->convertible = storage;

  }

};

PyArrayObject* make_ndarray(int nd, npy_intp* dims, int type) {
  return (PyArrayObject*)PyArray_SimpleNew(nd, dims, type);
}

/**
 * Objects of this type bind blitz::Array<T,N> to numpy arrays. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a Numpy array.
 */
template <typename T, int N> struct bz_to_npy {

  typedef typename blitz::Array<T,N> array_type;
  typedef typename blitz::TinyVector<int,N> shape_type;

  static PyObject* convert(const array_type& tv) {
    npy_intp dims[N];
    for (int i=0; i<N; ++i) dims[i] = tv.extent(i);
    PyArrayObject* retval = make_ndarray(N, dims, tp::TYPEMAP.type_to_enum<T>());

    //wrap new PyArray in a blitz layer and then copy the data
    shape_type shape;
    for (int k=0; k<retval->nd; ++k) shape[k] = retval->dimensions[k];
    shape_type stride;
    for (int k=0; k<retval->nd; ++k) stride[k] = retval->strides[k];
    array_type bzdest((T*)retval->data, shape, stride, blitz::neverDeleteData);
    bzdest = tv;

    return reinterpret_cast<PyObject*>(retval);
  }

  static const PyTypeObject* get_pytype() { return &PyArray_Type; }

};

template <typename T, int N>
void register_bz_to_npy() {
  bp::to_python_converter<typename blitz::Array<T,N>, bz_to_npy<T,N>
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_core_bz_numpy () {
  // Gets the current dlopenflags and save it
  PyThreadState *tstate = PyThreadState_GET();
  if(!tstate)
    throw std::runtime_error("Can not get python dlopenflags.");
  int old_value = tstate->interp->dlopenflags;

  // Unsets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value & (~RTLD_GLOBAL);
  // Loads numpy with the RTLD_GLOBAL flag unset
  import_array();
  // Resets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value;

  //Sets the boost::python::numeric::array interface to use numpy.ndarray
  //as basis. This is not strictly required, but good to set as a baseline.
  bp::numeric::array::set_module_and_type("numpy", "ndarray");

  /**
   * The following struct constructors will make sure we can input
   * blitz::Array<T,N> in our bound C++ routines w/o needing to specify
   * special converters each time. The rvalue converters allow boost::python to
   * automatically map the following inputs:
   *
   * a) const blitz::Array<T,N>& (pass by const reference)
   * b) blitz::Array<T,N> (pass by value -- DO NEVER DO THIS!!!)
   *
   * Please note that the last case:
   * 
   * c) blitz::Array<T,N>& (pass by non-const reference)
   *
   * is NOT covered by these converters. The reason being that because the
   * object may be changed, there is no way for boost::python to update the
   * original python object, in a sensible manner, at the return of the method.
   *
   * Avoid passing by non-const reference in your methods.
   */
#  define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
#  define BOOST_PP_LOCAL_MACRO(D) \
   bz_from_npy<bool,D>();\
   bz_from_npy<int8_t,D>();\
   bz_from_npy<int16_t,D>();\
   bz_from_npy<int32_t,D>();\
   bz_from_npy<int64_t,D>();\
   bz_from_npy<uint8_t,D>();\
   bz_from_npy<uint16_t,D>();\
   bz_from_npy<uint32_t,D>();\
   bz_from_npy<uint64_t,D>();\
   bz_from_npy<float,D>();\
   bz_from_npy<double,D>();\
   bz_from_npy<long double,D>();\
   bz_from_npy<std::complex<float>,D>();\
   bz_from_npy<std::complex<double>,D>();\
   bz_from_npy<std::complex<long double>,D>();
#  include BOOST_PP_LOCAL_ITERATE()
  
  /**
   * The following struct constructors will make C++ return values of type
   * blitz::Array<T,N> to show up in the python side as numpy arrays.
   */
#  define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
#  define BOOST_PP_LOCAL_MACRO(D) \
   register_bz_to_npy<bool,D>();\
   register_bz_to_npy<int8_t,D>();\
   register_bz_to_npy<int16_t,D>();\
   register_bz_to_npy<int32_t,D>();\
   register_bz_to_npy<int64_t,D>();\
   register_bz_to_npy<uint8_t,D>();\
   register_bz_to_npy<uint16_t,D>();\
   register_bz_to_npy<uint32_t,D>();\
   register_bz_to_npy<uint64_t,D>();\
   register_bz_to_npy<float,D>();\
   register_bz_to_npy<double,D>();\
   register_bz_to_npy<long double,D>();\
   register_bz_to_npy<std::complex<float>,D>();\
   register_bz_to_npy<std::complex<double>,D>();\
   register_bz_to_npy<std::complex<long double>,D>();
#  include BOOST_PP_LOCAL_ITERATE()

}
