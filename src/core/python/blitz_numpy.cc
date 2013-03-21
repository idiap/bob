/**
 * @file python/blitz_numpy.cc
 * @date Mon Sep 26 11:47:30 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Automatic converters to-from python for blitz::Array's
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <bob/core/python/ndarray.h>

template<typename T, int N>
void npy_copy_cast(blitz::Array<T,N>& bz, PyArrayObject* arrobj) {
  PYTHON_ERROR(TypeError, "unsupported number of dimensions: %d", N);
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
    boost::python::converter::registry::push_back(&convertible, &construct, 
        boost::python::type_id<array_type>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * a Array<T,N>
   */
  static void* convertible(PyObject* obj_ptr) {
    boost::python::handle<> hdl(boost::python::borrowed(boost::python::allow_null(obj_ptr)));
    boost::python::object obj(hdl);

    bob::core::array::typeinfo tinfo(bob::core::array::getElementType<T>(), N);

    bob::python::convert_t result = bob::python::convertible_to(obj, tinfo, false, true);

    // we cannot afford copying, only referencing.
    if (result == bob::python::BYREFERENCE) return obj_ptr;

    // but, if the user passed an array of the right type, but we still need to
    // copy, warn the user as this is a tricky case to debug.
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(obj_ptr);
    if (result == bob::python::WITHARRAYCOPY && 
        bob::python::ctype_to_num<T>() == arr->descr->type_num) {
      PYTHON_ERROR(RuntimeError, "The bindings you are trying to use to this C++ method require a numpy.ndarray -> blitz::Array<%s,%d> conversion, but the array you passed, despite the correct type, is not C-style contiguous and/or properly aligned, so I cannot automatically wrap it. You can check this by yourself by printing the flags on such a variable with the command 'print(<varname>.flags)'. The only way to circumvent this problem, from python, is to create a copy the variable by issuing '<varname>.copy()' before calling the bound method. Otherwise, if you wish the copy to be executed automatically, you have to re-bind the method to use our custom 'const_ndarray' type.", bob::core::array::stringize<T>(), N);
    }

    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the blitz::Array<> storage area
    void* storage = ((boost::python::converter::rvalue_from_python_storage<array_type>*)data)->storage.bytes;

    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(obj_ptr);
    
    //mounts the numpy memory at the "newly allocated" blitz::Array
    shape_type shape;
    shape_type stride;
    for (int k=0; k<N; ++k) {
      shape[k] = arr->dimensions[k];
      stride[k] = (arr->strides[k]/sizeof(T));
    }
    new (storage) array_type((T*)arr->data, shape, stride,
        blitz::neverDeleteData); //place operator
    data->convertible = storage;

  }

};

/**
 * Avoids the big number of warnings...
 */
static PyArrayObject* make_pyarray(int nd, npy_intp* dims, int type) {
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

    PyArrayObject* retval = make_pyarray(N, dims, bob::python::ctype_to_num<T>());

    //wrap new PyArray in a blitz layer and then copy the data
    shape_type shape=0;
    for (int k=0; k<retval->nd; ++k) shape[k] = retval->dimensions[k];
    shape_type stride=0;
    for (int k=0; k<retval->nd; ++k) stride[k] = (retval->strides[k]/sizeof(T));
    array_type bzdest((T*)retval->data, shape, stride, blitz::neverDeleteData);
    bzdest = tv;

    return reinterpret_cast<PyObject*>(retval);
  }

  static const PyTypeObject* get_pytype() { return &PyArray_Type; }

};

template <typename T, int N>
void register_bz_to_npy() {
  boost::python::to_python_converter<typename blitz::Array<T,N>, bz_to_npy<T,N>
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_core_bz_numpy () {
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
#  define BOOST_PP_LOCAL_LIMITS (1, BOB_MAX_DIM)
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
#  define BOOST_PP_LOCAL_LIMITS (1, BOB_MAX_DIM)
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
