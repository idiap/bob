/**
 * @file python/numpy_scalars.cc
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon  1 Oct 16:29:17 2012 CEST
 *
 * @brief Declares from-python converters to some numpy scalar types
 */

#include <bob/core/python/ndarray.h>

template<typename T> bool checker(PyObject* o) {
  PYTHON_ERROR(TypeError, "Type '%s' is not supported as a numpy scalar to C++ conversion", typeid(T).name());
}

template<> bool checker<bool>(PyObject* o)
{ return PyArray_IsScalar(o, Bool); }
template<> bool checker<int8_t>(PyObject* o)
{ return PyArray_IsScalar(o, Int8); }
template<> bool checker<uint8_t>(PyObject* o)
{ return PyArray_IsScalar(o, UInt8); }
template<> bool checker<int16_t>(PyObject* o)
{ return PyArray_IsScalar(o, Int16); }
template<> bool checker<uint16_t>(PyObject* o)
{ return PyArray_IsScalar(o, UInt16); }
template<> bool checker<int32_t>(PyObject* o)
{ return PyArray_IsScalar(o, Int32); }
template<> bool checker<uint32_t>(PyObject* o)
{ return PyArray_IsScalar(o, UInt32); }
template<> bool checker<int64_t>(PyObject* o)
{ return PyArray_IsScalar(o, Int64); }
template<> bool checker<uint64_t>(PyObject* o)
{ return PyArray_IsScalar(o, UInt64); }
template<> bool checker<float>(PyObject* o)
{ return PyArray_IsScalar(o, Float); }
template<> bool checker<double>(PyObject* o)
{ return PyArray_IsScalar(o, Double); }
template<> bool checker<long double>(PyObject* o)
{ return PyArray_IsScalar(o, LongDouble); }
template<> bool checker<std::complex<float> >(PyObject* o)
{ return PyArray_IsScalar(o, CFloat); }
template<> bool checker<std::complex<double> >(PyObject* o)
{ return PyArray_IsScalar(o, CDouble); }
template<> bool checker<std::complex<long double> >(PyObject* o)
{ return PyArray_IsScalar(o, CLongDouble); }

/**
 * Objects of this type create a binding between numpy scalars and
 * C++ basic scalars.
 */
template<typename T> struct scalar_from_npy {

  /**
   * Registers converter from numpy scalar into a C++ scalar
   */
  scalar_from_npy() {
    boost::python::converter::registry::push_back(&convertible, &construct, 
        boost::python::type_id<T>());
  }

  /**
   * Determines convertibility
   */
  static void* convertible(PyObject* obj_ptr) {
    if (checker<T>(obj_ptr)) return obj_ptr;
    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the bob::python::ndarray storage area
    void* storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
    new (storage) T();
    PyArray_ScalarAsCtype(obj_ptr, storage);
    data->convertible = storage;

  }

};

void bind_core_numpy_scalars () {
  scalar_from_npy<bool>();
  scalar_from_npy<int8_t>();
  scalar_from_npy<int16_t>();
  scalar_from_npy<int32_t>();
  scalar_from_npy<int64_t>();
  scalar_from_npy<uint8_t>();
  scalar_from_npy<uint16_t>();
  scalar_from_npy<uint32_t>();
  scalar_from_npy<uint64_t>();
  scalar_from_npy<float>();
  scalar_from_npy<double>();
  scalar_from_npy<long double>();
  scalar_from_npy<std::complex<float> >();
  scalar_from_npy<std::complex<double> >();
  scalar_from_npy<std::complex<long double> >();
}
