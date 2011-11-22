/**
 * @file python/core/src/ndarray_numpy.cc
 * @date Thu Nov 17 14:33:20 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Automatic converters to-from python for tp::ndarray
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"

namespace bp = boost::python;
namespace tp = Torch::python;
 
/**
 * Objects of this type create a binding between tp::ndarray and
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
struct ndarray_from_npy {

  /**
   * Registers converter from numpy array into a tp::ndarray
   */
  ndarray_from_npy() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<tp::ndarray>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * an ndarray. To do that, the object has to be of type PyArrayObject
   */
  static void* convertible(PyObject* obj_ptr) {
    if (PyArray_Check(obj_ptr)) return obj_ptr;
    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the tp::ndarray storage area
    void* storage = ((bp::converter::rvalue_from_python_storage<tp::ndarray>*)data)->storage.bytes;

    new (storage) tp::ndarray(tp::make_non_null_borrowed_object(obj_ptr));
    data->convertible = storage;

  }

};

/**
 * Objects of this type bind tp::ndarray's to numpy arrays. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a Numpy array.
 */
struct ndarray_to_npy {

  static PyObject* convert(const tp::ndarray& tv) {
    return bp::incref(const_cast<tp::ndarray*>(&tv)->self().ptr());
  }

  static const PyTypeObject* get_pytype() { return &PyArray_Type; }

};

void register_ndarray_to_npy() {
  bp::to_python_converter<tp::ndarray, ndarray_to_npy
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

/**
 * The same as for ndarray_from_npy, but bindings the const specialization. The
 * difference is that we don't require that the object given as input to be,
 * strictly, a NumPy ndarray, but are more relaxed.
 */
struct const_ndarray_from_npy {

  /**
   * Registers converter from numpy array into a tp::ndarray
   */
  const_ndarray_from_npy() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<tp::const_ndarray>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * an ndarray. To do that, the object has to convertible to a NumPy ndarray.
   */
  static void* convertible(PyObject* obj_ptr) {
    bp::object obj = tp::make_non_null_borrowed_object(obj_ptr);
    if (tp::convertible_to(obj, false, true)) //writeable=false, behaved=true
      return obj_ptr;
    return 0;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {

    //black-magic required to setup the tp::ndarray storage area
    void* storage = ((bp::converter::rvalue_from_python_storage<tp::const_ndarray>*)data)->storage.bytes;

    new (storage) tp::const_ndarray(tp::make_non_null_borrowed_object(obj_ptr));
    data->convertible = storage;

  }

};

/**
 * Objects of this type bind tp::ndarray's to numpy arrays. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a Numpy array.
 */
struct const_ndarray_to_npy {

  static PyObject* convert(const tp::const_ndarray& tv) {
    return bp::incref(const_cast<tp::const_ndarray*>(&tv)->self().ptr());
  }

  static const PyTypeObject* get_pytype() { return &PyArray_Type; }

};

void register_const_ndarray_to_npy() {
  bp::to_python_converter<tp::const_ndarray, const_ndarray_to_npy
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_core_ndarray_numpy () {
   ndarray_from_npy();
   register_ndarray_to_npy();
   const_ndarray_from_npy();
   register_const_ndarray_to_npy();
}
