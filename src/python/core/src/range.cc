/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Automatic converters to-from python for blitz::Range
 */

#include <boost/python.hpp>
#include <boost/python/slice.hpp>
#include <blitz/range.h>

#include "core/python/array_indexing.h"

namespace tp = Torch::python;
namespace bp = boost::python;

/**
 * Objects of this type create a binding between blitz::Range and python
 * slices. You can specify a python slice as a parameter to a bound method that
 * would normally receive a Range or a const Range& and the conversion will
 * just magically happen.
 */
struct range_from_slice {
  typedef blitz::Range container_type;

  /**
   * Registers converter from a python slice into a blitz::Range
   */
  range_from_slice() {
    bp::converter::registry::push_back(&convertible, &construct, 
        bp::type_id<container_type>());
  }

  /**
   * This method will determine if the input python object is convertible into
   * a blitz::Range
   *
   * Conditions:
   * - Always successful as long the the python object is a slice.
   */
  static void* convertible(PyObject* obj_ptr) {

    /**
     * this bit will check if the input obj is one of the expected input types
     * It will return 0 if the element in question is neither:
     */
    if (!PySlice_Check(obj_ptr)) return 0;
    return obj_ptr;
  }

  /**
   * This method will finally construct the C++ element out of the python
   * object that was input. Please note that when boost::python reaches this
   * method, the object has already been checked for convertibility.
   */
  static void construct(PyObject* obj_ptr,
      bp::converter::rvalue_from_python_stage1_data* data) {
    //access the slice object.
    PySliceObject* slice = (PySliceObject*)obj_ptr;

    //Programming note: Note that I have tried to extract a bp::slice from the
    //input obj_ptr, but that causes memory corruption to the interpreter in
    //interactive shells. I don't understand the reason for that, but casting
    //to PySliceObject and then accessing the individual components of the
    //slice seems to work reliably.
    // OBSERVATION: Problems only appear for indexing 1D arrays and is probably
    // due to some code in the begin of the index method. See
    // array_indexing_1.cc for the code and more comments.

    //prepares the inner Range object (storage) that will be constructed
    void* storage = ((bp::converter::rvalue_from_python_storage<container_type>*)data)->storage.bytes;
    new (storage) container_type();
    data->convertible = storage;
    container_type& result = *((container_type*)storage);

    //retrieves the indexes and sets the Range object

    //the start value may be None
    int start = tp::range::fromStart;
    if (slice->start != Py_None) {
      bp::handle<> handle(slice->start);
      bp::object obj(handle);
      start = bp::extract<int>(obj);
    }

    //the stop value may be None
    int stop = tp::range::toEnd;
    if (slice->stop != Py_None) {
      bp::handle<> handle(slice->stop);
      bp::object obj(handle);
      stop = bp::extract<int>(obj) - 1;
    }

    //the step value may be None
    int step = 1;
    if (slice->step != Py_None) {
      bp::handle<> handle(slice->step);
      bp::object obj(handle);
      step = bp::extract<int>(obj);
    }
    
    result.setRange(start, stop, step);
  }

};

/**
 * Objects of this type bind blitz::Ranges to python slices. Your method
 * generates as output an object of this type and the object will be
 * automatically converted into a python slice.
 */
struct range_to_slice {
  typedef blitz::Range container_type;

  static PyObject* convert(const container_type& tv) {
    static const bp::slice_nil _;
    bp::object retval;
    if (tv.first(tp::range::fromStart) == tp::range::fromStart) {
      if (tv.last(tp::range::toEnd) == tp::range::toEnd) {
        retval = bp::slice(_, _, tv.stride());
      }
      else {
        retval = bp::slice(_, tv.last()+1, tv.stride());
      }
    }
    else {
      if (tv.last(tp::range::toEnd) == tp::range::toEnd) {
        retval = bp::slice(tv.first(), _, tv.stride());
      }
      else {
        retval = bp::slice(tv.first(), tv.last()+1, tv.stride());
      }
    }
    return boost::python::incref(retval.ptr());
  }

  static const PyTypeObject* get_pytype() { return &PySlice_Type; }

};

void register_range_to_slice() {
  bp::to_python_converter<blitz::Range, 
                          range_to_slice
#if defined BOOST_PYTHON_SUPPORTS_PY_SIGNATURES
                          ,true
#endif
              >();
}

void bind_core_array_range () {

  /**
   * The following struct constructors will make sure we can input
   * blitz::Range in our bound C++ routines w/o needing to specify
   * special converters each time. The rvalue converters allow boost::python to
   * automatically map the following inputs:
   *
   * a) const blitz::Range& (pass by const reference)
   * b) blitz::Range (pass by value)
   *
   * Please note that the last case:
   * 
   * c) blitz::Range& (pass by non-const reference)
   *
   * is NOT covered by these converters. The reason being that because the
   * object may be changed, there is no way for boost::python to update the
   * original python object, in a sensible manner, at the return of the method.
   *
   * Avoid passing by non-const reference in your methods.
   */
  range_from_slice();

  /**
   * The following struct constructors will make C++ return values of type
   * blitz::Range to show up in the python side as tuples.
   */
  register_range_to_slice();
}
