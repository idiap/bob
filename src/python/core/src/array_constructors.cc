/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  9 Mar 17:28:42 2011 
 *
 * @brief Implements the constructor bindings for blitz::Array<> 
 */

#include <boost/python.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"
#include "core/python/ndarray.h"

namespace tp = Torch::python;
namespace bp = boost::python;

/**
 * Constructs a new Torch/Blitz array starting from a python sequence. The
 * data from the python object is copied.
 */
template<typename T, int N> static boost::shared_ptr<blitz::Array<T,N> >
iterable_to_blitz (bp::object o, const blitz::TinyVector<int,N>& shape, blitz::GeneralArrayStorage<N> storage) {
  /**
   * Conditions we have to check for:
   * 1. The object "o" has to be iterable
   * 2. The number of elements in o has to be exactly the total defined by
   * the shape.
   * 3. All elements in "o" have to be convertible to T
   */
  bool type_check = PyList_Check(o.ptr()) || PyTuple_Check(o.ptr()) ||
    PyIter_Check(o.ptr());
  if (!type_check) {
    PyErr_SetString(PyExc_TypeError, "input object has to be of type list, tuple or iterable");
    bp::throw_error_already_set();
  }
  /**
   * Check length (item 2)
   */
  Py_ssize_t length = PyObject_Length(o.ptr());
  if (length != blitz::product(shape)) {
    boost::format s("input object does not contain %d elements, but %d");
    s % blitz::product(shape) % length;
    PyErr_SetString(PyExc_RuntimeError, s.str().c_str());
    bp::throw_error_already_set();
  }
  /**
   * This bit will run the filling and will check at the same time
   */
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
}

template<typename T, int N> static boost::shared_ptr<blitz::Array<T,N> >
iterable_to_blitz_c (bp::object o, const blitz::TinyVector<int,N>& shape) {
  return iterable_to_blitz<T,N>(o, shape, blitz::GeneralArrayStorage<N>());
}

template <typename T, int N> static boost::shared_ptr<blitz::Array<T,N> > 
make_blitz(const bp::ndarray& array) {
  return array.to_blitz<T,N>();
}

/**
 * This method will just execute the binding
 */
template <typename T, int N> 
static void bind_constructors (tp::array<T,N>& array) {
  typedef typename tp::array<T,N>::storage_type storage_type;
  typedef typename tp::array<T,N>::array_type array_type;
  typedef typename tp::array<T,N>::shape_type shape_type;

  //intialization with only the storage type
  array.object()->def(bp::init<storage_type>(bp::arg("storage"), "Constructs a new array with a specific storage, but with no contents"));

  //initialization from another array of the same type
  array.object()->def(bp::init<array_type>((bp::arg("other")), "Initializes by referencing the data from another array."));

  //initialization using extents
  switch(N) { 
    case 1: 

      array.object()->def(bp::init<int>((bp::arg("dim0")), "Builds array with the given size"));

      array.object()->def(bp::init<int,storage_type>((bp::arg("dim0"), bp::arg("storage")), "Builds array with the given size and a storage type."));

      break;

    case 2: 
      array.object()->def(bp::init<int,int>((bp::arg("dim0"), bp::arg("dim1")), "Builds array with the given size"));

      array.object()->def(bp::init<int,int,storage_type>((bp::arg("dim0"), bp::arg("dim1"), bp::arg("storage")), "Builds array with the given size and a storage type."));

      break;

    case 3: 
      array.object()->def(bp::init<int, int, int>((bp::arg("dim0"), bp::arg("dim1"), bp::arg("dim2")), "Builds array with the given size"));

      array.object()->def(bp::init<int,int,int,storage_type>((bp::arg("dim0"), bp::arg("dim1"), bp::arg("dim2"), bp::arg("storage")), "Builds array with the given size and a storage type."));

      break;

    case 4: 
      array.object()->def(bp::init<int, int, int, int>((bp::arg("dim0"), bp::arg("dim1"), bp::arg("dim2"), bp::arg("dim3")), "Builds array with the given size"));

      array.object()->def(bp::init<int,int,int,int,storage_type>((bp::arg("dim0"), bp::arg("dim1"), bp::arg("dim2"), bp::arg("dim3"), bp::arg("storage")), "Builds array with the given size and a storage type."));

      break;
  }

  //initialization using a TinyVector<int,T> (bound to tuple)
  array.object()->def(bp::init<const shape_type&>((bp::arg("extent")), "Initalizes the array with extents described in a tuple"));

  array.object()->def(bp::init<const shape_type&,storage_type>((bp::arg("extent"), bp::arg("storage")), "Initalizes the array with extents described in a tuple and a storage type."));

  //initialization from a numpy array or iterable
  array.object()->def("__init__", make_constructor(iterable_to_blitz<T,N>, bp::default_call_policies(), (bp::arg("iterable"), bp::arg("shape"), bp::arg("storage"))), "Builds an array from a python sequence. Please note that the length of the sequence or iterable must be exactly the same as defined by the array shape parameter. You should also specify a storage order (GeneralArrayStorage or FortranArray).");

  array.object()->def("__init__", make_constructor(iterable_to_blitz_c<T,N>, bp::default_call_policies(), (bp::arg("iterable"), bp::arg("shape"))), "Builds an array from a python sequence. Please note that the length of the sequence or iterable must be exactly the same as defined by the array shape parameter. This version will build a C-storage.");

  array.object()->def("__init__", make_constructor(&make_blitz<T,N>, bp::default_call_policies(), (bp::arg("array"))), "Builds an array copying data from a numpy array");
}

void bind_array_constructors () {
  bind_constructors(tp::bool_1);
  bind_constructors(tp::bool_2);
  bind_constructors(tp::bool_3);
  bind_constructors(tp::bool_4);
  
  bind_constructors(tp::int8_1);
  bind_constructors(tp::int8_2);
  bind_constructors(tp::int8_3);
  bind_constructors(tp::int8_4);
  
  bind_constructors(tp::int16_1);
  bind_constructors(tp::int16_2);
  bind_constructors(tp::int16_3);
  bind_constructors(tp::int16_4);
  
  bind_constructors(tp::int32_1);
  bind_constructors(tp::int32_2);
  bind_constructors(tp::int32_3);
  bind_constructors(tp::int32_4);
  
  bind_constructors(tp::int64_1);
  bind_constructors(tp::int64_2);
  bind_constructors(tp::int64_3);
  bind_constructors(tp::int64_4);
  
  bind_constructors(tp::uint8_1);
  bind_constructors(tp::uint8_2);
  bind_constructors(tp::uint8_3);
  bind_constructors(tp::uint8_4);
  
  bind_constructors(tp::uint16_1);
  bind_constructors(tp::uint16_2);
  bind_constructors(tp::uint16_3);
  bind_constructors(tp::uint16_4);
  
  bind_constructors(tp::uint32_1);
  bind_constructors(tp::uint32_2);
  bind_constructors(tp::uint32_3);
  bind_constructors(tp::uint32_4);
  
  bind_constructors(tp::uint64_1);
  bind_constructors(tp::uint64_2);
  bind_constructors(tp::uint64_3);
  bind_constructors(tp::uint64_4);
  
  bind_constructors(tp::float32_1);
  bind_constructors(tp::float32_2);
  bind_constructors(tp::float32_3);
  bind_constructors(tp::float32_4);
  
  bind_constructors(tp::float64_1);
  bind_constructors(tp::float64_2);
  bind_constructors(tp::float64_3);
  bind_constructors(tp::float64_4);
  
  //bind_constructors(tp::float128_1);
  //bind_constructors(tp::float128_2);
  //bind_constructors(tp::float128_3);
  //bind_constructors(tp::float128_4);
  
  bind_constructors(tp::complex64_1);
  bind_constructors(tp::complex64_2);
  bind_constructors(tp::complex64_3);
  bind_constructors(tp::complex64_4);
  
  bind_constructors(tp::complex128_1);
  bind_constructors(tp::complex128_2);
  bind_constructors(tp::complex128_3);
  bind_constructors(tp::complex128_4);
  
  //bind_constructors(tp::complex256_1);
  //bind_constructors(tp::complex256_2);
  //bind_constructors(tp::complex256_3);
  //bind_constructors(tp::complex256_4);
}
