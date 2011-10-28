/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::io::Arrayset
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>

#include "io/Arrayset.h"
#include "core/array_assert.h"
#include "core/python/vector.h"
#include "core/python/exception.h"

#include "io/python/pyio.h"

using namespace boost::python;
namespace io = Torch::io;
namespace core = Torch::core;
namespace array = Torch::core::array;
namespace tp = Torch::python;

static void extend_with_iterable1(io::Arrayset& s, object iter) {
  stl_input_iterator<object> end;
  for (stl_input_iterator<object> it(iter); it != end; ++it) {
    s.add(io::Array(boost::make_shared<tp::npyarray>(*it, object())));
  }
}

static void extend_with_iterable2(io::Arrayset& s, object iter, object dtype) {
  stl_input_iterator<object> end;
  for (stl_input_iterator<object> it(iter); it != end; ++it) {
    s.add(io::Array(boost::make_shared<tp::npyarray>(*it, dtype)));
  }
}

static boost::shared_ptr<io::Arrayset> make_from_array_iterable1(object iter) {
  boost::shared_ptr<io::Arrayset> retval = boost::make_shared<io::Arrayset>();
  extend_with_iterable1(*retval, iter);
  return retval;
}

static boost::shared_ptr<io::Arrayset> make_from_array_iterable2(object iter,
    object dtype) {
  boost::shared_ptr<io::Arrayset> retval = boost::make_shared<io::Arrayset>();
  extend_with_iterable2(*retval, iter, dtype);
  return retval;
}

static io::Array get_array(io::Arrayset& s, size_t index) {
  return s[index];
}

static void set_array(io::Arrayset& s, size_t index, const io::Array& a) {
  s.set(index, a);
}

static void append_array(io::Arrayset& s, const io::Array& a) {
  s.add(a);
}

static object get_ndarray(io::Arrayset& s, size_t index) {
  return tp::buffer_object(s[index].get());
}

static void set_ndarray1(io::Arrayset& s, size_t index, object o) {
  s.set(index, io::Array(tp::npyarray(o, object())));
}

static void set_ndarray2(io::Arrayset& s, size_t index, object o, object dtype) {
  s.set(index, io::Array(tp::npyarray(o, dtype)));
}

static void set_string(io::Arrayset& s, size_t index, const std::string& filename) {
  s.set(index, io::Array(filename));
}

static void append_ndarray1(io::Arrayset& s, object o) {
  s.add(io::Array(boost::make_shared<tp::npyarray>(o, object())));
}

static void append_ndarray2(io::Arrayset& s, object o, object dtype) {
  s.add(io::Array(boost::make_shared<tp::npyarray>(o, dtype)));
}

static void append_string(io::Arrayset& s, const std::string& filename) {
  s.add(io::Array(filename));
}

void bind_io_arrayset() {
  class_<io::Arrayset, boost::shared_ptr<io::Arrayset> >("Arrayset", "Arraysets represent lists of Arrays that share the same element type and dimension properties.", init<const std::string&>((arg("filename")), "Initializes a new arrayset from an external file."))
    .def(init<>("Creates a new empty arraset with an inlined representation."))
    .def(init<boost::shared_ptr<io::File>, optional<size_t,size_t> >((arg("file"), arg("begin"), arg("end")), "Start with all or some arrays in a given file. You can select the start and/or the end. Numbers past the end of the given file are ignored. For example, if a file contains 5 arrays, this constructor will work ok if you leave 'end' on its default (maximum possible unsigned integer)."))
    .def("__init__", make_constructor(make_from_array_iterable1, default_call_policies(), (arg("iterable"))), "Creates a new Arrayset from a python iterable containing array-like objects.")
    .def("__init__", make_constructor(make_from_array_iterable2, default_call_policies()), "Creates a new Arrayset from a python iterable containing array-like objects, with an optional coertion dtype that will be applied to all arrays.")
    .add_property("type", make_function(&io::Arrayset::type, return_value_policy<copy_const_reference>()), "Typing information for this arrayset")
    .def("save", &io::Arrayset::save, (arg("self"), arg("filename")), "Saves the arrayset into a single file. The file is truncated before this operation.")
    .def("load", &io::Arrayset::load, "Loads all Arrays into memory, if that is not the case")
    .def("get", &get_array, (arg("self"), arg("index")), "Gets an array from this set given its id")
    
    //some list-like entries
    .def("__len__", &io::Arrayset::size, "The number of arrays stored in this set.")
    .def("__getitem__", &get_ndarray, (arg("self"), arg("index")), "Gets a numpy array given its relative position")
    .def("__setitem__", &set_ndarray1, (arg("self"), arg("index"), arg("array")), "Sets an array-like in the set. May raise an exception if there is no such index in the set.")
    .def("__setitem__", &set_ndarray2, (arg("self"), arg("index"), arg("array")), "Sets an array-like in the set. May raise an exception if there is no such index in the set. This variant allows you to define a coertion type that will be applied to the object before storing it.")
    .def("__setitem__", &set_string, (arg("self"), arg("index"), arg("filename")), "Sets an array encoded in a file at the set. May raise an exception if there is no such index in the set.")
    .def("__setitem__", &set_array, (arg("self"), arg("index"), arg("array")), "Sets an array in the set. May raise an exception if there is no such index in the set.")
    .def("__delitem__", &io::Arrayset::remove, (arg("self"), arg("index")), "Removes the array given its id. May raise an exception if there is no such array inside.")
    .def("append", &append_array, (arg("self"), arg("array")), "Adds an array to the set")
    .def("append", &append_ndarray1, (arg("self"), arg("array")), "Appends an array-like object to the set.")
    .def("append", &append_ndarray2, (arg("self"), arg("array"), arg("dtype")), "Appends an array-like object to the set. This version allows you to set a coertion type to be applied to the input array.")
    .def("append", &append_string, (arg("self"), arg("filename")), "Appends an array encoded in a file to the set")
    .def("extend", &extend_with_iterable1, (arg("self"), arg("iterable")), "Extends this array set with array-like objects coming from the given iterable")
    .def("extend", &extend_with_iterable2, (arg("self"), arg("iterable"), arg("dtype")), "Extends this array set with array-like objects coming from the given iterable. This variant offers a type coertion parameter you can set to force type casting on the input data.")
    //.def("extend") //TODO: extension with other Arrays/NumPy Arrays
    ;

  tp::vector_no_compare<io::Arrayset>("ArraysetVector");
}
