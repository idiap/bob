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

template<typename T>
static boost::shared_ptr<io::Arrayset> make_from_array_iterable(T iter) {
  boost::shared_ptr<io::Arrayset> retval = boost::make_shared<io::Arrayset>();
  stl_input_iterator<numeric::array> end;
  for (stl_input_iterator<numeric::array> it(iter); it != end; ++it) {
    retval->add(io::Array(boost::make_shared<tp::npyarray>(*it)));
  }
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

static void set_ndarray(io::Arrayset& s, size_t index, numeric::array a) {
  s.set(index, io::Array(tp::npyarray(a)));
}

static void set_string(io::Arrayset& s, size_t index, const std::string& filename) {
  s.set(index, io::Array(filename));
}

static void append_ndarray(io::Arrayset& s, numeric::array a) {
  s.add(io::Array(tp::npyarray(a)));
}

static void append_string(io::Arrayset& s, const std::string& filename) {
  s.add(io::Array(filename));
}

void bind_io_arrayset() {
  class_<io::Arrayset, boost::shared_ptr<io::Arrayset> >("Arrayset", "Arraysets represent lists of Arrays that share the same element type and dimension properties.", init<const std::string&>((arg("filename")), "Initializes a new arrayset from an external file."))
    .def(init<>("Creates a new empty arraset with an inlined representation."))
    .def(init<boost::shared_ptr<io::File>, optional<size_t,size_t> >((arg("file"), arg("begin"), arg("end")), "Start with all or some arrays in a given file. You can select the start and/or the end. Numbers past the end of the given file are ignored. For example, if a file contains 5 arrays, this constructor will work ok if you leave 'end' on its default (maximum possible unsigned integer)."))
    .def("__init__", make_constructor(make_from_array_iterable<tuple>), "Creates a new Arrayset from a python tuple containing numpy arrays.")
    .def("__init__", make_constructor(make_from_array_iterable<list>), "Creates a new Arrayset from a python list containing numpy arrays.")
    .add_property("type", make_function(&io::Arrayset::type, return_value_policy<copy_const_reference>()), "Typing information for this arrayset")
    .def("save", &io::Arrayset::save, (arg("self"), arg("filename")), "Saves the arrayset into a single file. The file is truncated before this operation.")
    .def("load", &io::Arrayset::load, "Loads all Arrays into memory, if that is not the case")
    .def("get", &get_array, (arg("self"), arg("index")), "Gets an array from this set given its id")
    
    //some list-like entries
    .def("__len__", &io::Arrayset::size, "The number of arrays stored in this set.")
    .def("__getitem__", &get_ndarray, (arg("self"), arg("index")), "Gets a numpy array given its relative position")
    .def("__setitem__", &set_ndarray, (arg("self"), arg("index"), arg("array")), "Sets an array in the set. May raise an exception if there is no such index in the set.")
    .def("__setitem__", &set_ndarray, (arg("self"), arg("index"), arg("filename")), "Sets an array encoded in a file at the set. May raise an exception if there is no such index in the set.")
    .def("__setitem__", &set_array, (arg("self"), arg("index"), arg("array")), "Sets an array in the set. May raise an exception if there is no such index in the set.")
    .def("__delitem__", &io::Arrayset::remove, (arg("self"), arg("index")), "Removes the array given its id. May raise an exception if there is no such array inside.")
    .def("append", &append_array, (arg("self"), arg("array")), "Adds an array to the set")
    .def("append", &append_ndarray, (arg("self"), arg("array")), "Appends a numpy ndarray to the set")
    .def("append", &append_string, (arg("self"), arg("filename")), "Appends an array encoded in a file to the set")

    //.def("extend") //TODO
    ;

  tp::vector_no_compare<io::Arrayset>("ArraysetVector");
}
