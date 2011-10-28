/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::io::Array 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "io/Array.h"

#include "io/python/pyio.h"

using namespace boost::python;
namespace io = Torch::io;
namespace core = Torch::core;
namespace array = Torch::core::array;
namespace tp = Torch::python;

/**
 * Creates a new io::Array from a NumPy ndarray
 */
static boost::shared_ptr<io::Array> array_from_any1(object o) {
  return boost::make_shared<io::Array>(boost::make_shared<tp::npyarray>(o, object()));
}

static boost::shared_ptr<io::Array> array_from_any2(object o, object dtype) {
  return boost::make_shared<io::Array>(boost::make_shared<tp::npyarray>(o, dtype));
}

/**
 * Wraps a buffer with a NumPy array skin
 */
static object get_array(io::Array& a) {
  return tp::buffer_object(a.get());
}

static void set_array1(io::Array& a, object o) {
  a.set(boost::make_shared<tp::npyarray>(o, object()));
}

static void set_array2(io::Array& a, object o, object dtype) {
  a.set(boost::make_shared<tp::npyarray>(o, dtype));
}

void bind_io_array() {

  enum_<array::ElementType>("ElementType")
    .value(array::stringize(array::t_unknown), array::t_unknown)
    .value(array::stringize(array::t_bool), array::t_bool)
    .value(array::stringize(array::t_int8), array::t_int8)
    .value(array::stringize(array::t_int16), array::t_int16)
    .value(array::stringize(array::t_int32), array::t_int32)
    .value(array::stringize(array::t_int64), array::t_int64)
    .value(array::stringize(array::t_uint8), array::t_uint8)
    .value(array::stringize(array::t_uint16), array::t_uint16)
    .value(array::stringize(array::t_uint32), array::t_uint32)
    .value(array::stringize(array::t_uint64), array::t_uint64)
    .value(array::stringize(array::t_float32), array::t_float32)
    .value(array::stringize(array::t_float64), array::t_float64)
    .value(array::stringize(array::t_float128), array::t_float128)
    .value(array::stringize(array::t_complex64), array::t_complex64)
    .value(array::stringize(array::t_complex128), array::t_complex128)
    .value(array::stringize(array::t_complex256), array::t_complex256)
    ;

  //base class declaration
  class_<io::Array, boost::shared_ptr<io::Array> >("Array", "Arrays represent pointers to concrete data serialized on a file. You can load or refer to real numpy.ndarrays using this type.", init<const std::string&>((arg("filename")), "Initializes a new array from an external file"))
    .def(init<boost::shared_ptr<io::File> >((arg("file")), "Builds a new Array from a file opened with io::open. Reads all file contents."))
    .def(init<boost::shared_ptr<io::File>, size_t>((arg("file"), arg("index")), "Builds a new Array from a specific array in a file opened with io::open"))
    .def("__init__", make_constructor(array_from_any1, default_call_policies(), (arg("array"))), "Builds a new Array from an array-like object using a reference to the data, if possible.")
    .def("__init__", make_constructor(array_from_any2, default_call_policies(), (arg("array"), arg("dtype"))), "Builds a new Array from an array-like object with an optional data type coertion specification. References the data, if possible.")
    .def("get", &get_array, (arg("self")), "Retrieves a representation of myself, as an numpy ndarray in the most efficient way possible.")
    .def("set", &set_array1, (arg("self"), arg("array")), "Sets this array with an array-like object. References the data, if possible.")
    .def("set", &set_array2, (arg("self"), arg("array"), arg("dtype")), "Sets this array with an array-like object, with a data type coertion specification. References the data, if possible.")
    .add_property("type", make_function(&io::Array::type, return_value_policy<copy_const_reference>()), "Typing information for this array")
    .def("load", &io::Array::load, "Loads this array into memory, if that is not already the case")
    .add_property("index", &io::Array::getIndex, &io::Array::setIndex)
    .add_property("loadsAll", &io::Array::loadsAll)
    .add_property("filename", make_function(&io::Array::getFilename, return_value_policy<copy_const_reference>()), "Filename -- empty if loaded in memory")
    .add_property("codec", &io::Array::getCodec, "File object being read, if any")
    .def("save", &io::Array::save, (arg("self"), arg("filename")), "Save the array contents to a file, truncating it before")
    ;
}
