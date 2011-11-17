/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Python bindings for torch::io::Array 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "io/Array.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace io = Torch::io;
namespace core = Torch::core;
namespace ca = Torch::core::array;
namespace tp = Torch::python;

/**
 * Creates a new io::Array from a NumPy ndarray
 */
static boost::shared_ptr<io::Array> array_from_any1(object o) {
  return boost::make_shared<io::Array>(boost::make_shared<tp::py_array>(o, object()));
}

static boost::shared_ptr<io::Array> array_from_any2(object o, object dtype) {
  return boost::make_shared<io::Array>(boost::make_shared<tp::py_array>(o, dtype));
}

/**
 * Wraps a buffer with a NumPy array skin
 */
static object get_array(io::Array& a) {
  return tp::py_array(a.get()).pyobject();
}

static void set_array1(io::Array& a, object o) {
  a.set(boost::make_shared<tp::py_array>(o, object()));
}

static void set_array2(io::Array& a, object o, object d) {
  a.set(boost::make_shared<tp::py_array>(o, d));
}

void bind_io_array() {

  enum_<ca::ElementType>("ElementType")
    .value(ca::stringize(ca::t_unknown), ca::t_unknown)
    .value(ca::stringize(ca::t_bool), ca::t_bool)
    .value(ca::stringize(ca::t_int8), ca::t_int8)
    .value(ca::stringize(ca::t_int16), ca::t_int16)
    .value(ca::stringize(ca::t_int32), ca::t_int32)
    .value(ca::stringize(ca::t_int64), ca::t_int64)
    .value(ca::stringize(ca::t_uint8), ca::t_uint8)
    .value(ca::stringize(ca::t_uint16), ca::t_uint16)
    .value(ca::stringize(ca::t_uint32), ca::t_uint32)
    .value(ca::stringize(ca::t_uint64), ca::t_uint64)
    .value(ca::stringize(ca::t_float32), ca::t_float32)
    .value(ca::stringize(ca::t_float64), ca::t_float64)
    .value(ca::stringize(ca::t_float128), ca::t_float128)
    .value(ca::stringize(ca::t_complex64), ca::t_complex64)
    .value(ca::stringize(ca::t_complex128), ca::t_complex128)
    .value(ca::stringize(ca::t_complex256), ca::t_complex256)
    ;

  //base class declaration
  class_<io::Array, boost::shared_ptr<io::Array> >("Array", "Arrays represent pointers to concrete data loaded or serialized on a file. You can load or refer to real numpy.ndarrays using this type.", no_init)

    //attention: order here is crucial - last defined is tried first by the
    //boost::python overloading resolution system; the last thing we want to
    //try are the numpy handlers.
    .def("__init__", make_constructor(array_from_any1, default_call_policies(), (arg("array"))), "Builds a new Array from an array-like object using a reference to the data, if possible.")
    .def("__init__", make_constructor(array_from_any2, default_call_policies(), (arg("array"), arg("dtype"))), "Builds a new Array from an array-like object with an optional data type coertion specification. References the data, if possible.")
    .def(init<boost::shared_ptr<io::File> >((arg("file")), "Builds a new Array from a file opened with io::open. Reads all file contents."))
    .def(init<boost::shared_ptr<io::File>, size_t>((arg("file"), arg("index")), "Builds a new Array from a specific array in a file opened with io::open"))
    .def(init<const std::string&>((arg("filename")), "Initializes a new array from an external file"))

    //attention: order here is crucial - last defined is tried first by the
    //boost::python overloading resolution system; the last thing we want to
    //try are the numpy handlers.
    .def("set", &set_array1, (arg("self"), arg("array")), "Sets this array with an array-like object. References the data, if possible.")
    .def("set", &set_array2, (arg("self"), arg("array"), arg("dtype")), "Sets this array with an array-like object, with a data type coertion specification. References the data, if possible.")
   
    .def("get", &get_array, (arg("self")), "Retrieves a representation of myself, as an numpy ndarray in the most efficient way possible.")
    .add_property("type", make_function(&io::Array::type, return_value_policy<copy_const_reference>()), "Typing information for this array")
    .def("load", &io::Array::load, "Loads this array into memory, if that is not already the case")
    .add_property("index", &io::Array::getIndex, &io::Array::setIndex)
    .add_property("loadsAll", &io::Array::loadsAll)
    .add_property("filename", make_function(&io::Array::getFilename, return_value_policy<copy_const_reference>()), "Filename -- empty if loaded in memory")
    .add_property("codec", &io::Array::getCodec, "File object being read, if any")
    .def("save", &io::Array::save, (arg("self"), arg("filename")), "Save the array contents to a file, truncating it before")
    ;
}
