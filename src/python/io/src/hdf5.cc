/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 14 Apr 09:41:47 2011 
 *
 * @brief Binds our C++ HDF5 interface to python 
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>

#include "core/python/exception.h"

#include "io/HDF5File.h"

using namespace boost::python;
namespace core = Torch::core::python;
namespace io = Torch::io;

/**
 * Allows us to write HDF5File("filename.hdf5", "rb")
 */
static boost::shared_ptr<io::HDF5File>
hdf5file_make_fromstr(const std::string& filename, const std::string& opmode) {
  static const char* help = "Supported flags are 'r' (read-only), 'w' (read/append), 't' (read/write/truncate) or 'x' (exclusive)";
  if (opmode.size() > 1) {
    PyErr_SetString(PyExc_RuntimeError, help);
    throw_error_already_set();
  }
  io::HDF5File::mode_t mode = io::HDF5File::inout;
  if (opmode[0] == 'r') mode = io::HDF5File::in;
  else if (opmode[0] == 'w') mode = io::HDF5File::inout;
  else if (opmode[0] == 't') mode = io::HDF5File::trunc;
  else if (opmode[0] == 'x') mode = io::HDF5File::excl;
  else { //anything else is just unsupported for the time being
    PyErr_SetString(PyExc_RuntimeError, help);
    throw_error_already_set();
  }
  return boost::make_shared<io::HDF5File>(filename, mode);
}

/**
 * Allows us to write HDF5File("filename.hdf5") and open that file for input
 * and output (without truncation)
 */
static boost::shared_ptr<io::HDF5File>
hdf5file_make_readwrite(const std::string& filename) {
  return boost::make_shared<io::HDF5File>(filename, io::HDF5File::inout);
}

/**
 * Transforms the shape input into a tuple
 */
static tuple hdf5type_shape(const io::HDF5Type& t) {
  const io::HDF5Shape& shape = t.shape();
  switch (shape.n()) {
    case 1:
      return make_tuple(shape[0]);
    case 2:
      return make_tuple(shape[0], shape[1]);
    case 3:
      return make_tuple(shape[0], shape[1], shape[2]);
    case 4:
      return make_tuple(shape[0], shape[1], shape[2], shape[3]);
    default:
      break;
  }
  return make_tuple();
}

/**
 * Returns a list of all paths inside a HDF5File
 */
static list hdf5file_paths(const io::HDF5File& f) {
  list retval;
  std::vector<std::string> values;
  f.paths(values);
  for (size_t i=0; i<values.size(); ++i) retval.append(str(values[i]));
  return retval;
}

/**
 * Functionality to read from or replace at HDF5File's
 */
template <typename T> static T hdf5file_read_scalar(io::HDF5File& f, const std::string& p, size_t pos) {
  T tmp;
  f.read(p, pos, tmp);
  return tmp;
}

template <typename T> static void hdf5file_replace_scalar(io::HDF5File& f, const std::string& p, size_t pos, const T& value) {
  f.replace(p, pos, value);
}

template <typename T> static void hdf5file_read_array(io::HDF5File& f, const std::string& p, size_t pos, T& value) {
  f.readArray(p, pos, value);
}

template <typename T> static void hdf5file_replace_array(io::HDF5File& f, const std::string& p, size_t pos, const T& value) {
  f.replaceArray(p, pos, value);
}

template <typename T> static void hdf5file_append_array(io::HDF5File& f, 
    const std::string& path, const T& value, size_t compression) {
  f.appendArray(path, value, compression);
}

template <typename T> static void hdf5file_set_array(io::HDF5File& f, 
    const std::string& path, const T& value, size_t compression) {
  f.setArray(path, value, compression);
}

void bind_io_hdf5() {

  //specific exceptions that require special bindings
  core::register_exception_translator<io::HDF5Exception>(PyExc_RuntimeError);
  core::register_exception_translator<io::HDF5UnsupportedCxxTypeError>(PyExc_TypeError);
  core::register_exception_translator<io::HDF5UnsupportedTypeError>(PyExc_TypeError);
  core::register_exception_translator<io::HDF5IndexError>(PyExc_IndexError);
  core::register_exception_translator<io::HDF5IncompatibleIO>(PyExc_IOError);

  //this class describes an HDF5 type
  class_<io::HDF5Type, boost::shared_ptr<io::HDF5Type> >("HDF5Type", "Support to compare data types, convert types into runtime equivalents and make our life easier when deciding what to input and output.", no_init)
    .def("__eq__", &io::HDF5Type::operator==)
    .def("__ne__", &io::HDF5Type::operator!=)
#   define DECLARE_SUPPORT(T) .def("compatible", &io::HDF5Type::compatible<T>, (arg("self"), arg("value")), "Tests compatibility of this type against a given scalar")
    DECLARE_SUPPORT(bool)
    DECLARE_SUPPORT(int8_t)
    DECLARE_SUPPORT(int16_t)
    DECLARE_SUPPORT(int32_t)
    DECLARE_SUPPORT(int64_t)
    DECLARE_SUPPORT(uint8_t)
    DECLARE_SUPPORT(uint16_t)
    DECLARE_SUPPORT(uint32_t)
    DECLARE_SUPPORT(uint64_t)
    DECLARE_SUPPORT(float)
    DECLARE_SUPPORT(double)
    //DECLARE_SUPPORT(long double)
    DECLARE_SUPPORT(std::complex<float>)
    DECLARE_SUPPORT(std::complex<double>)
    //DECLARE_SUPPORT(std::complex<long double>)
    DECLARE_SUPPORT(std::string)
#   undef DECLARE_SUPPORT
#   define DECLARE_SUPPORT(T,N) .def("compatible", &io::HDF5Type::compatible<blitz::Array<T,N> >, (arg("self"), arg("value")), "Tests compatibility of this type against a given array")

#   define DECLARE_BZ_SUPPORT(T) \
    DECLARE_SUPPORT(T,1) \
    DECLARE_SUPPORT(T,2) \
    DECLARE_SUPPORT(T,3) \
    DECLARE_SUPPORT(T,4)
    DECLARE_BZ_SUPPORT(bool)
    DECLARE_BZ_SUPPORT(int8_t)
    DECLARE_BZ_SUPPORT(int16_t)
    DECLARE_BZ_SUPPORT(int32_t)
    DECLARE_BZ_SUPPORT(int64_t)
    DECLARE_BZ_SUPPORT(uint8_t)
    DECLARE_BZ_SUPPORT(uint16_t)
    DECLARE_BZ_SUPPORT(uint32_t)
    DECLARE_BZ_SUPPORT(uint64_t)
    DECLARE_BZ_SUPPORT(float)
    DECLARE_BZ_SUPPORT(double)
    //DECLARE_BZ_SUPPORT(long double)
    DECLARE_BZ_SUPPORT(std::complex<float>)
    DECLARE_BZ_SUPPORT(std::complex<double>)
    //DECLARE_BZ_SUPPORT(std::complex<long double>)
#   undef DECLARE_BZ_SUPPORT
#   undef DECLARE_SUPPORT
    .def("is_array", &io::HDF5Type::is_array, (arg("self")), "Tests if this type is an array")
    .def("__str__", &io::HDF5Type::str)
    .def("shape", &hdf5type_shape, (arg("self")), "Returns the shape of the elements described by this type")
    .def("type_str", &io::HDF5Type::type_str, (arg("self")), "Returns a stringified representation of the base element type")
    .def("element_type", &io::HDF5Type::element_type, (arg("self")), "Returns a representation of the element type one of the Torch supported element types.")
    ;

  //this is the main class
  class_<io::HDF5File, boost::shared_ptr<io::HDF5File>, boost::noncopyable>("HDF5File", "A HDF5File allows users to read and write data from and to files containing standard Torch binary coded data in HDF5 format. For an introduction to HDF5, please visit http://www.hdfgroup.org/HDF5.", no_init)
    .def("__init__", make_constructor(hdf5file_make_fromstr, default_call_policies(), (arg("filename"), arg("openmode_string"))), "Opens a new file in one of these supported modes: 'r' (read-only), 'w' (read/write/append), 't' (read/write/truncate) or 'x' (read/write/exclusive)")
    .def("__init__", make_constructor(hdf5file_make_readwrite, default_call_policies(), (arg("filename"))), "Opens a new HDF5File for reading and writing.")
    .def("cd", &io::HDF5File::cd, (arg("self"), arg("path")), "Changes the current prefix path. When this object is started, the prefix path is empty, which means all following paths to data objects should be given using the full path. If you set this to a different value, it will be used as a prefix to any subsequent operation until you reset it. If path starts with '/', it is treated as an absolute path. '..' and '.' are supported. This object should be a std::string. If the value is relative, it is added to the current path. If it is absolute, it causes the prefix to be reset. Note all operations taking a relative path, following a cd(), will be considered relative to the value defined by the 'cwd' property of this object.")
    .add_property("cwd", make_function(&io::HDF5File::cwd, return_value_policy<copy_const_reference>()), &io::HDF5File::cd)
    .def("__contains__", &io::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("has_key", &io::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("describe", &io::HDF5File::describe, return_value_policy<copy_const_reference>(), (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, return a type description of objects recorded in such a dataset, otherwise, raises an exception.")
    .def("size", &io::HDF5File::size, (arg("self"), arg("key")), "Returns the number of objects stored in this dataset")
    .def("unlink", &io::HDF5File::unlink, (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, unlinks it. Please note this will note remove the data from the file, just make it inaccessible. If you wish to cleanup, save the reacheable objects from this file to another HDF5File object using copy(), for example.")
    .def("rename", &io::HDF5File::rename, (arg("self"), arg("from"), arg("to")), "If a given path to an HDF5 dataset exists in the file, rename it")
    .def("keys", &hdf5file_paths, (arg("self")), "Returns all paths to datasets available inside this file")
    .def("paths", &hdf5file_paths, (arg("self")), "Returns all paths to datasets available inside this file")
    .def("copy", &io::HDF5File::copy, (arg("self"), arg("file")), "Copies all accessible content to another HDF5 file")
#   define DECLARE_SUPPORT(T,E) \
    .def(BOOST_PP_STRINGIZE(__read_ ## E ## __), &hdf5file_read_scalar<T>, (arg("self"), arg("key"), arg("pos")), "Reads a given scalar from a dataset") \
    .def(BOOST_PP_STRINGIZE(__replace_ ## E ## __), &hdf5file_replace_scalar<T>, (arg("self"), arg("key"), arg("pos"), arg("value")), "Modifies the value of a scalar inside the file.") \
    .def(BOOST_PP_STRINGIZE(__append_ ## E ## __), &io::HDF5File::append<T>, (arg("self"), arg("key"), arg("value")), "Appends a scalar to a dataset. If the dataset does not yet exist, one is created with the type characteristics.") \
    .def(BOOST_PP_STRINGIZE(__set_ ## E ## __), &io::HDF5File::set<T>, (arg("self"), arg("key"), arg("value")), "Sets the scalar at position 0 to the given value. This method is equivalent to checking if the scalar at position 0 exists and then replacing it. If the path does not exist, we append the new scalar.") 
    DECLARE_SUPPORT(bool, bool)
    DECLARE_SUPPORT(int8_t, int8)
    DECLARE_SUPPORT(int16_t, int16)
    DECLARE_SUPPORT(int32_t, int32)
    DECLARE_SUPPORT(int64_t, int64)
    DECLARE_SUPPORT(uint8_t, uint8)
    DECLARE_SUPPORT(uint16_t, uint16)
    DECLARE_SUPPORT(uint32_t, uint32)
    DECLARE_SUPPORT(uint64_t, uint64)
    DECLARE_SUPPORT(float, float32)
    DECLARE_SUPPORT(double, float64)
    //DECLARE_SUPPORT(long double, float128)
    DECLARE_SUPPORT(std::complex<float>, complex64)
    DECLARE_SUPPORT(std::complex<double>, complex128)
    //DECLARE_SUPPORT(std::complex<long double>, complex256)
    DECLARE_SUPPORT(std::string, string)
#   undef DECLARE_SUPPORT
#   define DECLARE_SUPPORT(T,N) .def("__read_array__", &hdf5file_read_array<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("pos"), arg("array")), "Reads a given array from a dataset") \
    .def("__replace_array__", &hdf5file_replace_array<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("pos"), arg("array")), "Modifies the value of a array inside the file.") \
    .def("__append_array__", &hdf5file_append_array<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("array")), "Appends a array to a dataset. If the dataset does not yet exist, one is created with the type characteristics.\n\nIf a new Dataset is to be created, you can also set the compression level. Note this setting has no effect if the Dataset already exists on file, in which case the current setting for that dataset is respected. The maximum value for the gzip compression is 9. The value of zero turns compression off (the default).") \
    .def("__set_array__", &hdf5file_set_array<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("array")), "Sets the array at position 0 to the given value. This method is equivalent to checking if the array at position 0 exists and then replacing it. If the path does not exist, we append the new array.\n\nIf a new Dataset is to be created, you can also set the compression level. Note this setting has no effect if the Dataset already exists on file, in which case the current setting for that dataset is respected. The maximum value for the gzip compression is 9. The value of zero turns compression off (the default).") 
#   define DECLARE_BZ_SUPPORT(T) \
    DECLARE_SUPPORT(T,1) \
    DECLARE_SUPPORT(T,2) \
    DECLARE_SUPPORT(T,3) \
    DECLARE_SUPPORT(T,4)
    DECLARE_BZ_SUPPORT(bool)
    DECLARE_BZ_SUPPORT(int8_t)
    DECLARE_BZ_SUPPORT(int16_t)
    DECLARE_BZ_SUPPORT(int32_t)
    DECLARE_BZ_SUPPORT(int64_t)
    DECLARE_BZ_SUPPORT(uint8_t)
    DECLARE_BZ_SUPPORT(uint16_t)
    DECLARE_BZ_SUPPORT(uint32_t)
    DECLARE_BZ_SUPPORT(uint64_t)
    DECLARE_BZ_SUPPORT(float)
    DECLARE_BZ_SUPPORT(double)
    //DECLARE_BZ_SUPPORT(long double)
    DECLARE_BZ_SUPPORT(std::complex<float>)
    DECLARE_BZ_SUPPORT(std::complex<double>)
    //DECLARE_BZ_SUPPORT(std::complex<long double>)
#   undef DECLARE_BZ_SUPPORT
#   undef DECLARE_SUPPORT
    ;
}
