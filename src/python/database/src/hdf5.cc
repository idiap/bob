/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 14 Apr 09:41:47 2011 
 *
 * @brief Binds our C++ HDF5 interface to python 
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>

#include "core/python/exception.h"

#include "database/HDF5File.h"

using namespace boost::python;
namespace core = Torch::core::python;
namespace db = Torch::database;

/**
 * Allows us to write HDF5File("filename.hdf5", "rb")
 */
static boost::shared_ptr<db::HDF5File>
hdf5file_make_fromstr(const std::string& filename, const std::string& opmode) {
  static const char* help = "Supported flags are 'r' (read-only), 'w' (read/append), 't' (read/write/truncate) or 'x' (exclusive)";
  if (opmode.size() > 1) {
    PyErr_SetString(PyExc_RuntimeError, help);
    throw_error_already_set();
  }
  db::HDF5File::mode_t mode = db::HDF5File::inout;
  if (opmode[0] == 'r') mode = db::HDF5File::in;
  else if (opmode[0] == 'w') mode = db::HDF5File::inout;
  else if (opmode[0] == 't') mode = db::HDF5File::trunc;
  else if (opmode[0] == 'x') mode = db::HDF5File::excl;
  else { //anything else is just unsupported for the time being
    PyErr_SetString(PyExc_RuntimeError, help);
    throw_error_already_set();
  }
  return boost::make_shared<db::HDF5File>(filename, mode);
}

/**
 * Allows us to write HDF5File("filename.hdf5") and open that file for input
 * and output (without truncation)
 */
static boost::shared_ptr<db::HDF5File>
hdf5file_make_readwrite(const std::string& filename) {
  return boost::make_shared<db::HDF5File>(filename, db::HDF5File::inout);
}

/**
 * Transforms the shape input into a tuple
 */
static tuple hdf5type_shape(const db::HDF5Type& t) {
  const db::HDF5Shape& shape = t.shape();
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
static list hdf5file_paths(const db::HDF5File& f) {
  list retval;
  std::vector<std::string> values;
  f.paths(values);
  for (size_t i=0; i<values.size(); ++i) retval.append(str(values[i]));
  return retval;
}

/**
 * Functionality to read from HDF5File's
 */
template <typename T> T hdf5file_read_scalar(db::HDF5File& f, const std::string& p, size_t pos) {
  T tmp;
  f.read(p, pos, tmp);
  return tmp;
}

void bind_database_hdf5() {

  //the exceptions that can be thrown are catchable in python
  core::CxxToPythonTranslator<db::HDF5Exception, Torch::database::Exception>("HDF5Exception", "Generic exception, should never be raised or used, here just as a general catcher.");
  core::CxxToPythonTranslatorPar<db::HDF5InvalidFileAccessModeError, db::HDF5Exception, const unsigned int>("HDF5InvalidFileAccessModeError", "Thrown when the user tries to open an HDF5 file with a mode that is not supported by the HDF5 library");
  core::CxxToPythonTranslator<db::HDF5UnsupportedCxxTypeError, db::HDF5Exception>("HDF5UnsupportedCxxTypeError", "Thrown when we don't support the input type that we got from our API.");
  core::CxxToPythonTranslatorPar<db::HDF5UnsupportedTypeError, db::HDF5Exception, const boost::shared_ptr<hid_t>&>("HDF5UnsupportedTypeError", "Thrown when we don't support a type that was read from the input file.");
  core::CxxToPythonTranslatorPar<db::HDF5UnsupportedDimensionError, db::HDF5Exception, size_t>("HDF5UnsupportedDimensionError", "Thrown when the user tries to read/write using an unsupported number of dimensions from arrays.");
  core::CxxToPythonTranslatorPar2<db::HDF5InvalidPath, db::HDF5Exception, const std::string&, const std::string&>("HDF5InvalidPath", "This exception is raised when the user asks for a particular path (i.e. 'group' in HDF5 jargon) that does not exist in the file.");
  core::CxxToPythonTranslatorPar2<db::HDF5StatusError, db::HDF5Exception, const std::string&, herr_t>("HDF5StatusError", "This exception is raised when we call the HDF5 C-API and that returns less than zero as a status output.");
  core::CxxToPythonTranslatorPar4<db::HDF5IndexError, db::HDF5Exception, const std::string&, const std::string&, size_t, size_t>("HDF5IndexError", "This exception is raised when the user asks for a certain array in an array list that is out of bounds.");
  core::CxxToPythonTranslatorPar4<db::HDF5IncompatibleIO, db::HDF5Exception, const std::string&, const std::string&, const std::string&, const std::string&>("HDF5IncompatibleIO", "This exception is raised when the user tries to read or write to or from an existing dataset using a type that is incompatible with the established one for that dataset.");
  core::CxxToPythonTranslatorPar2<db::HDF5NotExpandible, db::HDF5Exception, const std::string&, const std::string&>("HDF5NotExpandible", "This exception is raised when the user tries to append to a certain dataset that is not expandible.");

  //this class describes an HDF5 type
  class_<db::HDF5Type, boost::shared_ptr<db::HDF5Type> >("HDF5Type", "Support to compare data types, convert types into runtime equivalents and make our life easier when deciding what to input and output.", no_init)
    .def("__eq__", &db::HDF5Type::operator==)
    .def("__ne__", &db::HDF5Type::operator!=)
#   define DECLARE_SUPPORT(T) .def("compatible", &db::HDF5Type::compatible<T>, (arg("self"), arg("value")), "Tests compatibility of this type against a given scalar")
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
#   define DECLARE_SUPPORT(T,N) .def("compatible", &db::HDF5Type::compatible<blitz::Array<T,N> >, (arg("self"), arg("value")), "Tests compatibility of this type against a given array")

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
    .def("is_array", &db::HDF5Type::is_array, (arg("self")), "Tests if this type is an array")
    .def("__str__", &db::HDF5Type::str)
    .def("shape", &hdf5type_shape, (arg("self")), "Returns the shape of the elements described by this type")
    .def("type_str", &db::HDF5Type::type_str, (arg("self")), "Returns a stringified representation of the base element type")
    .def("element_type", &db::HDF5Type::element_type, (arg("self")), "Returns a representation of the element type one of the Torch supported element types.")
    ;

  //this is the main class
  class_<db::HDF5File, boost::shared_ptr<db::HDF5File>, boost::noncopyable>("HDF5File", "A HDF5File allows users to read and write data from and to files containing standard Torch binary coded data in HDF5 format. For an introduction to HDF5, please visit http://www.hdfgroup.org/HDF5.", no_init)
    .def("__init__", make_constructor(hdf5file_make_fromstr, default_call_policies(), (arg("filename"), arg("openmode_string"))), "Opens a new file in one of these supported modes: 'r' (read-only), 'w' (read/write/append), 't' (read/write/truncate) or 'x' (read/write/exclusive)")
    .def("__init__", make_constructor(hdf5file_make_readwrite, default_call_policies(), (arg("filename"))), "Opens a new HDF5File for reading and writing.")
    .def("cd", &db::HDF5File::cd, (arg("self"), arg("path")), "Changes the current prefix path. When this object is started, the prefix path is empty, which means all following paths to data objects should be given using the full path. If you set this to a different value, it will be used as a prefix to any subsequent operation until you reset it. If path starts with '/', it is treated as an absolute path. '..' and '.' are supported. This object should be a std::string. If the value is relative, it is added to the current path. If it is absolute, it causes the prefix to be reset. Note all operations taking a relative path, following a cd(), will be considered relative to the value defined by the 'cwd' property of this object.")
    .add_property("cwd", make_function(&db::HDF5File::cwd, return_value_policy<copy_const_reference>()), &db::HDF5File::cd)
    .def("__contains__", &db::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("has_key", &db::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("describe", &db::HDF5File::describe, return_value_policy<copy_const_reference>(), (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, return a type description of objects recorded in such a dataset, otherwise, raises an exception.")
    .def("size", &db::HDF5File::size, (arg("self"), arg("key")), "Returns the number of objects stored in this dataset")
    .def("unlink", &db::HDF5File::unlink, (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, unlinks it. Please note this will note remove the data from the file, just make it inaccessible. If you wish to cleanup, save the reacheable objects from this file to another HDF5File object using copy(), for example.")
    .def("rename", &db::HDF5File::rename, (arg("self"), arg("from"), arg("to")), "If a given path to an HDF5 dataset exists in the file, rename it")
    .def("keys", &hdf5file_paths, (arg("self")), "Returns all paths to datasets available inside this file")
    .def("paths", &hdf5file_paths, (arg("self")), "Returns all paths to datasets available inside this file")
    .def("copy", &db::HDF5File::copy, (arg("self"), arg("file")), "Copies all accessible content to another HDF5 file")
#   define DECLARE_SUPPORT(T,E) \
    .def(BOOST_PP_STRINGIZE(__read_ ## E ## __), &hdf5file_read_scalar<T>, (arg("self"), arg("key"), arg("pos")), "Reads a given scalar from a dataset") \
    .def(BOOST_PP_STRINGIZE(__replace_ ## E ## __), &db::HDF5File::replace<T>, (arg("self"), arg("key"), arg("pos"), arg("value")), "Modifies the value of a scalar inside the file.") \
    .def(BOOST_PP_STRINGIZE(__append_ ## E ## __), &db::HDF5File::append<T>, (arg("self"), arg("key"), arg("value")), "Appends a scalar to a dataset. If the dataset does not yet exist, one is created with the type characteristics.") 
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
#   define DECLARE_SUPPORT(T,N) .def("__read_array__", &db::HDF5File::readArray<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("pos"), arg("array")), "Reads a given array from a dataset") \
    .def("__replace_array__", &db::HDF5File::replaceArray<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("pos"), arg("array")), "Modifies the value of a array inside the file.") \
    .def("__append_array__", &db::HDF5File::appendArray<blitz::Array<T,N> >, (arg("self"), arg("key"), arg("array")), "Appends a array to a dataset. If the dataset does not yet exist, one is created with the type characteristics.") 
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
