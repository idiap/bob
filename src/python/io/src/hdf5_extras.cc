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
    case 5:
      return make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4]);
    default:
      break;
  }
  return make_tuple();
}

void bind_io_hdf5_extras() {

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
    .def("shape", &hdf5type_shape, (arg("self")), "Returns the shape of the elements described by this type")
    .def("type_str", &io::HDF5Type::type_str, (arg("self")), "Returns a stringified representation of the base element type")
    .def("element_type", &io::HDF5Type::element_type, (arg("self")), "Returns a representation of the element type one of the Torch supported element types.")
    ;

  //defines the descriptions returned by HDF5File::describe()
  class_<io::HDF5Descriptor, boost::shared_ptr<io::HDF5Descriptor> >("HDF5Descriptor", "A dataset descriptor describes one of the possible ways to read a dataset", no_init)
    .def_readonly("type", &io::HDF5Descriptor::type)
    .def_readonly("size", &io::HDF5Descriptor::size)
    .def_readonly("expandable", &io::HDF5Descriptor::expandable)
    ;
}
