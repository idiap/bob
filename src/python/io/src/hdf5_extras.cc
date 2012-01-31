/**
 * @file python/io/src/hdf5_extras.cc
 * @date Thu Aug 11 12:25:19 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds our C++ HDF5 interface to python
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include <boost/python.hpp>
#include <boost/make_shared.hpp>

#include "core/python/ndarray.h"
#include "core/python/exception.h"

#include "io/HDF5File.h"

using namespace boost::python;
namespace tp = bob::python;
namespace io = bob::io;
namespace ca = bob::core::array;

/**
 * Transforms the shape input into a tuple
 */
static tuple hdf5type_shape(const io::HDF5Type& t) {
  const io::HDF5Shape& shape = t.shape();
  list retval;
  for (size_t i=0; i<shape.n(); ++i) retval.append(shape[i]);
  return tuple(retval);
}

static bool hdf5type_compatible(const io::HDF5Type& t, numeric::array a) {
  ca::typeinfo ti;
  tp::typeinfo_ndarray_(a, ti);
  return t.compatible(ti);
}

void bind_io_hdf5_extras() {

  //specific exceptions that require special bindings
  tp::register_exception_translator<io::HDF5Exception>(PyExc_RuntimeError);
  tp::register_exception_translator<io::HDF5UnsupportedCxxTypeError>(PyExc_TypeError);
  tp::register_exception_translator<io::HDF5UnsupportedTypeError>(PyExc_TypeError);
  tp::register_exception_translator<io::HDF5IndexError>(PyExc_IndexError);
  tp::register_exception_translator<io::HDF5IncompatibleIO>(PyExc_IOError);

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
    .def("compatible", &hdf5type_compatible, (arg("self"), arg("array")), "Tests compatibility of this type against a given array")
    .def("shape", &hdf5type_shape, (arg("self")), "Returns the shape of the elements described by this type")
    .def("type_str", &io::HDF5Type::type_str, (arg("self")), "Returns a stringified representation of the base element type")
    .def("element_type", &io::HDF5Type::element_type, (arg("self")), "Returns a representation of the element type one of the bob supported element types.")
    ;

  //defines the descriptions returned by HDF5File::describe()
  class_<io::HDF5Descriptor, boost::shared_ptr<io::HDF5Descriptor> >("HDF5Descriptor", "A dataset descriptor describes one of the possible ways to read a dataset", no_init)
    .def_readonly("type", &io::HDF5Descriptor::type)
    .def_readonly("size", &io::HDF5Descriptor::size)
    .def_readonly("expandable", &io::HDF5Descriptor::expandable)
    ;
}
