/**
 * @file python/core/src/typeinfo.cc
 * @date Tue Nov 8 18:35:46 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds ca::typeinfo
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

static object typeinfo_dtype (const ca::typeinfo& info) {
  return tp::dtype(info.dtype).self();
}

static tuple ti_shape(const ca::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.shape[i]);
  return tuple(retval);
}

static tuple ti_stride(const ca::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.stride[i]);
  return tuple(retval);
}


void bind_core_typeinfo() {
  
  class_<ca::typeinfo>("typeinfo", "Type information for bob C++ data", 
      no_init)
    .add_property("dtype", &typeinfo_dtype)
    .def_readonly("cxxtype", &ca::typeinfo::dtype)
    .def_readonly("nd", &ca::typeinfo::nd)
    .add_property("shape", &ti_shape)
    .add_property("stride", &ti_stride)
    .def("__str__", &ca::typeinfo::str)
    ;

}
