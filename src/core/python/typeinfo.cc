/**
 * @file core/python/typeinfo.cc
 * @date Tue Nov 8 18:35:46 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds bob::core::array::typeinfo
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/python/ndarray.h>

using namespace boost::python;

static object typeinfo_dtype (const bob::core::array::typeinfo& info) {
  return bob::python::dtype(info.dtype).self();
}

static tuple ti_shape(const bob::core::array::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.shape[i]);
  return tuple(retval);
}

static tuple ti_stride(const bob::core::array::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.stride[i]);
  return tuple(retval);
}


void bind_core_typeinfo() {
  
  class_<bob::core::array::typeinfo>("typeinfo", "Type information for bob C++ data", 
      no_init)
    .add_property("dtype", &typeinfo_dtype)
    .def_readonly("cxxtype", &bob::core::array::typeinfo::dtype)
    .def_readonly("nd", &bob::core::array::typeinfo::nd)
    .add_property("shape", &ti_shape)
    .add_property("stride", &ti_stride)
    .def("__str__", &bob::core::array::typeinfo::str)
    ;

}
