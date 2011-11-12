/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  8 Nov 17:55:32 2011
 *
 * @brief Binds ca::typeinfo
 */

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

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


void bind_core_array_typeinfo() {
  
  class_<ca::typeinfo>("typeinfo", "Type information for Torch C++ data", 
      no_init)
    .add_property("dtype", &typeinfo_dtype)
    .def_readonly("cxxtype", &ca::typeinfo::dtype)
    .def_readonly("nd", &ca::typeinfo::nd)
    .add_property("shape", &ti_shape)
    .add_property("stride", &ti_stride)
    .def("__str__", &ca::typeinfo::str)
    ;

}
