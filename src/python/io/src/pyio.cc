/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 12:29:54 2011
 *
 * @brief Implementation on python::io toolbox
 */

#include "io/python/pyio.h"

namespace bp = boost::python;
namespace io = Torch::io;
namespace tp = Torch::python;

bp::object tp::buffer_object (boost::shared_ptr<io::buffer> b) {
  bp::handle<> tmp((PyObject*)tp::buffer_array(b));
  bp::object retval(tmp);
  return retval;
}

bp::object tp::buffer_object (const io::buffer& b) {
  bp::handle<> tmp((PyObject*)tp::buffer_array(b));
  bp::object retval(tmp);
  return retval;
}

bp::object tp::npyarray_object (tp::npyarray& b) {
  bp::handle<> tmp((PyObject*)b.shallow_copy_force());
  bp::object retval(tmp);
  return retval;
}

bp::object tp::array_from_any (bp::object& o) {
  PyObject* arr = PyArray_FromAny(o.ptr(), 0, 0, 0, 0, 0);
  bp::handle<> tmp(arr);
  bp::object retval(tmp);
  return retval;
}
