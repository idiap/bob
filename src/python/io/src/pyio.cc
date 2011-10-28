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
  bp::object retval = bp::object(bp::handle<>((PyObject*)tp::buffer_array(b)));
  return retval;
}

bp::object tp::buffer_object (const io::buffer& b) {
  bp::object retval = bp::object(bp::handle<>((PyObject*)tp::buffer_array(b)));
  return retval;
}

bp::object tp::npyarray_object (tp::npyarray& b) {
  bp::object retval = bp::object(bp::handle<>((PyObject*)b.shallow_copy_force()));
  return retval;
}
