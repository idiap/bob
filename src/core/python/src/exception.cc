/**
 * @file src/exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the core extension into Python. Please note that, for each
 * exception type you only need to declare once the translator. All other
 * modules will benefit from it automatically.
 */

#include <boost/python.hpp>

#include "core/Exception.h"

using namespace boost::python;

/**
 * The following lines of code implement exception translation from C++ into
 * python using Boost.Python and the instructions found on this webpage:
 *
 * http://stackoverflow.com/questions/2261858/boostpython-export-custom-exception
 */
static PyObject *pyTorchExceptionType = NULL;
static void translateException(const Torch::core::Exception& e) {
  assert(pyTorchExceptionType != NULL);
  boost::python::object pythonExceptionInstance(e);
  PyErr_SetObject(pyTorchExceptionType, pythonExceptionInstance.ptr());
}

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw Torch::core::Exception();
}

void bind_core_exception() {
  class_<Torch::core::Exception> torchCoreException("Exception", "The core Exception class should be used as a basis for all Torch-Python exceptions.", init<>("Creates a new exception"));
  torchCoreException.def("__str__", &Torch::core::Exception::what);
  pyTorchExceptionType = torchCoreException.ptr();
  register_exception_translator<Torch::core::Exception>(&translateException);
  def("throw_exception", &throw_exception);
}
