/**
 * @file src/python/core/src/exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the core extension into Python. Please note that, for each
 * exception type you only need to declare once the translator. All other
 * modules will benefit from it automatically.
 */

#include "core/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

/**
 * This method is only useful to test exception throwing in Python code.
 */
static void throw_exception(void) {
  throw Torch::core::Exception();
}

void bind_core_exception() {
  BaseCxxToPythonTranslator<Torch::core::Exception>("Exception", "The core Exception class should be used as a basis for all Torch-Python exceptions.");
  boost::python::def("throw_exception", &throw_exception);
}
