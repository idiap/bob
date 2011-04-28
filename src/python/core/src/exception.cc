/**
 * @file src/python/core/src/exception.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the core extension into Python. Please note that, for each
 * exception type you only need to declare once the translator. All other
 * modules will benefit from it automatically.
 */

#include "core/Exception.h"
#include "core/convert_exception.h"
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
  CxxToPythonTranslatorPar<Torch::core::DeprecationError, Torch::core::Exception, const std::string&>("DeprecationError", "A deprecation error is raised when the developer wants to avoid the use of certain functionality in the code and for the user to migrate his code.");
  CxxToPythonTranslator<Torch::core::ConvertZeroInputRange, Torch::core::Exception>("ConvertZeroInputRange", "A ConvertZeroInputRange error is raised when the user try to convert an array which has a zero width input range.");
  CxxToPythonTranslatorPar2<Torch::core::ConvertInputAboveMaxRange, Torch::core::Exception, const double, const double>("ConvertInputAboveMaxRange", "A ConvertInputAboveMaxRange exception is raised when an input value is larger than the maximum of the input range given to the convert function.");
  CxxToPythonTranslatorPar2<Torch::core::ConvertInputBelowMinRange, Torch::core::Exception, const double, const double>("ConvertInputBelowMinRange", "A ConvertInputBelowMinRange exception is raised when an input value is smaller than the minimum of the input range given to the convert function.");
  boost::python::def("throw_exception", &throw_exception);
}
