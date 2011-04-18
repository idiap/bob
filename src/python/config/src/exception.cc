/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 30 Mar 13:34:49 2011 
 *
 * @brief Binds some configuration exceptions
 */


#include "config/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_config_exception() {
  CxxToPythonTranslator<Torch::config::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");
  CxxToPythonTranslatorPar<Torch::config::KeyError, Torch::config::Exception, const std::string&>("KeyError", "Raised when we cannot find a variable with the given name");
  CxxToPythonTranslatorPar3<Torch::config::UnsupportedConversion, Torch::config::Exception, const std::string&, const std::type_info&, boost::python::object>("UnsupportedConversion", "Raised when we cannot cast an object to the user required type");
  CxxToPythonTranslator<Torch::config::PythonError, Torch::config::Exception>("PythonError", "Raised when we have problems with an underlying python interperter");
  CxxToPythonTranslatorPar2<Torch::config::NotImplemented, Torch::config::Exception, const std::string&, const std::string&>("NotImplemented", "Raised when the user asks for an operation an API exists for, but the underlying functionality is still not implemented.");
}
