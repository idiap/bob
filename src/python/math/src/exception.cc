/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Mon 6 June 11:44:14 2011 
 *
 * @brief Math exception bindings
 */

#include <Python.h>
#include "math/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_math_exception() {
  CxxToPythonTranslator<Torch::math::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslatorPar<Torch::math::LapackError, Torch::math::Exception, const std::string&>("LapackError", "Raised when a LAPACK function is generated unexpected values.");

}
