/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Fri 25 Mar 2011 15:23:05 CET
 *
 * @brief Image processing exceptions 
 */

#include "ip/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_database_exception() {
  CxxToPythonTranslator<Torch::ip::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");
}
