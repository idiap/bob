/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Fri 13 May 20:10:09 2011 
 *
 * @brief Machine exception bindings
 */

#include "machine/Exception.h"
#include "machine/EigenMachineException.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_machine_exception() {
  CxxToPythonTranslator<Torch::machine::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslatorPar2<Torch::machine::NInputsMismatch, Torch::machine::Exception, const int, const int>("NInputsMismatch", "Raised when a machine tries to set parameters which are causing mismatch in the number of inputs.");

  CxxToPythonTranslatorPar2<Torch::machine::NOutputsMismatch, Torch::machine::Exception, const int, const int>("NOutputsMismatch", "Raised when a machine tries to set parameters which are causing mismatch in the number of outputs.");

  CxxToPythonTranslatorPar2<Torch::machine::EigenMachineNOutputsTooLarge, Torch::machine::Exception, const int, const int>("EigenMachineNOutputsTooLarge", "Raised when an EigenMachine tries to set a too large number of outputs, wrt. to the number of eigenvectors/eigenvalues.");
  
}
