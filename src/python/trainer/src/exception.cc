#include "trainer/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;

void bind_trainer_exception() {
  CxxToPythonTranslator<Torch::trainer::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslator<Torch::trainer::NoPriorGMM, Torch::trainer::Exception>("NoPriorGMM", "Raised when some computations need a prior GMM and no one is set");

}
