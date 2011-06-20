/**
 * @author Francois Moulin <francois.moulin@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 16 Jun 14:12:23 2011 CEST
 *
 * Binds some trainer exceptions to Python
 */

#include "core/python/exception.h"
#include "trainer/Exception.h"

using namespace Torch::core::python;

void bind_trainer_exception() {
  CxxToPythonTranslator<Torch::trainer::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");

  CxxToPythonTranslator<Torch::trainer::NoPriorGMM, Torch::trainer::Exception>("NoPriorGMM", "Raised when some computations need a prior GMM and no one is set");

  CxxToPythonTranslatorPar<Torch::trainer::WrongNumberOfClasses, Torch::trainer::Exception, size_t>("WrongNumberOfClasses", "Raised when the number of classes is insufficient");

  CxxToPythonTranslatorPar3<Torch::trainer::WrongNumberOfFeatures, Torch::trainer::Exception, size_t, size_t, size_t>("WrongNumberOfFeatures", "Raised when the number of features is different from other classes in the dataset");

}
