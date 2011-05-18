#include "trainer/Exception.h"
#include <boost/format.hpp>


Torch::trainer::Exception::Exception() throw() {
}

Torch::trainer::Exception::~Exception() throw() {
}

const char* Torch::trainer::Exception::Exception::what() const throw() {
 static const char* what_string = "Generic trainer::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}


Torch::trainer::NoPriorGMM::NoPriorGMM() throw() {
}

Torch::trainer::NoPriorGMM::~NoPriorGMM() throw() {
}

const char* Torch::trainer::NoPriorGMM::what() const throw() {
  return "MAP_GMMTrainer: Prior GMM has not been set";
}
