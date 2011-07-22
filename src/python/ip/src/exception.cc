/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Fri 25 Mar 2011 15:23:05 CET
 *
 * @brief Image processing exceptions 
 */

#include <Python.h>
#include "ip/Exception.h"
#include "core/python/exception.h"
#include <string>

using namespace Torch::core::python;

void bind_ip_exception() {
  CxxToPythonTranslator<Torch::ip::Exception, Torch::core::Exception>("Exception", "Raised when no other exception type is better to describe the problem. You should never use this!");
  CxxToPythonTranslatorPar4<Torch::ip::ParamOutOfBoundaryError, Torch::ip::Exception, const std::string&, const bool, const int, const int>("ParamOutOfBoundaryError", "Raised when a parameter from the IP module is out of boundary!");
  CxxToPythonTranslator<Torch::ip::UnknownScalingAlgorithm, Torch::ip::Exception>("UnknownScalingAlgorithm", "Raised when the given scaling algorithm is not valid!");
  CxxToPythonTranslator<Torch::ip::UnknownRotatingAlgorithm, Torch::ip::Exception>("UnknownRotatingAlgorithm", "Raised when the given rotating algorithm is not valid!");
  CxxToPythonTranslatorPar<Torch::ip::LBPUnsupportedNNeighbours, Torch::ip::Exception, const int>("LBPUnsupportedNNeighbours", "Raised when the construction of an LBP operator is called, with a number of neighbours != 4 & != 8 (currently not implemented).");
}
