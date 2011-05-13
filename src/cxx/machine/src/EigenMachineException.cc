/**
  * @file src/cxx/machine/src/EigenMachineException.cc
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Fri 13 May 19:33:24 2011
  *
  * @brief Implements the exceptions for the EigenMachine 
  */

#include "machine/EigenMachineException.h"
#include <boost/format.hpp>

namespace machine = Torch::machine;

machine::EigenMachineNOutputsTooLarge::EigenMachineNOutputsTooLarge(const int n_outputs, const int n_outputs_max) throw(): 
  m_n_outputs(n_outputs), m_n_outputs_max(n_outputs_max) 
{
}

machine::EigenMachineNOutputsTooLarge::~EigenMachineNOutputsTooLarge() throw() {
}

const char* machine::EigenMachineNOutputsTooLarge::what() const throw() {
  try {
    boost::format message("Trying to set a too large number of outputs '%d', as only '%d' eigenvalues/eigenvectors have been set in the machine.");
    message % m_n_outputs;
    message % m_n_outputs_max;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "machine::EigenMachineNOutputsTooLarge: cannot format, exception raised";
    return emergency;
  }
}

