/**
  * @file src/cxx/machine/machine/Exception.h
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Fri 13 May 19:26:04 2011 
  *
  * @brief Exceptions used throughout the machine subsystem of Torch
  */

#ifndef TORCH5SPRO_MACHINE_EIGENMACHINEEXCEPTION_H 
#define TORCH5SPRO_MACHINE_EIGENMACHINEEXCEPTION_H

#include <cstdlib>
#include "machine/Exception.h"

namespace Torch { namespace machine {

  class EigenMachineNOutputsTooLarge: public Exception {
    public:
      EigenMachineNOutputsTooLarge(const int n_outputs, const int n_outputs_max) throw();
      virtual ~EigenMachineNOutputsTooLarge() throw();
      virtual const char* what() const throw();

    private:
      int m_n_outputs;
      int m_n_outputs_max;
      mutable std::string m_message;
  };

}}

#endif /* TORCH5SPRO_MACHINE_EIGENMACHINEEXCEPTION_H */
