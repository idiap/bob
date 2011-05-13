/**
  * @file src/cxx/machine/machine/Exception.h
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Fri 13 May 19:26:04 2011 
  *
  * @brief Exceptions used throughout the machine subsystem of Torch
  */

#ifndef TORCH5SPRO_MACHINE_EXCEPTION_H 
#define TORCH5SPRO_MACHINE_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"

namespace Torch { namespace machine {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class NInputsMismatch: public Exception {
    public:
      NInputsMismatch(const int n_inputs1, const int n_inputs2) throw();
      virtual ~NInputsMismatch() throw();
      virtual const char* what() const throw();

    private:
      int m_n_inputs1;
      int m_n_inputs2;
      mutable std::string m_message;
  };

}}

#endif /* TORCH5SPRO_MACHINE_EXCEPTION_H */
