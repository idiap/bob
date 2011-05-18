#ifndef TORCH5SPRO_TRAINER_EXCEPTION_H
#define TORCH5SPRO_TRAINER_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { namespace trainer {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  /**
   * Raised when some computations need a prior GMM and no one is set
   */
  class NoPriorGMM: public Exception {
    public:
      NoPriorGMM() throw();
      virtual ~NoPriorGMM() throw();
      virtual const char* what() const throw();
  };
}}

#endif /* TORCH5SPRO_TRAINER_EXCEPTION_H */
