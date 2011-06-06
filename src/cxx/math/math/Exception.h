/**
  * @file src/cxx/math/math/Exception.h
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  * @date Mon 6 June 11:02:04 2011 
  *
  * @brief Exceptions used throughout the math subsystem of Torch
  */

#ifndef TORCH5SPRO_MATH_EXCEPTION_H 
#define TORCH5SPRO_MATH_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"

namespace Torch { namespace math {

  class Exception: public Torch::core::Exception {
    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  /**
   * Raised when a LAPACK function returns a non-zero value.
   */
  class LapackError: public Exception {
    public:
      LapackError(const std::string& msg) throw();
      virtual ~LapackError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_submessage;
      mutable std::string m_message;
  };

}}

#endif /* TORCH5SPRO_MATH_EXCEPTION_H */
