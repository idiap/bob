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

  class NOutputsMismatch: public Exception {
    public:
      NOutputsMismatch(const int n_outputs1, const int n_outputs2) throw();
      virtual ~NOutputsMismatch() throw();
      virtual const char* what() const throw();

    private:
      int m_n_outputs1;
      int m_n_outputs2;
      mutable std::string m_message;
  };

  /**
   * Raised when the frames of a FrameSample have not the expected size
   */
  class IncompatibleFrameSample: public Exception {
    public:
      IncompatibleFrameSample (const int expected_framesize, const int received_framesize) throw();
      virtual ~IncompatibleFrameSample () throw();
      virtual const char* what() const throw();

    private:
      int expected_framesize;
      int received_framesize;
      mutable std::string m_message;
  };
}}

#endif /* TORCH5SPRO_MACHINE_EXCEPTION_H */
