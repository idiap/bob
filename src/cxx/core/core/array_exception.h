/**
 * @file src/cxx/core/core/array_exception.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Defines exceptions related to multidimensional arrays
 */

#ifndef TORCH5SPRO_CORE_ARRAY_EXCEPTION_H 
#define TORCH5SPRO_CORE_ARRAY_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { 
  
  namespace core {

    /**
     * The non-zero base error exception occurs when some function, which 
     * requires blitz Arrays to have zero base indices (for efficiency 
     * purpose), is used with an array which does not fulfill this 
     * requirement.
     */
    class NonZeroBaseError: public Exception {
      public:
        NonZeroBaseError(const int dim, const int base) throw();
        virtual ~NonZeroBaseError() throw();
        virtual const char* what() const throw();

      private:
        int m_dim;
        int m_base;
        mutable std::string m_message;
    };


    /**
     * The non-one base error exception occurs when some function, which 
     * requires blitz Arrays to have one base indices (for efficiency 
     * purpose), is used with an array which does not fulfill this 
     * requirement.
     */
    class NonOneBaseError: public Exception {
      public:
        NonOneBaseError(const int dim, const int base) throw();
        virtual ~NonOneBaseError() throw();
        virtual const char* what() const throw();

      private:
        int m_dim;
        int m_base;
        mutable std::string m_message;
    };


    /**
     * The non-C contiguous error exception occurs when some function, which 
     * requires blitz Arrays to be stored contiguously in memory, is used 
     * with an array which does not fulfill this requirement.
     */
    class NonCContiguousError: public Exception {
      public:
        NonCContiguousError() throw();
        virtual ~NonCContiguousError() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };


    /**
     * The non-Fortran contiguous error exception occurs when some function, 
     * which requires blitz Arrays to be stored contiguously in memory, is 
     * used with an array which does not fulfill this requirement.
     */
    class NonFortranContiguousError: public Exception {
      public:
        NonFortranContiguousError() throw();
        virtual ~NonFortranContiguousError() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

    /**
     * The UnexpectedShapeError exception occurs when a blitz array does not
     * have the expected size.
     */
    class UnexpectedShapeError: public Exception {
      public:
        UnexpectedShapeError() throw();
        virtual ~UnexpectedShapeError() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

  }

}

#endif /* TORCH_CORE_EXCEPTION_H */
