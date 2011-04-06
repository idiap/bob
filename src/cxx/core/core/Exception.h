/**
 * @file core/Exception.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief A torch-based exception
 */

#ifndef TORCH_CORE_EXCEPTION_H 
#define TORCH_CORE_EXCEPTION_H

#include <stdexcept>

namespace Torch { 
  
  namespace core {

    /**
     * The stock Torch exception class should be used as a base of any other
     * exception type in Torch. There are no input parameters you can specify
     * and that is on purpose. If you need to be specific about a problem,
     * derive from this one, virtually.
     *
     * Example: My filter only accepts gray-scaled images.
     *
     * class NotGrayScaleImage: virtual Torch::core::Exception {
     * ...
     * }
     *
     * Make sure to specify that your exception does not throw anything by
     * providing formal exception specification on their methods.
     */
    class Exception: virtual std::exception {

      public:

      /**
       * C'tor
       */
      Exception() throw();

      /**
       * Copy c'tor
       *
       * @param other Copying data from the other exception.
       */
      Exception(const Exception& other) throw();

      /**
       * Virtual d'tor as on the manual ;-)
       */
      virtual ~Exception() throw();

      /**
       * Returns a string representation of myself.
       */
      virtual const char* what() const throw();

    };


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
  }

}

#endif /* TORCH_CORE_EXCEPTION_H */
