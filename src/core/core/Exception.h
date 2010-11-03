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

  }

}

#endif /* TORCH_CORE_EXCEPTION_H */
