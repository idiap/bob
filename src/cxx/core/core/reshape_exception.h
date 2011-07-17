/**
 * @file src/cxx/core/core/reshape_exception.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Defines exceptions related to the reshape function
 */

#ifndef TORCH5SPRO_CORE_RESHAPE_EXCEPTION_H 
#define TORCH5SPRO_CORE_RESHAPE_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { 
  
  namespace core {

    /**
     * The DifferentNumberOfElements exception occurs when the 2D dst array
     * of the reshape() functions does not contain the same number of elements
     * as the 2D src array.
     */
    class ReshapeDifferentNumberOfElements: public Exception {
      public:
        ReshapeDifferentNumberOfElements(const int expected, const int got) throw();
        virtual ~ReshapeDifferentNumberOfElements() throw();
        virtual const char* what() const throw();

      private:
        int m_expected;
        int m_got;
        mutable std::string m_message;
    };

  }

}

#endif /* TORCH_CORE_RESHAPE_EXCEPTION_H */
