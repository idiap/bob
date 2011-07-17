/**
 * @file src/cxx/core/core/repmat_exception.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Defines exceptions related to the repmat function
 */

#ifndef TORCH5SPRO_CORE_REPMAT_EXCEPTION_H 
#define TORCH5SPRO_CORE_REPMAT_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { 
  
  namespace core {

    /**
     * The NonMultipleLength exception occurs when a dimension of the 2D dst 
     * array of the repmat() function is not a multiple of the corresponding
     * dimension in the 2D src array.
     */
    class RepmatNonMultipleLength: public Exception {
      public:
        RepmatNonMultipleLength(const int src_dim, const int dst_dim) throw();
        virtual ~RepmatNonMultipleLength() throw();
        virtual const char* what() const throw();

      private:
        int m_src_dim;
        int m_dst_dim;
        mutable std::string m_message;
    };

  }

}

#endif /* TORCH_CORE_REPMAT_EXCEPTION_H */
