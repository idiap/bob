/**
 * @file cxx/core/core/reshape_exception.h
 * @date Sun Jul 17 13:31:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Defines exceptions related to the reshape function
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
