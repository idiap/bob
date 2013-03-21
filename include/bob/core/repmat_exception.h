/**
 * @file bob/core/repmat_exception.h
 * @date Sun Jul 17 14:10:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Defines exceptions related to the repmat function
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_CORE_REPMAT_EXCEPTION_H 
#define BOB_CORE_REPMAT_EXCEPTION_H

#include <bob/core/Exception.h>

namespace bob { 
  
  namespace core {

    /**
     * @ingroup CORE
     * @brief The NonMultipleLength exception occurs when a dimension of the
     * 2D dst array of the repmat() function is not a multiple of the 
     * corresponding dimension in the 2D src array.
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

#endif /* BOB_CORE_REPMAT_EXCEPTION_H */
