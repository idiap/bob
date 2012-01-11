/**
 * @file cxx/math/math/Exception.h
 * @date Mon Jun 6 11:49:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Exceptions used throughout the math subsystem of bob
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

#ifndef BOB5SPRO_MATH_EXCEPTION_H 
#define BOB5SPRO_MATH_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"

namespace bob { namespace math {

  class Exception: public bob::core::Exception {
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

  /**
   * Raised when the parameter p of the norminv function is not in the range 
   * ]0,1[.
   */
  class NorminvPNotInRangeError: public Exception {
    public:
      NorminvPNotInRangeError(const double p) throw();
      virtual ~NorminvPNotInRangeError() throw();
      virtual const char* what() const throw();

    private:
      double m_p;
      mutable std::string m_message;
  };

}}

#endif /* BOB5SPRO_MATH_EXCEPTION_H */
