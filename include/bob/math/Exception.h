/**
 * @file bob/math/Exception.h
 * @date Mon Jun 6 11:49:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Exceptions used throughout the math subsystem of bob
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

#ifndef BOB_MATH_EXCEPTION_H 
#define BOB_MATH_EXCEPTION_H

#include <cstdlib>
#include "bob/core/Exception.h"

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

  /**
   * Raised when a gradient is computed along a dimension whose length is 
   * strictly smaller than 2.
   */
  class GradientDimTooSmall: public Exception {
    public:
      GradientDimTooSmall(const size_t index, const size_t size) throw();
      virtual ~GradientDimTooSmall() throw();
      virtual const char* what() const throw();

    private:
      size_t m_ind;
      size_t m_size;
      mutable std::string m_message;
  };

  /**
   * Raised when a sample distance parameter of the gradient computation is
   * not strictly positive.
   */
  class GradientNonPositiveSampleDistance: public Exception {
    public:
      GradientNonPositiveSampleDistance(const size_t index, const double val) throw();
      virtual ~GradientNonPositiveSampleDistance() throw();
      virtual const char* what() const throw();

    private:
      size_t m_ind;
      double m_val;
      mutable std::string m_message;
  };

}}

#endif /* BOB_MATH_EXCEPTION_H */
