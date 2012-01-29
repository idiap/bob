/**
 * @file cxx/core/core/convert_exception.h
 * @date Thu Apr 28 16:09:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Exception for the convert functions
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

#ifndef BOB_CORE_CONVERT_EXCEPTION_H 
#define BOB_CORE_CONVERT_EXCEPTION_H

#include "core/Exception.h"

namespace bob { 
  
  namespace core {

    /**
      * A ConvertZeroInputRange is thrown when the specified input range is
      * empty
      */
    class ConvertZeroInputRange: public Exception {

      public:
        ConvertZeroInputRange() throw();
        virtual ~ConvertZeroInputRange() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

    /**
      * A ConvertInputAboveMaxRange is thrown when an input value is above 
      * the maximum of the given input range.
      */
    class ConvertInputAboveMaxRange: public Exception {

      public:
        ConvertInputAboveMaxRange(const double v, const double m) throw();
        virtual ~ConvertInputAboveMaxRange() throw();
        virtual const char* what() const throw();

      private:
        double m_val;
        double m_max;
        mutable std::string m_message;
    };

    /**
      * A ConvertInputBelowMinRange is thrown when an input value is below 
      * the minimum of the given input range.
      */
    class ConvertInputBelowMinRange: public Exception {

      public:
        ConvertInputBelowMinRange(const double v, const double m) throw();
        virtual ~ConvertInputBelowMinRange() throw();
        virtual const char* what() const throw();

      private:
        double m_val;
        double m_min;
        mutable std::string m_message;
    };

  }

}

#endif /* BOB_CORE_CONVERT_EXCEPTION_H */
