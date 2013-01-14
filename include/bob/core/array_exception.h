/**
 * @file bob/core/array_exception.h
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Defines exceptions related to multidimensional arrays
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

#ifndef BOB_CORE_ARRAY_EXCEPTION_H 
#define BOB_CORE_ARRAY_EXCEPTION_H

#include "bob/core/Exception.h"

namespace bob { 
  
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

    /**
     * The DifferentBaseError exception occurs when two blitz arrays do not
     * have the same base indices, whereas this is required.
     */
    class DifferentBaseError: public Exception {
      public:
        DifferentBaseError() throw();
        virtual ~DifferentBaseError() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

  }

}

#endif /* BOB_CORE_ARRAY_EXCEPTION_H */
