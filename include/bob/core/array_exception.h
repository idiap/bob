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

#include <bob/core/Exception.h>

namespace bob { namespace core { namespace array {
    /**
     * @ingroup CORE_ARRAY
     * @{
     */

    /**
     * @brief The non-zero base error exception occurs when some function,
     * which requires blitz Arrays to have zero base indices (for efficiency 
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
     * @brief The non-one base error exception occurs when some function,
     * which requires blitz Arrays to have one base indices (for efficiency 
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
     * @brief The non-C contiguous error exception occurs when some function,
     * which requires blitz Arrays to be stored contiguously in memory, is 
     * used with an array which does not fulfill this requirement.
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
     * @brief The non-Fortran contiguous error exception occurs when some
     * function, which requires blitz Arrays to be stored contiguously in 
     * memory, is used with an array which does not fulfill this requirement.
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
     * @brief The UnexpectedShapeError exception occurs when a blitz array 
     * does not have the expected size.
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
     * @brief The DifferentBaseError exception occurs when two blitz arrays 
     * do not have the same base indices, whereas this is required.
     */
    class DifferentBaseError: public Exception {
      public:
        DifferentBaseError() throw();
        virtual ~DifferentBaseError() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

    /**
     * @brief A ConvertZeroInputRange is thrown when the specified input range
     * is empty
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
     * @brief A ConvertInputAboveMaxRange is thrown when an input value is 
     * above the maximum of the given input range.
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
     * @brief A ConvertInputBelowMinRange is thrown when an input value is 
     * below the minimum of the given input range.
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

    /**
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

    /**
     * @brief The DifferentNumberOfElements exception occurs when the 2D dst 
     * array of the reshape() functions does not contain the same number of 
     * elements as the 2D src array.
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

    /**
     * @}
     */
}}}

#endif /* BOB_CORE_ARRAY_EXCEPTION_H */
