/**
 * @file bob/core/Exception.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief A bob-based exception
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

#ifndef BOB_CORE_EXCEPTION_H
#define BOB_CORE_EXCEPTION_H

#include <stdexcept>
#include <sstream>

namespace bob {

  namespace core {

    /**
     * @brief The stock bob exception class should be used as a base of any
     * other exception type in bob. There are no input parameters you can
     * specify and that is on purpose. If you need to be specific about a 
     * problem, derive from this one.
     *
     * Example: My filter only accepts gray-scaled images.
     *
     * class NotGrayScaleImage: public bob::core::Exception {
     * ...
     * }
     *
     * Make sure to specify that your exception does not throw anything by
     * providing formal exception specification on their methods.
     */
    class Exception: public std::exception {

      public:

      /**
       * @brief C'tor
       */
      Exception() throw();

      /**
       * @brief Copy c'tor
       *
       * @param other Copying data from the other exception.
       */
      Exception(const Exception& other) throw();

      /**
       * @brief Virtual d'tor as on the manual ;-)
       */
      virtual ~Exception() throw();

      /**
       * @brief Returns a string representation of myself.
       */
      virtual const char* what() const throw();

    };

    /**
     * @brief A deprecation error is raised when the developer wants to avoid
     * the use of certain functionality in the code and for the user to
     * migrate the code.
     */
    class DeprecationError: public Exception {

      public:
        DeprecationError(const std::string& op) throw();
        virtual ~DeprecationError() throw();
        virtual const char* what() const throw();

      private:
        std::string m_op;
        mutable std::string m_message;
    };


    /**
     * @brief A NotImplementedError is raised when a specific function of a class
     * is only implementable in certain subclasses, but not in the current one.
     */
    class NotImplementedError: public Exception {

      public:
        NotImplementedError(const std::string& reason = "This function cannot be implemented in this class") throw();
        virtual ~NotImplementedError() throw();
        virtual const char* what() const throw();

      private:
        std::string m_reason;
    };

    /**
     * @brief An InvalidArgumentException is raised when a function receives a 
     * parameter that it cannot handle.
     */
    class InvalidArgumentException: public Exception {

      public:
        /** 
         * @brief Create exception with a self-chosen error message
         * @param  reason  The reason why the exception was thrown
         */
        InvalidArgumentException(const std::string& reason) throw();

        /**
         * @brief Create exception with a default error message
         * @param  parameter_name  The name of the parameter that was incorrect
         * @param  value  The value of the parameter that was incorrect
         */
        template <typename T>
          InvalidArgumentException(const std::string& parameter_name, const T& value) throw(){
            std::ostringstream s;
            s << "The given parameter '" << parameter_name << "' has an invalid value '" << value << "'.";
            m_reason = s.str();
          }

        /** 
         * @brief Create exception with a default error message
         * @param  parameter_name  The name of the parameter that was incorrect
         * @param  value  The value of the parameter that was incorrect
         * @param  min    The minimum value the parameter accepts
         * @param  max    The maximum value the parameter accepts
         */
        template <typename T>
          InvalidArgumentException(const std::string& parameter_name, const T& value, const T& min, const T& max) throw(){
            std::ostringstream s;
            s << "The value '" << value << "' of parameter '" << parameter_name << "' is out of range [ " << min << ", " << max << "].";
            m_reason = s.str();
          }
        virtual ~InvalidArgumentException() throw();
        virtual const char* what() const throw();

      private:
        std::string m_reason;
    };

  }

}

#endif /* BOB_CORE_EXCEPTION_H */
