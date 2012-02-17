/**
 * @file cxx/core/core/Exception.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief A bob-based exception
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

namespace bob { 
  
  namespace core {

    /**
     * The stock bob exception class should be used as a base of any other
     * exception type in bob. There are no input parameters you can specify
     * and that is on purpose. If you need to be specific about a problem,
     * derive from this one, virtually.
     *
     * Example: My filter only accepts gray-scaled images.
     *
     * class NotGrayScaleImage: virtual bob::core::Exception {
     * ...
     * }
     *
     * Make sure to specify that your exception does not throw anything by
     * providing formal exception specification on their methods.
     */
    class Exception: virtual std::exception {

      public:

      /**
       * C'tor
       */
      Exception() throw();

      /**
       * Copy c'tor
       *
       * @param other Copying data from the other exception.
       */
      Exception(const Exception& other) throw();

      /**
       * Virtual d'tor as on the manual ;-)
       */
      virtual ~Exception() throw();

      /**
       * Returns a string representation of myself.
       */
      virtual const char* what() const throw();

    };

    /**
     * A deprecation error is raised when the developer wants to avoid the use
     * of certain functionality in the code and for the user to migrate his
     * code.
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

  }

}

#endif /* BOB_CORE_EXCEPTION_H */
