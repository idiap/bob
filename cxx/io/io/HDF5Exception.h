/**
 * @file cxx/io/io/HDF5Exception.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Exceptions thrown by the HDF5 framework
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

#ifndef BOB_IO_HDF5EXCEPTION_H 
#define BOB_IO_HDF5EXCEPTION_H

#include <hdf5.h>
#include "io/Exception.h"

namespace bob { namespace io {

  /**
   * This is a generic exception for HDF5 errors. You should not throw it, but
   * since all others inherit from this one, you can catch by reference and
   * treat HDF5 problems in a generic way.
   */
  class HDF5Exception: public bob::io::Exception {
    public:
      HDF5Exception() throw();
      virtual ~HDF5Exception() throw();
      virtual const char* what() const throw();
  };

  /**
   * Thrown when the user tries to open an HDF5 file with a mode that is not
   * supported by the HDF5 library
   */
  class HDF5InvalidFileAccessModeError: public HDF5Exception {
    public:
      HDF5InvalidFileAccessModeError(const unsigned int mode) throw();
      virtual ~HDF5InvalidFileAccessModeError() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_mode;
      mutable std::string m_message;
  };

  /**
   * Thrown when we don't support the input type that we got from our API.
   */
  class HDF5UnsupportedCxxTypeError: public HDF5Exception {

    public:
      HDF5UnsupportedCxxTypeError() throw();
      virtual ~HDF5UnsupportedCxxTypeError() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

  /**
   * Thrown when we don't support a type that was read from the input file.
   */
  class HDF5UnsupportedTypeError: public HDF5Exception {
    
    public:
      HDF5UnsupportedTypeError() throw();
      HDF5UnsupportedTypeError(const boost::shared_ptr<hid_t>& datatype) throw();
      virtual ~HDF5UnsupportedTypeError() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

  /**
   * Thrown when the user tries to read/write using an unsupported number of
   * dimensions from arrays.
   */
  class HDF5UnsupportedDimensionError: public HDF5Exception {
    public:
      HDF5UnsupportedDimensionError(size_t n_dim) throw();
      virtual ~HDF5UnsupportedDimensionError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_n_dim;
      mutable std::string m_message;
  };

  /**
   * This exception is raised when we call the HDF5 C-API and that returns less
   * than zero as a status output.
   */
  class HDF5StatusError: public HDF5Exception {
    public:
      HDF5StatusError(const std::string& call, herr_t status) throw();
      virtual ~HDF5StatusError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_call;
      herr_t m_status;
      mutable std::string m_message;
  };

  /**
   * This exception is raised when the user asks for a certain array in an
   * array list that is out of bounds.
   */
  class HDF5IndexError: public HDF5Exception {
    public:
      HDF5IndexError(const std::string& location,
          size_t size, size_t asked_for) throw();
      virtual ~HDF5IndexError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_location; ///< the path to the dataset that has problems
      size_t m_size; ///< the current dataset size
      size_t m_asked_for; ///< which index the user asked for
      mutable std::string m_message;
  };

  /**
   * This exception is raised when the user tries to read or write to or from
   * an existing dataset using a type that is incompatible with the established
   * one for that dataset.
   */
  class HDF5IncompatibleIO: public HDF5Exception {
    public:
      HDF5IncompatibleIO(const std::string& location, 
          const std::string& supported,
          const std::string& user_input) throw();
      virtual ~HDF5IncompatibleIO() throw();
      virtual const char* what() const throw();

    private:
      std::string m_location; ///< the path to the location that has problems
      std::string m_supported; ///< string representation of supported type
      std::string m_user_input; ///< string representation of user input
      mutable std::string m_message;
  };

  /**
   * This exception is raised when the user tries to append to a certain
   * dataset that is not expandible. 
   */
  class HDF5NotExpandible: public HDF5Exception {
    public:
      HDF5NotExpandible(const std::string& location) throw();
      virtual ~HDF5NotExpandible() throw();
      virtual const char* what() const throw();

    private:
      std::string m_location; ///< the path to the location that has problems
      mutable std::string m_message;
  };

}}

#endif /* BOB_IO_HDF5EXCEPTION_H */
