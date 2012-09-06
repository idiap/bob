/**
 * @file bob/io/Exception.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Exceptions used throughout the IO subsystem of bob
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

#ifndef BOB_IO_EXCEPTION_H 
#define BOB_IO_EXCEPTION_H

#include <cstdlib>
#include "bob/core/Exception.h"
#include "bob/core/array_type.h"

namespace bob { namespace io {

  class Exception: public bob::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class NameError: public Exception {
    public:
      NameError(const std::string& key) throw();
      virtual ~NameError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class IndexError: public Exception {
    public:
      IndexError(size_t id) throw();
      virtual ~IndexError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_id;
      mutable std::string m_message;
  };

  class DimensionError: public Exception {
    public:
      DimensionError(size_t got, size_t expected) throw();
      virtual ~DimensionError() throw();
      virtual const char* what() const throw();

    private:
      size_t m_got;
      size_t m_expected;
      mutable std::string m_message;
  };

  class TypeError: public Exception {
    public:
      TypeError(bob::core::array::ElementType got, 
          bob::core::array::ElementType expected) throw();
      virtual ~TypeError() throw();
      virtual const char* what() const throw();

    private:
      bob::core::array::ElementType m_got;
      bob::core::array::ElementType m_expected;
      mutable std::string m_message;
  };

  class UnsupportedTypeError: public Exception {
    public:
      UnsupportedTypeError(bob::core::array::ElementType eltype) throw();
      virtual ~UnsupportedTypeError() throw();
      virtual const char* what() const throw();

    private:
      bob::core::array::ElementType m_eltype;
      mutable std::string m_message;
  };

  class NonExistingElement: public Exception {
    public:
      NonExistingElement();
      virtual ~NonExistingElement() throw();
      virtual const char* what() const throw();
  };

  class Uninitialized: public Exception {
    public:
      Uninitialized() throw();
      virtual ~Uninitialized() throw();
      virtual const char* what() const throw();
  };

  class FileNotReadable: public Exception {
    public:
      FileNotReadable(const std::string& filename) throw();
      virtual ~FileNotReadable() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class ExtensionNotRegistered: public Exception {
    public:
      ExtensionNotRegistered(const std::string& extension) throw();
      virtual ~ExtensionNotRegistered() throw();
      virtual const char* what() const throw();

    private:
      std::string m_extension;
      mutable std::string m_message;
  };

  class CodecNotFound: public Exception {
    public:
      CodecNotFound(const std::string& codecname) throw();
      virtual ~CodecNotFound() throw();
      virtual const char* what() const throw();

    private:
      std::string m_name;
      mutable std::string m_message;
  };

  class PathIsNotAbsolute: public Exception {
    public:
      PathIsNotAbsolute(const std::string& path) throw();
      virtual ~PathIsNotAbsolute() throw();
      virtual const char* what() const throw();

    private:
      std::string m_path;
      mutable std::string m_message;
  };

  class ImageUnsupportedDimension: public Exception {
    public:
      ImageUnsupportedDimension(const size_t n_dim) throw();
      virtual ~ImageUnsupportedDimension() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_n_dim;
      mutable std::string m_message;
  };
  
  class ImageUnsupportedType: public Exception {
    public:
      ImageUnsupportedType(const bob::core::array::ElementType el_type)
        throw();
      virtual ~ImageUnsupportedType() throw();
      virtual const char* what() const throw();

    private:
      bob::core::array::ElementType m_el_type;
      mutable std::string m_message;
  };

  class ImageUnsupportedDepth: public Exception {
    public:
      ImageUnsupportedDepth(const unsigned int depth) throw();
      virtual ~ImageUnsupportedDepth() throw();
      virtual const char* what() const throw();

    private:
      unsigned int m_depth;
      mutable std::string m_message;
  };

}}

#endif /* BOB_IO_EXCEPTION_H */
