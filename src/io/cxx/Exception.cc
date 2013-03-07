/**
 * @file io/cxx/Exception.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief
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

#include <boost/format.hpp>
#include "bob/io/Exception.h"

namespace io = bob::io;
namespace core = bob::core;
namespace array = bob::core::array;

io::Exception::Exception() throw() {
}

io::Exception::~Exception() throw() {
}

const char* io::Exception::what() const throw() {
 static const char* what_string = "Generic io::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

io::IndexError::IndexError(size_t id) throw(): m_id(id) {
}

io::IndexError::~IndexError() throw() {
}

const char* io::IndexError::what() const throw() {
  try {
    boost::format message("Cannot use index '%u'");
    message % m_id;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::IndexError: cannot format, exception raised";
    return emergency;
  }
}

io::DimensionError::DimensionError(size_t got, size_t expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

io::DimensionError::~DimensionError() throw() { }

const char* io::DimensionError::what() const throw() {
  try {
    boost::format message("Expected '%u' dimensions, but got '%u'");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::DimensionError: cannot format, exception raised";
    return emergency;
  }
}

io::TypeError::TypeError(bob::core::array::ElementType got, 
    bob::core::array::ElementType expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

io::TypeError::~TypeError() throw() { }

const char* io::TypeError::what() const throw() {
  try {
    boost::format message("Expected element type '%s', but got '%s'");
    message % array::stringize(m_expected) % array::stringize(m_got);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::TypeError: cannot format, exception raised";
    return emergency;
  }
}

io::UnsupportedTypeError::UnsupportedTypeError(bob::core::array::ElementType eltype) throw() :
  m_eltype(eltype)
{
}

io::UnsupportedTypeError::~UnsupportedTypeError() throw() { }

const char* io::UnsupportedTypeError::what() const throw() {
  try {
    boost::format message("The type '%s' is not supported for this operation");
    message % array::stringize(m_eltype);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::UnsupportedTypeError: cannot format, exception raised";
    return emergency;
  }
}

io::Uninitialized::Uninitialized() throw() { }
io::Uninitialized::~Uninitialized() throw() { }
const char* io::Uninitialized::what() const throw() {
  static const char* message = "The current Relationset you are trying to use is not properly initialized with rules and a dataset parent";
  return message;
}

io::FileNotReadable::FileNotReadable(const std::string& filename) throw() :
  m_name(filename)
{
}

io::FileNotReadable::~FileNotReadable() throw() { }

const char* io::FileNotReadable::what() const throw() {
  try {
    boost::format message("Cannot read file '%s'");
    message % m_name;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::FileNotReadable: cannot format, exception raised";
    return emergency;
  }
}

io::ImageUnsupportedDimension::ImageUnsupportedDimension(size_t n_dim) throw() :
  m_n_dim(n_dim)
{
}

io::ImageUnsupportedDimension::~ImageUnsupportedDimension() throw() { }

const char* io::ImageUnsupportedDimension::what() const throw() {
  try {
    boost::format message("Got an array with '%u' dimensions. ImageArrayCodec only supports 2 or 3 dimensions. For the 3D case, the size of the first dimension should be 3, which corresponds to the 3 RGB planes.");
    message % m_n_dim;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::ImageUnsupportedDimension: cannot format, exception raised";
    return emergency;
  }
}

io::ImageUnsupportedType::ImageUnsupportedType() throw() { }
io::ImageUnsupportedType::~ImageUnsupportedType() throw() { }

const char* io::ImageUnsupportedType::what() const throw() {
  try {
    boost::format message("Got an array of unsupported type. Only uint8_t and uint16_t array types are supported.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::ImageUnsupportedType: cannot format, exception raised";
    return emergency;
  }
}

io::ImageUnsupportedColorspace::ImageUnsupportedColorspace() throw()
{
}

io::ImageUnsupportedColorspace::~ImageUnsupportedColorspace() throw() { }

const char* io::ImageUnsupportedColorspace::what() const throw() {
  try {
    boost::format message("Does not support image with this colorspace.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "io::ImageUnsupportedColorspace: cannot format, exception raised";
    return emergency;
  }
}
