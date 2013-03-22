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
#include <bob/io/Exception.h>

bob::io::Exception::Exception() throw() {
}

bob::io::Exception::~Exception() throw() {
}

const char* bob::io::Exception::what() const throw() {
 static const char* what_string = "Generic bob::io::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

bob::io::IndexError::IndexError(size_t id) throw(): m_id(id) {
}

bob::io::IndexError::~IndexError() throw() {
}

const char* bob::io::IndexError::what() const throw() {
  try {
    boost::format message("Cannot use index '%u'");
    message % m_id;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::IndexError: cannot format, exception raised";
    return emergency;
  }
}

bob::io::DimensionError::DimensionError(size_t got, size_t expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

bob::io::DimensionError::~DimensionError() throw() { }

const char* bob::io::DimensionError::what() const throw() {
  try {
    boost::format message("Expected '%u' dimensions, but got '%u'");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::DimensionError: cannot format, exception raised";
    return emergency;
  }
}

bob::io::TypeError::TypeError(bob::core::array::ElementType got, 
    bob::core::array::ElementType expected) throw() :
  m_got(got),
  m_expected(expected)
{
}

bob::io::TypeError::~TypeError() throw() { }

const char* bob::io::TypeError::what() const throw() {
  try {
    boost::format message("Expected element type '%s', but got '%s'");
    message % bob::core::array::stringize(m_expected) % bob::core::array::stringize(m_got);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::TypeError: cannot format, exception raised";
    return emergency;
  }
}

bob::io::UnsupportedTypeError::UnsupportedTypeError(bob::core::array::ElementType eltype) throw() :
  m_eltype(eltype)
{
}

bob::io::UnsupportedTypeError::~UnsupportedTypeError() throw() { }

const char* bob::io::UnsupportedTypeError::what() const throw() {
  try {
    boost::format message("The type '%s' is not supported for this operation");
    message % bob::core::array::stringize(m_eltype);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::UnsupportedTypeError: cannot format, exception raised";
    return emergency;
  }
}

bob::io::Uninitialized::Uninitialized() throw() { }
bob::io::Uninitialized::~Uninitialized() throw() { }
const char* bob::io::Uninitialized::what() const throw() {
  static const char* message = "The current Relationset you are trying to use is not properly initialized with rules and a dataset parent";
  return message;
}

bob::io::FileNotReadable::FileNotReadable(const std::string& filename) throw() :
  m_name(filename)
{
}

bob::io::FileNotReadable::~FileNotReadable() throw() { }

const char* bob::io::FileNotReadable::what() const throw() {
  try {
    boost::format message("Cannot read file '%s'");
    message % m_name;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::FileNotReadable: cannot format, exception raised";
    return emergency;
  }
}

bob::io::ImageUnsupportedDimension::ImageUnsupportedDimension(size_t n_dim) throw() :
  m_n_dim(n_dim)
{
}

bob::io::ImageUnsupportedDimension::~ImageUnsupportedDimension() throw() { }

const char* bob::io::ImageUnsupportedDimension::what() const throw() {
  try {
    boost::format message("Got an array with '%u' dimensions. ImageArrayCodec only supports 2 or 3 dimensions. For the 3D case, the size of the first dimension should be 3, which corresponds to the 3 RGB planes.");
    message % m_n_dim;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::ImageUnsupportedDimension: cannot format, exception raised";
    return emergency;
  }
}

bob::io::ImageUnsupportedType::ImageUnsupportedType() throw() { }
bob::io::ImageUnsupportedType::~ImageUnsupportedType() throw() { }

const char* bob::io::ImageUnsupportedType::what() const throw() {
  try {
    boost::format message("Got an array of unsupported type. Only uint8_t and uint16_t array types are supported.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::ImageUnsupportedType: cannot format, exception raised";
    return emergency;
  }
}

bob::io::ImageUnsupportedColorspace::ImageUnsupportedColorspace() throw()
{
}

bob::io::ImageUnsupportedColorspace::~ImageUnsupportedColorspace() throw() { }

const char* bob::io::ImageUnsupportedColorspace::what() const throw() {
  try {
    boost::format message("Does not support image with this colorspace.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::io::ImageUnsupportedColorspace: cannot format, exception raised";
    return emergency;
  }
}
