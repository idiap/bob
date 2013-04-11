/**
 * @file core/cxx/Exception.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the Exception class. Makes sure we have at least one
 * virtual method implemented in a cxx file so that the pythonic bindings work
 * as expected.
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
#include <bob/core/Exception.h>

bob::core::Exception::Exception(const std::string& reason) throw()
: m_reason(reason){
}

bob::core::Exception::Exception(const bob::core::Exception& other) throw()
: m_reason(other.m_reason)
{
}

bob::core::Exception::~Exception() throw() {
}

const char* bob::core::Exception::what() const throw() {
 return m_reason.c_str();
}

bob::core::DeprecationError::DeprecationError(const std::string& op) throw():
  m_op(op) {
}

bob::core::DeprecationError::~DeprecationError() throw() {
}

const char* bob::core::DeprecationError::what() const throw() {
  try {
    boost::format message("Deprecated operation: %s");
    message % m_op;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::DeprecationError: cannot format, exception raised";
    return emergency;
  }
}

bob::core::NotImplementedError::NotImplementedError(const std::string& reason) throw()
: m_reason(reason)
  {}

bob::core::NotImplementedError::~NotImplementedError() throw() {
}

const char* bob::core::NotImplementedError::what() const throw() {
  return m_reason.c_str();
}


bob::core::InvalidArgumentException::InvalidArgumentException(const std::string& reason) throw()
  : m_reason(reason)
  {}

bob::core::InvalidArgumentException::~InvalidArgumentException() throw() {
}

const char* bob::core::InvalidArgumentException::what() const throw() {
  return m_reason.c_str();
}

