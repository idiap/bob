/**
 * @file cxx/core/src/Exception.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the Exception class. Makes sure we have at least one
 * virtual method implemented in a cxx file so that the pythonic bindings work
 * as expected.
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

#include <boost/format.hpp>
#include "core/Exception.h"

Torch::core::Exception::Exception() throw() {
}

Torch::core::Exception::Exception(const Torch::core::Exception&) throw() {
}

Torch::core::Exception::~Exception() throw() {
}

const char* Torch::core::Exception::what() const throw() {
 static const char* what_string = "Generic core::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

Torch::core::DeprecationError::DeprecationError(const std::string& op) throw(): 
  m_op(op) {
}

Torch::core::DeprecationError::~DeprecationError() throw() {
}

const char* Torch::core::DeprecationError::what() const throw() {
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

