/**
 * @file cxx/core/src/array_exception.cc
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Exceptions related to the blitz++ multidimensional arrays
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
#include "core/array_exception.h"

bob::core::NonZeroBaseError::NonZeroBaseError( const int dim, 
  const int base) throw(): 
    m_dim(dim), m_base(base) 
{
}

bob::core::NonZeroBaseError::~NonZeroBaseError() throw() {
}

const char* bob::core::NonZeroBaseError::what() const throw() {
  try {
    boost::format message(
      "The input array has dimension '%d' with a non-zero base index (base=%d).");
    message % m_dim;
    message % m_base;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonZeroBaseError: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::NonOneBaseError::NonOneBaseError( const int dim, 
  const int base) throw(): 
    m_dim(dim), m_base(base) 
{
}

bob::core::NonOneBaseError::~NonOneBaseError() throw() {
}

const char* bob::core::NonOneBaseError::what() const throw() {
  try {
    boost::format message(
      "The input array has dimension '%d' with a non-one base index (base=%d).");
    message % m_dim;
    message % m_base;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonOneBaseError: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::NonCContiguousError::NonCContiguousError() throw() {
}

bob::core::NonCContiguousError::~NonCContiguousError() throw() {
}

const char* bob::core::NonCContiguousError::what() const throw() {
  try {
    boost::format message(
      "The input array is not a C-style array (row-major order) stored in contiguous memory area.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonCContiguousError: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::NonFortranContiguousError::NonFortranContiguousError() throw() {
}

bob::core::NonFortranContiguousError::~NonFortranContiguousError() throw() {
}

const char* bob::core::NonFortranContiguousError::what() const throw() {
  try {
    boost::format message(
      "The input array is not a Fortran-style array (column-major order) stored in contiguous memory area.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::NonFortranContiguousError: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::UnexpectedShapeError::UnexpectedShapeError() throw() {
}

bob::core::UnexpectedShapeError::~UnexpectedShapeError() throw() {
}

const char* bob::core::UnexpectedShapeError::what() const throw() {
  try {
    boost::format message(
      "The array does not have the expected size.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::UnexpectedShapeError: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::DifferentBaseError::DifferentBaseError() throw() {
}

bob::core::DifferentBaseError::~DifferentBaseError() throw() {
}

const char* bob::core::DifferentBaseError::what() const throw() {
  try {
    boost::format message(
      "The array does not have the expected size.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::DifferentBaseError: cannot \
      format, exception raised";
    return emergency;
  }
}

