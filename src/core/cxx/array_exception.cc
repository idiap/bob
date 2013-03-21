/**
 * @file core/cxx/array_exception.cc
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Exceptions related to the blitz++ multidimensional arrays
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
#include <bob/core/array_exception.h>

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


bob::core::ConvertZeroInputRange::ConvertZeroInputRange() throw() {
}

bob::core::ConvertZeroInputRange::~ConvertZeroInputRange() throw() {
}

const char* bob::core::ConvertZeroInputRange::what() const throw() {
  try {
    boost::format message("Cannot convert an array with a zero width input range.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertZeroInputRange: cannot format, exception raised";
    return emergency;
  }
}

bob::core::ConvertInputAboveMaxRange::ConvertInputAboveMaxRange(const double v, const double m) throw():
  m_val(v), m_max(m)
{
}

bob::core::ConvertInputAboveMaxRange::~ConvertInputAboveMaxRange() throw() {
}

const char* bob::core::ConvertInputAboveMaxRange::what() const throw() {
  try {
    boost::format message("The value %f of the input array is above the maximum %f of the given input range.");
    message % m_val;
    message % m_max;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertInputAboveMaxRange: cannot format, exception raised";
    return emergency;
  }
}

bob::core::ConvertInputBelowMinRange::ConvertInputBelowMinRange(const double v, const double m) throw():
  m_val(v), m_min(m)
{
}

bob::core::ConvertInputBelowMinRange::~ConvertInputBelowMinRange() throw() {
}

const char* bob::core::ConvertInputBelowMinRange::what() const throw() {
  try {
    boost::format message("The value %f of the input array is below the minimum %f of the given input range.");
    message % m_val;
    message % m_min;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertInputBelowMinRange: cannot format, exception raised";
    return emergency;
  }
}


bob::core::RepmatNonMultipleLength::RepmatNonMultipleLength( 
  const int src_dim, const int dst_dim) throw(): 
    m_src_dim(src_dim), m_dst_dim(dst_dim) 
{
}

bob::core::RepmatNonMultipleLength::~RepmatNonMultipleLength() throw() {
}

const char* bob::core::RepmatNonMultipleLength::what() const throw() {
  try {
    boost::format message(
      "The 2D src array has a dimension of length '%d', whereas the 2D dst\
      array has a corresponding dimension of length '%d', which is not a\
      multiple of '%d'.");
    message % m_src_dim;
    message % m_dst_dim;
    message % m_src_dim;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::RepmatNonMultipleLength: cannot \
      format, exception raised";
    return emergency;
  }
}


bob::core::ReshapeDifferentNumberOfElements::ReshapeDifferentNumberOfElements( 
  const int expected, const int got) throw(): 
    m_expected(expected), m_got(got) 
{
}

bob::core::ReshapeDifferentNumberOfElements::~ReshapeDifferentNumberOfElements() throw() {
}

const char* bob::core::ReshapeDifferentNumberOfElements::what() const throw() {
  try {
    boost::format message(
      "The 2D dst array has '%d' elements whereas tje 2D src array as '%d' elements.");
    message % m_got;
    message % m_expected;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::DifferentNumberOfElements: cannot \
      format, exception raised";
    return emergency;
  }
}

