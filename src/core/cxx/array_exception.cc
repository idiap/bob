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

bob::core::array::NonZeroBaseError::NonZeroBaseError( const int dim,
  const int base) throw():
    std::runtime_error("exception not set yet"),
    m_dim(dim), m_base(base)
{
}

bob::core::array::NonZeroBaseError::~NonZeroBaseError() throw() {
}

const char* bob::core::array::NonZeroBaseError::what() const throw() {
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


bob::core::array::NonOneBaseError::NonOneBaseError( const int dim,
  const int base) throw():
    std::runtime_error("exception not set yet"),
    m_dim(dim), m_base(base)
{
}

bob::core::array::NonOneBaseError::~NonOneBaseError() throw() {
}

const char* bob::core::array::NonOneBaseError::what() const throw() {
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


bob::core::array::NonCContiguousError::NonCContiguousError() throw():
  std::runtime_error("exception not set yet")
{
}

bob::core::array::NonCContiguousError::~NonCContiguousError() throw() {
}

const char* bob::core::array::NonCContiguousError::what() const throw() {
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


bob::core::array::NonFortranContiguousError::NonFortranContiguousError() throw(): std::runtime_error("exception not set yet")
{
}

bob::core::array::NonFortranContiguousError::~NonFortranContiguousError() throw() {
}

const char* bob::core::array::NonFortranContiguousError::what() const throw() {
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


bob::core::array::UnexpectedShapeError::UnexpectedShapeError() throw():
  std::runtime_error("exception not set yet")
{
}

bob::core::array::UnexpectedShapeError::~UnexpectedShapeError() throw() {
}

const char* bob::core::array::UnexpectedShapeError::what() const throw() {
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


bob::core::array::DifferentBaseError::DifferentBaseError() throw():
  std::runtime_error("exception not set yet")
{
}

bob::core::array::DifferentBaseError::~DifferentBaseError() throw() {
}

const char* bob::core::array::DifferentBaseError::what() const throw() {
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


bob::core::array::ConvertZeroInputRange::ConvertZeroInputRange() throw():
  std::runtime_error("exception not set yet")
{
}

bob::core::array::ConvertZeroInputRange::~ConvertZeroInputRange() throw() {
}

const char* bob::core::array::ConvertZeroInputRange::what() const throw() {
  try {
    boost::format message("Cannot convert an array with a zero width input range.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertZeroInputRange: cannot format, exception raised";
    return emergency;
  }
}

bob::core::array::ConvertInputAboveMaxRange::ConvertInputAboveMaxRange(const double v, const double m) throw():
  std::runtime_error("exception not set yet"),
  m_val(v), m_max(m)
{
}

bob::core::array::ConvertInputAboveMaxRange::~ConvertInputAboveMaxRange() throw() {
}

const char* bob::core::array::ConvertInputAboveMaxRange::what() const throw() {
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

bob::core::array::ConvertInputBelowMinRange::ConvertInputBelowMinRange(const double v, const double m) throw():
  std::runtime_error("exception not set yet"),
  m_val(v), m_min(m)
{
}

bob::core::array::ConvertInputBelowMinRange::~ConvertInputBelowMinRange() throw() {
}

const char* bob::core::array::ConvertInputBelowMinRange::what() const throw() {
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


bob::core::array::RepmatNonMultipleLength::RepmatNonMultipleLength(
  const int src_dim, const int dst_dim) throw():
    std::runtime_error("exception not set yet"),
    m_src_dim(src_dim), m_dst_dim(dst_dim)
{
}

bob::core::array::RepmatNonMultipleLength::~RepmatNonMultipleLength() throw() {
}

const char* bob::core::array::RepmatNonMultipleLength::what() const throw() {
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
