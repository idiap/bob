/**
 * @file cxx/core/src/convert_exception.cc
 * @date Thu Apr 28 16:09:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Exception for the convert functions
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
#include "core/convert_exception.h"

Torch::core::ConvertZeroInputRange::ConvertZeroInputRange() throw() {
}

Torch::core::ConvertZeroInputRange::~ConvertZeroInputRange() throw() {
}

const char* Torch::core::ConvertZeroInputRange::what() const throw() {
  try {
    boost::format message("Cannot convert an array with a zero width input range.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "core::ConvertZeroInputRange: cannot format, exception raised";
    return emergency;
  }
}

Torch::core::ConvertInputAboveMaxRange::ConvertInputAboveMaxRange(const double v, const double m) throw():
  m_val(v), m_max(m)
{
}

Torch::core::ConvertInputAboveMaxRange::~ConvertInputAboveMaxRange() throw() {
}

const char* Torch::core::ConvertInputAboveMaxRange::what() const throw() {
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

Torch::core::ConvertInputBelowMinRange::ConvertInputBelowMinRange(const double v, const double m) throw():
  m_val(v), m_min(m)
{
}

Torch::core::ConvertInputBelowMinRange::~ConvertInputBelowMinRange() throw() {
}

const char* Torch::core::ConvertInputBelowMinRange::what() const throw() {
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
