/**
 * @file core/cxx/reshape_exception.cc
 * @date Sun Jul 17 13:31:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Exceptions related to the reshape function.
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
#include "bob/core/reshape_exception.h"

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

