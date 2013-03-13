/**
 * @file math/cxx/Exception.cc
 * @date Mon Jun 6 11:49:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Implements a generic exception for the math subsystem of bob
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

#include <bob/math/Exception.h>
#include <boost/format.hpp>

bob::math::Exception::Exception() throw() {
}

bob::math::Exception::~Exception() throw() {
}

const char* bob::math::Exception::what() const throw() {
 static const char* what_string = "Generic bob::math::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

bob::math::LapackError::LapackError(const std::string& msg) throw(): m_submessage(msg) {
}

bob::math::LapackError::~LapackError() throw() {
}

const char* bob::math::LapackError::what() const throw() {
  try {
    boost::format message("Error when executing a LAPACK function. '%s'");
    message % m_submessage;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::math::LapackError: cannot format, exception raised";
    return emergency;
  }
}

bob::math::GradientDimTooSmall::GradientDimTooSmall(const size_t ind, const size_t size) throw(): 
  m_ind(ind), m_size(size)
{
}

bob::math::GradientDimTooSmall::~GradientDimTooSmall() throw() {
}

const char* bob::math::GradientDimTooSmall::what() const throw() {
  try {
    boost::format message("The dimension '%d' is of length '%d', strictly smaller than 2. \
      No Gradient could be computed.");
    message % m_ind;
    message % m_size;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::math::GradientDimTooSmall: cannot format, exception raised";
    return emergency;
  }
}

bob::math::GradientNonPositiveSampleDistance::GradientNonPositiveSampleDistance(
  const size_t ind, const double val) throw(): 
  m_ind(ind), m_val(val)
{
}

bob::math::GradientNonPositiveSampleDistance::~GradientNonPositiveSampleDistance() throw() {
}

const char* bob::math::GradientNonPositiveSampleDistance::what() const throw() {
  try {
    boost::format message("The sample distance '%f' for dimension '%d' is NOT strictly positive.\
      No Gradient could be computed.");
    message % m_val;
    message % m_ind;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::math::GradientNonPositiveSampleDistance: cannot format, exception raised";
    return emergency;
  }
}
