/**
 * @file cxx/sp/src/Exception.cc
 * @date Tue Mar 8 12:06:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the exceptions used throughout the SP subsystem of
 * bob.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "sp/Exception.h"

namespace sp = bob::sp;
namespace core = bob::core;

sp::Exception::Exception() throw() {
}

sp::Exception::~Exception() throw() {
}

const char* sp::Exception::what() const throw() {
  static const char* what_string = "Generic ip::Exception: user \
    specialization has not overwritten what() or is throwing an object of \
    this class (in which case, please fix it!)";
  return what_string;
}


sp::ExtrapolationDstTooSmall::ExtrapolationDstTooSmall() throw()
{
}

sp::ExtrapolationDstTooSmall::~ExtrapolationDstTooSmall() throw()
{
}

const char* sp::ExtrapolationDstTooSmall::what() const throw() {
  try {
    boost::format message(
      "The destination array is smaller than the source input array.");
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "sp::ExtrapolationDstTooSmall: cannot \
      format, exception raised";
    return emergency;
  }
}

