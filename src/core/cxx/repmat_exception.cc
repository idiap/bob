/**
 * @file core/cxx/repmat_exception.cc
 * @date Sun Jul 17 14:10:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Exceptions related to the repmat function.
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
#include <bob/core/repmat_exception.h>

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

