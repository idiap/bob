/**
 * @file ip/cxx/shift.cc
 * @date Sun Apr 17 23:11:51 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include "bob/ip/shift.h"
#include "bob/ip/Exception.h"

void bob::ip::detail::shiftParameterCheck( const int shift_y, const int shift_x,
  const size_t src_height, const size_t src_width)
{
  // Check parameters and throw exception if required
  if( shift_y <= -(int)src_height ) {
    throw ParamOutOfBoundaryError("shift_y", false, shift_y, 
      -src_height+1);
  }
  if( shift_x <= -(int)src_width ) {
    throw ParamOutOfBoundaryError("shift_x", false, shift_x, 
      -src_width+1);
  }
  if( shift_y >= (int)src_height ) {
    throw ParamOutOfBoundaryError("shift_y", true, shift_y, 
      src_height-1);
  }
  if( shift_x >= (int)src_width ) {
    throw ParamOutOfBoundaryError("shift_x", true, shift_x, 
      src_width-1);
  }
}

