/**
 * @file cxx/ip/src/crop.cc
 * @date Sat Apr 16 00:00:44 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include "ip/crop.h"
#include "ip/Exception.h"

namespace ipd = bob::ip::detail;

void ipd::cropParameterCheck( const int crop_y, const int crop_x,
  const int crop_h, const int crop_w, const int src_height, 
  const int src_width)
{
  // Check parameters and throw exception if required
  if( crop_y<0) {
    throw ParamOutOfBoundaryError("crop_y", false, crop_y, 0);
  }
  if( crop_x<0 ) {
    throw ParamOutOfBoundaryError("crop_x", false, crop_x, 0);
  }
  if( crop_h<0) {
    throw ParamOutOfBoundaryError("crop_h", false, crop_h, 0);
  }
  if( crop_w<0) {
    throw ParamOutOfBoundaryError("crop_w", false, crop_w, 0);
  }
  if( crop_y+crop_h>src_height ) {
    throw ParamOutOfBoundaryError("crop_y+crop_h", true, crop_y+crop_h, 
      src_height );
  }
  if( crop_x+crop_w>src_width ) {
    throw ParamOutOfBoundaryError("crop_x+crop_w", true, crop_x+crop_w, 
      src_width );
  }
}

