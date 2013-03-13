/**
 * @file ip/cxx/crop.cc
 * @date Sat Apr 16 00:00:44 2011 +0200
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

#include "bob/ip/crop.h"
#include "bob/core/Exception.h"

void bob::ip::detail::cropParameterCheck( const int crop_y, 
  const int crop_x, const size_t crop_h, const size_t crop_w, 
  const size_t src_height, const size_t src_width)
{
  // Check parameters and throw exception if required
  if (crop_y < 0)
    throw bob::core::InvalidArgumentException("crop_y", crop_y, 0, 
            (int)src_height);
  if (crop_x < 0)
    throw bob::core::InvalidArgumentException("crop_x", crop_x, 0, 
            (int)src_width);
  if (crop_y+crop_h > src_height)
    throw bob::core::InvalidArgumentException("crop_y+crop_h",
            crop_y+(int)crop_h, 0, (int)src_height);
  if (crop_x+crop_w > src_width)
    throw bob::core::InvalidArgumentException("crop_x+crop_w",
            crop_x+(int)crop_w, 0, (int)src_width);
}

