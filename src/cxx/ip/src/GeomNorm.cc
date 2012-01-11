/**
 * @file cxx/ip/src/GeomNorm.cc
 * @date Mon Apr 11 22:17:04 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform geometric normalization.
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

#include "ip/GeomNorm.h"

namespace ip = bob::ip;

ip::GeomNorm::GeomNorm( const double rotation_angle, const double scaling_factor,
  const int crop_height, const int crop_width, const int crop_offset_h, 
  const int crop_offset_w): 
  m_rotate(new Rotate(rotation_angle)), m_scaling_factor(scaling_factor),
  m_crop_height(crop_height), m_crop_width(crop_width),
  m_crop_offset_h(crop_offset_h), m_crop_offset_w(crop_offset_w),
  m_out_shape(crop_height, crop_width)
{
}

ip::GeomNorm::~GeomNorm() { }

