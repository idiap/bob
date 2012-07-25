/**
 * @file cxx/ip/src/FaceEyesNorm.cc
 * @date Thu Apr 14 21:03:45 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform geometric normalization.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "ip/FaceEyesNorm.h"

bob::ip::FaceEyesNorm::FaceEyesNorm( const double eyes_distance,
    const size_t crop_height, const size_t crop_width, const double crop_offset_h,
    const double crop_offset_w):
  m_eyes_distance(eyes_distance), m_crop_height(crop_height),
  m_crop_width(crop_width), m_crop_offset_h(crop_offset_h),
  m_crop_offset_w(crop_offset_w), m_out_shape(crop_height, crop_width), 
  m_geom_norm(new GeomNorm(0., 0., crop_height, crop_width, crop_offset_h, 
    crop_offset_w) )
{
}

bob::ip::FaceEyesNorm::FaceEyesNorm( const FaceEyesNorm& other):
  m_eyes_distance(other.m_eyes_distance), m_crop_height(other.m_crop_height),
  m_crop_width(other.m_crop_width), m_crop_offset_h(other.m_crop_offset_h),
  m_crop_offset_w(other.m_crop_offset_w), m_out_shape(other.m_crop_height, other.m_crop_width), 
  m_geom_norm(new GeomNorm(0., 0., m_crop_height, m_crop_width, m_crop_offset_h, 
    m_crop_offset_w) )
{
}

bob::ip::FaceEyesNorm& 
bob::ip::FaceEyesNorm::operator=(const bob::ip::FaceEyesNorm& other)
{
  if (this != &other)
  {
    m_eyes_distance = other.m_eyes_distance;
    m_crop_height = other.m_crop_height;
    m_crop_width = other.m_crop_width;
    m_crop_offset_h = other.m_crop_offset_h;
    m_crop_offset_w = other.m_crop_offset_w;
    m_out_shape(0) = m_crop_height;
    m_out_shape(1) = m_crop_width;
    m_geom_norm.reset(new GeomNorm(0., 0, m_crop_height, m_crop_width, 
      m_crop_offset_h, m_crop_offset_w) );
  }
  return *this;
}

bool 
bob::ip::FaceEyesNorm::operator==(const bob::ip::FaceEyesNorm& b) const
{
  return (this->m_eyes_distance == b.m_eyes_distance && this->m_crop_height == b.m_crop_height && 
          this->m_crop_width == b.m_crop_width && this->m_crop_offset_h == b.m_crop_offset_h && 
          this->m_crop_offset_w == b.m_crop_offset_w);
}

bool 
bob::ip::FaceEyesNorm::operator!=(const bob::ip::FaceEyesNorm& b) const
{
  return !(this->operator==(b));
}


