/**
 * @file cxx/ip/src/GeomNorm.cc
 * @date Mon Apr 11 22:17:04 2011 +0200
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

#include "ip/GeomNorm.h"

bob::ip::GeomNorm::GeomNorm( const double rotation_angle, const double scaling_factor,
    const size_t crop_height, const size_t crop_width, const double crop_offset_h,
    const double crop_offset_w):
  m_rotation_angle(rotation_angle), m_scaling_factor(scaling_factor),
  m_crop_height(crop_height), m_crop_width(crop_width),
  m_crop_offset_h(crop_offset_h), m_crop_offset_w(crop_offset_w)
{
}

bob::ip::GeomNorm::GeomNorm(const bob::ip::GeomNorm& other):
  m_rotation_angle(other.m_rotation_angle), 
  m_scaling_factor(other.m_scaling_factor),
  m_crop_height(other.m_crop_height), m_crop_width(other.m_crop_width),
  m_crop_offset_h(other.m_crop_offset_h), m_crop_offset_w(other.m_crop_offset_w)
{
}

bob::ip::GeomNorm::~GeomNorm()
{
}

bob::ip::GeomNorm& 
bob::ip::GeomNorm::operator=(const bob::ip::GeomNorm& other)
{
  if (this != &other)
  {
    m_rotation_angle = other.m_rotation_angle;
    m_scaling_factor = other.m_scaling_factor;
    m_crop_height = other.m_crop_height;
    m_crop_width = other.m_crop_width;
    m_crop_offset_h = other.m_crop_offset_h;
    m_crop_offset_w = other.m_crop_offset_w;
  }
  return *this;
}

bool 
bob::ip::GeomNorm::operator==(const bob::ip::GeomNorm& b) const
{
  return (this->m_rotation_angle == b.m_rotation_angle && this->m_scaling_factor == b.m_scaling_factor && 
          this->m_crop_height == b.m_crop_height && this->m_crop_width == b.m_crop_width && 
          this->m_crop_offset_h == b.m_crop_offset_w);
}

bool 
bob::ip::GeomNorm::operator!=(const bob::ip::GeomNorm& b) const
{
  return !(this->operator==(b));
}

blitz::TinyVector<double,2>
bob::ip::GeomNorm::operator()(const blitz::TinyVector<double,2>& position,
  const double rot_c_y, const double rot_c_x) const
{
  // compute scale and angle parameters
  const double sin_angle = -sin(m_rotation_angle * M_PI / 180.) * m_scaling_factor,
               cos_angle = cos(m_rotation_angle * M_PI / 180.) * m_scaling_factor;

  const double centered_y = position(0) - rot_c_y,
               centered_x = position(1) - rot_c_x;

  return blitz::TinyVector<double,2> (
    centered_x * sin_angle - centered_y * cos_angle + m_crop_offset_h,
    centered_x * cos_angle + centered_y * sin_angle + m_crop_offset_w
  );

}
