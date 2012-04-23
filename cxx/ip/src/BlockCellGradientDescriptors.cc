/**
 * @file cxx/ip/src/BlockCellGradientDescriptors.cc
 * @date Sun Apr 22 19:55:44 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include "ip/BlockCellGradientDescriptors.h"
#include "ip/Exception.h"
#include "core/array_assert.h"

namespace ip = bob::ip;

ip::GradientMaps::GradientMaps(const size_t height, 
  const size_t width, const GradientMagnitudeType mag_type):
    m_gy(height, width), m_gx(height, width), m_mag_type(mag_type)
{
}

void ip::GradientMaps::resize(const size_t height, const size_t width)
{
  m_gy.resize((int)height,(int)width);
  m_gx.resize((int)height,(int)width);
}

void ip::GradientMaps::setHeight(const size_t height)
{
  m_gy.resize((int)height,m_gy.extent(1));
  m_gx.resize((int)height,m_gx.extent(1));
}

void ip::GradientMaps::setWidth(const size_t width)
{
  m_gy.resize(m_gy.extent(0),(int)width);
  m_gx.resize(m_gx.extent(0),(int)width);
}


