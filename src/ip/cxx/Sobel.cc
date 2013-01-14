/**
 * @file ip/cxx/Sobel.cc
 * @date Fri Apr 29 12:13:22 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with the Sobel operator
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

#include "bob/ip/Sobel.h"

bob::ip::Sobel::Sobel( const bool up_positive, const bool left_positive,
    const bob::sp::Conv::SizeOption size_opt,
    const bob::sp::Extrapolation::BorderType border_type):
  m_up_positive(up_positive), m_left_positive(left_positive),
  m_size_opt(size_opt), m_border_type(border_type)
{
  computeKernels();
}

bob::ip::Sobel::Sobel(const Sobel& other):
  m_up_positive(other.m_up_positive), m_left_positive(other.m_left_positive),
  m_size_opt(other.m_size_opt), m_border_type(other.m_border_type)
{   
  computeKernels();        
} 

bob::ip::Sobel& 
bob::ip::Sobel::operator=(const bob::ip::Sobel& other)
{
  if (this != &other)
  {
    m_up_positive = other.m_up_positive;
    m_left_positive = other.m_left_positive;
    m_size_opt = other.m_size_opt;
    m_border_type = other.m_border_type;
    computeKernels();
  }
  return *this;
}

bool 
bob::ip::Sobel::operator==(const bob::ip::Sobel& b) const
{
  return (this->m_up_positive == b.m_up_positive && 
          this->m_left_positive == b.m_left_positive && 
          this->m_size_opt == b.m_size_opt && 
          this->m_border_type == b.m_border_type);
}

bool 
bob::ip::Sobel::operator!=(const bob::ip::Sobel& b) const
{
  return !(this->operator==(b));
}

void bob::ip::Sobel::computeKernels()
{
  // Resize the kernels if required
  if( m_kernel_y.extent(0) != 3 || m_kernel_y.extent(1) != 3)
    m_kernel_y.resize(3,3);
  if( m_kernel_x.extent(0) != 3 || m_kernel_x.extent(1) != 3)
    m_kernel_x.resize(3,3);

  if(m_up_positive)
    m_kernel_y = 1, 2, 1, 0, 0, 0, -1, -2, -1;
  else
    m_kernel_y = -1, -2, -1, 0, 0, 0, 1, 2, 1;

  if(m_left_positive)
    m_kernel_x = 1, 0, -1, 2, 0, -2, 1, 0, -1;
  else
    m_kernel_x = -1, 0, 1, -2, 0, 2, -1, 0, 1;
}
