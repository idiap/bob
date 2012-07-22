/**
 * @file cxx/ip/src/MultiscaleRetinex.cc
 * @date Mon May 2 10:01:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the MultiscaleRetinex algorithm
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

#include "ip/MultiscaleRetinex.h"

void bob::ip::MultiscaleRetinex::computeKernels()
{
  for( size_t s=0; s<m_n_scales; ++s)
  {
    // size of the kernel 
    int s_size = m_size_min + s * m_size_step;
    // sigma of the kernel
    double s_sigma = m_sigma * s_size / m_size_min;
    // Initialize the Gaussian
    m_gaussians[s].reset(s_size, s_size, s_sigma, s_sigma, 
      m_conv_border);
  }
}

void bob::ip::MultiscaleRetinex::reset(const size_t n_scales, 
  const int size_min, const int size_step, const double sigma,
  const enum bob::sp::Extrapolation::BorderType border_type)
{
  m_n_scales = n_scales;
  m_gaussians.reset(new bob::ip::Gaussian[m_n_scales]);
  m_size_min = size_min;
  m_size_step = size_step;
  m_sigma = sigma;
  m_conv_border = border_type;
  computeKernels();
}

bob::ip::MultiscaleRetinex& 
bob::ip::MultiscaleRetinex::operator=(const bob::ip::MultiscaleRetinex& other)
{
  if (this != &other)
  {
    m_n_scales = other.m_n_scales;
    m_gaussians.reset(new bob::ip::Gaussian[m_n_scales]);
    m_size_min = other.m_size_min;
    m_size_step = other.m_size_step;
    m_sigma = other.m_sigma;
    m_conv_border = other.m_conv_border;
    computeKernels();
  }
  return *this;
}

bool 
bob::ip::MultiscaleRetinex::operator==(const bob::ip::MultiscaleRetinex& b) const
{
  return (this->m_n_scales == b.m_n_scales && this->m_size_min== b.m_size_min && 
          this->m_size_step == b.m_size_step && this->m_sigma == b.m_sigma && 
          this->m_conv_border == b.m_conv_border);
}

bool 
bob::ip::MultiscaleRetinex::operator!=(const bob::ip::MultiscaleRetinex& b) const
{
  return !(this->operator==(b));
}
