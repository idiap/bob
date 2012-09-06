/**
 * @file ip/cxx/SelfQuotientImage.cc
 * @date Thu Jul 19 11:52:08 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements the Self Quotient Image algorithm
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

#include "bob/ip/SelfQuotientImage.h"

void bob::ip::SelfQuotientImage::computeKernels()
{
  for( size_t s=0; s<m_n_scales; ++s)
  {
    // size of the kernel 
    size_t s_size = m_size_min + s * m_size_step;
    // sigma of the kernel
    double s_sigma2 = m_sigma2 * s_size / m_size_min;
    // Initialize the Gaussian
    m_wgaussians[s].reset(s_size, s_size, s_sigma2, s_sigma2, 
      m_conv_border);
  }
}

void bob::ip::SelfQuotientImage::reset(const size_t n_scales, 
  const size_t size_min, const size_t size_step, const double sigma2,
  const bob::sp::Extrapolation::BorderType border_type)
{
  m_n_scales = n_scales;
  m_wgaussians.reset(new bob::ip::WeightedGaussian[m_n_scales]);
  m_size_min = size_min;
  m_size_step = size_step;
  m_sigma2 = sigma2;
  m_conv_border = border_type;
  computeKernels();
}

bob::ip::SelfQuotientImage& 
bob::ip::SelfQuotientImage::operator=(const bob::ip::SelfQuotientImage& other)
{
  if (this != &other)
  {
    m_n_scales = other.m_n_scales;
    m_wgaussians.reset(new bob::ip::WeightedGaussian[m_n_scales]);
    m_size_min = other.m_size_min;
    m_size_step = other.m_size_step;
    m_sigma2 = other.m_sigma2;
    m_conv_border = other.m_conv_border;
    computeKernels();
  }
  return *this;
}

bool 
bob::ip::SelfQuotientImage::operator==(const bob::ip::SelfQuotientImage& b) const
{
  return (this->m_n_scales == b.m_n_scales && this->m_size_min== b.m_size_min && 
          this->m_size_step == b.m_size_step && this->m_sigma2 == b.m_sigma2 && 
          this->m_conv_border == b.m_conv_border);
}

bool 
bob::ip::SelfQuotientImage::operator!=(const bob::ip::SelfQuotientImage& b) const
{
  return !(this->operator==(b));
}
