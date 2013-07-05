/**
 * @file ip/cxx/TanTriggs.cc
 * @date Fri Mar 18 18:09:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform Tan and Triggs preprocessing.
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

#include "bob/ip/TanTriggs.h"

bob::ip::TanTriggs::TanTriggs( const double gamma, const double sigma0,
    const double sigma1, const size_t radius, const double threshold,
    const double alpha,
    const bob::sp::Extrapolation::BorderType border_type):
  m_gamma(gamma), m_sigma0(sigma0), m_sigma1(sigma1), m_radius(radius),
  m_threshold(threshold), m_alpha(alpha), m_border_type(border_type)
{
  //m_size = 2*floor( 3*m_sigma1)+1;
  computeDoG( m_sigma0, m_sigma1, 2*m_radius+1);
}

void bob::ip::TanTriggs::reset(const double gamma, const double sigma0,
  const double sigma1, const size_t radius, const double threshold,
  const double alpha,  const bob::sp::Extrapolation::BorderType border_type)
{
  m_gamma = gamma;
  m_sigma0 = sigma0;
  m_sigma1 = sigma1;
  m_radius = radius;
  m_threshold = threshold;
  m_alpha = alpha;
  m_border_type = border_type;
  computeDoG( m_sigma0, m_sigma1, 2*m_radius+1);
}

bob::ip::TanTriggs&
bob::ip::TanTriggs::operator=(const bob::ip::TanTriggs& other)
{
  if (this != &other)
  {
    m_gamma = other.m_gamma;
    m_sigma0 = other.m_sigma0;
    m_sigma1 = other.m_sigma1;
    m_radius = other.m_radius;
    m_threshold = other.m_threshold;
    m_alpha = other.m_alpha;
    m_border_type = other.m_border_type;
    computeDoG( m_sigma0, m_sigma1, 2*m_radius+1);
  }
  return *this;
}

bool bob::ip::TanTriggs::operator==(const bob::ip::TanTriggs& b) const
{
  return (this->m_gamma == b.m_gamma && this->m_sigma0 == b.m_sigma0 &&
          this->m_sigma1 == b.m_sigma1 && this->m_radius == b.m_radius &&
          this->m_threshold == b.m_threshold && this->m_alpha == b.m_alpha &&
          this->m_border_type == b.m_border_type);
}

bool bob::ip::TanTriggs::operator!=(const bob::ip::TanTriggs& b) const
{
  return !(this->operator==(b));
}

void
bob::ip::TanTriggs::performContrastEqualization(blitz::Array<double,2>& dst)
{
  const double inv_alpha = 1./m_alpha;
  const double wxh = dst.extent(0)*dst.extent(1);

  // first step: I:=I/mean(abs(I)^a)^(1/a)
  blitz::Range dst_y( dst.lbound(0), dst.ubound(0)),
               dst_x( dst.lbound(1), dst.ubound(1));
  double norm_fact =
    pow( sum( pow( fabs(dst(dst_y,dst_x)), m_alpha)) / wxh, inv_alpha);
  dst(dst_y,dst_x) /= norm_fact;

  // Second step: I:=I/mean(min(threshold,abs(I))^a)^(1/a)
  const double threshold_alpha = pow( m_threshold, m_alpha );
  norm_fact =  pow( sum( min( threshold_alpha,
    pow( fabs(dst(dst_y,dst_x)), m_alpha))) / wxh, inv_alpha);
  dst(dst_y,dst_x) /= norm_fact;

  // Last step: I:= threshold * tanh( I / threshold )
  dst(dst_y,dst_x) = m_threshold * tanh( dst(dst_y,dst_x) / m_threshold );
}


void bob::ip::TanTriggs::computeDoG(double sigma0, double sigma1, size_t size)
{
  // Generates two Gaussians with the given standard deviations
  // Warning: size should be odd
  blitz::Array<double,2> g0(size,size);
  blitz::Array<double,2> g1(size,size);
  const double inv_sigma0_2 = 0.5  / (sigma0*sigma0);
  const double inv_sigma1_2 = 0.5  / (sigma1*sigma1);
  int center = ((int)size) / 2;
  for(int y=0; y<(int)size; ++y)
    for(int x=0; x<(int)size; ++x)
    {
      int yy = y - center;
      int xx = x - center;
      int xx2 = xx*xx;
      int yy2 = yy*yy;

      g0(y,x) = exp( - inv_sigma0_2 * (xx2 + yy2) );
      g1(y,x) = exp( - inv_sigma1_2 * (xx2 + yy2) );
    }

  // Normalize the kernels such that the sum over the area is equal to 1
  // and compute the Difference of Gaussian filter
  const double inv_sum0 = 1. / blitz::sum(g0);
  const double inv_sum1 = 1. / blitz::sum(g1);
  m_kernel.resize( size, size);
  m_kernel = inv_sum0 * g0 - inv_sum1 * g1;
}

