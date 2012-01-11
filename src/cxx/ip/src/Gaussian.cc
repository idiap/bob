/**
 * @file cxx/ip/src/Gaussian.cc
 * @date Sat Apr 30 17:52:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with a Gaussian kernel
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

#include "ip/Gaussian.h"

namespace ip = bob::ip;

void ip::Gaussian::computeKernel()
{
  m_kernel_y.resize(2 * m_radius_y + 1);
  // Computes the kernel
  const double inv_sigma_y = 1.0 / m_sigma_y;
  for (int j = -m_radius_y; j <= m_radius_y; j ++)
      m_kernel_y(j + m_radius_y) = exp(- inv_sigma_y * (j * j));
  // Normalizes the kernel
  m_kernel_y /= blitz::sum(m_kernel_y);

  m_kernel_x.resize(2 * m_radius_x + 1);
  // Computes the kernel
  const double inv_sigma_x = 1.0 / m_sigma_x;
  for (int i = -m_radius_x; i <= m_radius_x; i++) {
    m_kernel_x(i + m_radius_x) = exp(- inv_sigma_x * (i * i));
  }
  // Normalizes the kernel
  m_kernel_x /= blitz::sum(m_kernel_x);
}

void ip::Gaussian::reset(const int radius_y, const int radius_x,
  const double sigma_y, const double sigma_x, 
  const enum bob::sp::Convolution::SizeOption size_opt,
  const enum bob::sp::Convolution::BorderOption border_opt)
{
  m_radius_y = radius_y;
  m_radius_x = radius_x;
  m_sigma_y = sigma_y;
  m_sigma_x = sigma_x;
  m_conv_size = size_opt;
  m_conv_border = border_opt;
  computeKernel();
}


template <>
void ip::Gaussian::operator()<double>(const blitz::Array<double,2>& src,
   blitz::Array<double,2>& dst)
{
  m_tmp_int.resize(bob::sp::getConvolveSepOutputSize(src, m_kernel_y, 0, m_conv_size));
  // Checks are postponed to the convolution function.
  bob::sp::convolveSep(src, m_kernel_y, m_tmp_int, 0,
    m_conv_size, m_conv_border);
  bob::sp::convolveSep(m_tmp_int, m_kernel_x, dst, 1,
    m_conv_size, m_conv_border);
}
