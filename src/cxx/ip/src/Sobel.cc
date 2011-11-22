/**
 * @file cxx/ip/src/Sobel.cc
 * @date Fri Apr 29 12:13:22 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with the Sobel operator
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

#include "ip/Sobel.h"

namespace ip = Torch::ip;

ip::Sobel::Sobel( const bool up_positive, const bool left_positive,
    const enum sp::Convolution::SizeOption size_opt,
    const enum sp::Convolution::BorderOption border_opt):
  m_up_positive(up_positive), m_left_positive(left_positive),
  m_size_opt(size_opt), m_border_opt(border_opt)
{
  computeKernels();
}

void ip::Sobel::computeKernels()
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
    m_kernel_x = 1, 0, -1, 2, 0, -1, 1, 0, -1;
  else
    m_kernel_x = -1, 0, 1, -2, 0, 1, -1, 0, 1;
}
