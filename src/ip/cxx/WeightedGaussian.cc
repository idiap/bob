/**
 * @file ip/cxx/WeightedGaussian.cc
 * @date Thu July 19 12:27:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with a weighted 
 *        Gaussian kernel (Used by the Self Quotient Image)
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

#include "bob/ip/WeightedGaussian.h"

void bob::ip::WeightedGaussian::computeKernel()
{
  m_kernel.resize(2 * m_radius_y + 1, 2 * m_radius_x + 1);
  m_kernel_weighted.resize(2 * m_radius_y + 1, 2 * m_radius_x + 1);
  // Computes the kernel
  const double inv_sigma2_y = 1.0 / m_sigma2_y;
  const double inv_sigma2_x = 1.0 / m_sigma2_x;
  for (int i = -(int)m_radius_y; i <= (int)m_radius_y; ++i) 
    for (int j = -(int)m_radius_x; j <= (int)m_radius_x; ++j)
      m_kernel(i + (int)m_radius_y, j + (int)m_radius_x) = 
        exp( -0.5 * (inv_sigma2_y * (i * i) + inv_sigma2_x * (j * j)));
  // Normalizes the kernel
  m_kernel /= blitz::sum(m_kernel);
}

void bob::ip::WeightedGaussian::reset(const size_t radius_y, const size_t radius_x,
  const double sigma2_y, const double sigma2_x, 
  const bob::sp::Extrapolation::BorderType border_type)
{
  m_radius_y = radius_y;
  m_radius_x = radius_x;
  m_sigma2_y = sigma2_y;
  m_sigma2_x = sigma2_x;
  m_conv_border = border_type;
  computeKernel();
}

bob::ip::WeightedGaussian& 
bob::ip::WeightedGaussian::operator=(const bob::ip::WeightedGaussian& other)
{
  if (this != &other)
  {
    m_radius_y = other.m_radius_y;
    m_radius_x = other.m_radius_x;
    m_sigma2_y = other.m_sigma2_y;
    m_sigma2_x = other.m_sigma2_x;
    m_conv_border = other.m_conv_border;
    computeKernel();
  }
  return *this;
}

bool 
bob::ip::WeightedGaussian::operator==(const bob::ip::WeightedGaussian& b) const
{
  return (this->m_radius_y == b.m_radius_y && this->m_radius_x == b.m_radius_x && 
          this->m_sigma2_y == b.m_sigma2_y && this->m_sigma2_x == b.m_sigma2_x && 
          this->m_conv_border == b.m_conv_border);
}

bool 
bob::ip::WeightedGaussian::operator!=(const bob::ip::WeightedGaussian& b) const
{
  return !(this->operator==(b));
}

template <>
void bob::ip::WeightedGaussian::operator()<double>(
  const blitz::Array<double,2>& src, blitz::Array<double,2>& dst)
{
  // Checks input
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertSameShape(src, dst);
  if(src.extent(0)<m_kernel.extent(0))
    throw bob::sp::ConvolutionKernelTooLarge(0, src.extent(0), m_kernel.extent(0));
  if(src.extent(1)<m_kernel.extent(1))
    throw bob::sp::ConvolutionKernelTooLarge(1, src.extent(0), m_kernel.extent(0));

  // 1/ Extrapolation of src
  // Resize temporary extrapolated src array
  blitz::TinyVector<int,2> shape = src.shape();
  shape(0) += 2 * (int)m_radius_y;
  shape(1) += 2 * (int)m_radius_x;
  m_src_extra.resize(shape);
  
  // Extrapolate
  if(m_conv_border == bob::sp::Extrapolation::Zero)
    bob::sp::extrapolateZero(src, m_src_extra);
  else if(m_conv_border == bob::sp::Extrapolation::NearestNeighbour)
    bob::sp::extrapolateNearest(src, m_src_extra);
  else if(m_conv_border == bob::sp::Extrapolation::Circular)
    bob::sp::extrapolateCircular(src, m_src_extra);
  else
    bob::sp::extrapolateMirror(src, m_src_extra);

  // 2/ Integral image then mean values
  shape += 1;
  m_src_integral.resize(shape);
  bob::ip::integral(m_src_extra, m_src_integral, true);

  // 3/ Convolution
  double n_elem = m_kernel.numElements();
  for(int y=0; y<src.extent(0); ++y)
    for(int x=0; x<src.extent(1); ++x)
    {
      // Computes the threshold associated to the current location
      // Integral image is used to speed up the process
      blitz::Array<double,2> src_slice = m_src_extra(
        blitz::Range(y,y+2*(int)m_radius_y), blitz::Range(x,x+2*(int)m_radius_x));
      double threshold = (m_src_integral(y,x) + 
          m_src_integral(y+2*(int)m_radius_y+1,x+2*(int)m_radius_x+1) - 
          m_src_integral(y,x+2*(int)m_radius_x+1) - 
          m_src_integral(y+2*(int)m_radius_y+1,x)
        ) / n_elem;
      // Computes the weighted Gaussian kernel at this location
      // a/ M1 is the set of pixels whose values are above the threshold
      if( blitz::sum(src_slice >= threshold) >= n_elem/2.)
        m_kernel_weighted = blitz::where(src_slice >= threshold, m_kernel, 0.);
      // b/ M1 is the set of pixels whose values are below the threshold
      else
        m_kernel_weighted = blitz::where(src_slice < threshold, m_kernel, 0.);
      // Normalizes the kernel
      m_kernel_weighted /= blitz::sum(m_kernel_weighted);
      // Convolves: This is indeed not a real convolution but a multiplication,
      // as it seems that the authors aim at exclusively using the M1 part
      dst(y,x) = blitz::sum(src_slice * m_kernel_weighted);
    }
}
