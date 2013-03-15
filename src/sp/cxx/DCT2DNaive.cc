/**
 * @file sp/cxx/DCT2DNaive.cc
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Discrete Cosine Transform
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

#include <bob/sp/DCT2DNaive.h>
#include <bob/core/assert.h>

bob::sp::detail::DCT2DNaiveAbstract::DCT2DNaiveAbstract(
    const size_t height, const size_t width): 
  m_height(height), m_width(width)
{
  // Initialize working arrays and normalization factors
  reset();
}

bob::sp::detail::DCT2DNaiveAbstract::DCT2DNaiveAbstract(
    const bob::sp::detail::DCT2DNaiveAbstract& other):
  m_height(other.m_height), m_width(other.m_width)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::DCT2DNaiveAbstract::~DCT2DNaiveAbstract()
{
}

bob::sp::detail::DCT2DNaiveAbstract& 
bob::sp::detail::DCT2DNaiveAbstract::operator=(const DCT2DNaiveAbstract& other)
{
  if (this != &other) {
    reset(other.m_height, other.m_width);
  }
  return *this;
}

bool bob::sp::detail::DCT2DNaiveAbstract::operator==(const bob::sp::detail::DCT2DNaiveAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::detail::DCT2DNaiveAbstract::operator!=(const bob::sp::detail::DCT2DNaiveAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::detail::DCT2DNaiveAbstract::reset(const size_t height, const size_t width)
{
  // Reset if required
  if (m_height != height || m_width != width) {
    // Update
    m_height = height;
    m_width = width;
    // Reset given the new height and width
    reset();
  }
}

void bob::sp::detail::DCT2DNaiveAbstract::setHeight(const size_t height)
{
  reset(height, m_width);
}

void bob::sp::detail::DCT2DNaiveAbstract::setWidth(const size_t width)
{
  reset(m_height, width);
} 
 
void bob::sp::detail::DCT2DNaiveAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
  // Precompute working array to save computation time
  initWorkingArrays();
}

void bob::sp::detail::DCT2DNaiveAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1h = sqrt(1./(int)m_height);
  m_sqrt_2h = sqrt(2./(int)m_height);
  m_sqrt_1w = sqrt(1./(int)m_width);
  m_sqrt_2w = sqrt(2./(int)m_width);
}

void bob::sp::detail::DCT2DNaiveAbstract::initWorkingArrays() 
{
  blitz::firstIndex i;

  m_wsave_h.resize(4*(int)m_height);
  m_wsave_h = cos(M_PI/(2*(int)m_height)*i);

  m_wsave_w.resize(4*(int)m_width);
  m_wsave_w = cos(M_PI/(2*(int)m_width)*i);
}

bob::sp::detail::DCT2DNaive::DCT2DNaive(const size_t height, const size_t width):
  bob::sp::detail::DCT2DNaiveAbstract(height, width)
{
}

bob::sp::detail::DCT2DNaive::DCT2DNaive(const bob::sp::detail::DCT2DNaive& other):
  bob::sp::detail::DCT2DNaiveAbstract(other)
{
}

bob::sp::detail::DCT2DNaive::~DCT2DNaive()
{
}

void bob::sp::detail::DCT2DNaive::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height, m_width);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape(dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::DCT2DNaive::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Compute the DCT
  dst = 0.;
  // index in the working arrays using the periodicity of cos()
  int ind_h, ind_w;
  for (int p=0; p<(int)m_height; ++p) {
    for (int q=0; q<(int)m_width; ++q) {
      for (int m=0; m<(int)m_height; ++m) {
        for (int n=0; n<(int)m_width; ++n) {
          ind_h = ((2*m+1)*p) % (4*(int)m_height);
          ind_w = ((2*n+1)*q) % (4*(int)m_width);
          dst(p,q) += src(m,n) * m_wsave_h(ind_h) * m_wsave_w(ind_w);
        }
      }
      dst(p,q) *= (p==0?m_sqrt_1h:m_sqrt_2h) * (q==0?m_sqrt_1w:m_sqrt_2w);
    }
  }
}

bob::sp::detail::IDCT2DNaive::IDCT2DNaive(const size_t height, const size_t width):
  bob::sp::detail::DCT2DNaiveAbstract::DCT2DNaiveAbstract(height, width)
{
}

bob::sp::detail::IDCT2DNaive::IDCT2DNaive(const bob::sp::detail::IDCT2DNaive& other):
  bob::sp::detail::DCT2DNaiveAbstract(other)
{
}

bob::sp::detail::IDCT2DNaive::~IDCT2DNaive()
{
}

void bob::sp::detail::IDCT2DNaive::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height, m_width);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::IDCT2DNaive::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Compute the IDCT
  // index in the working arrays using the periodicity of cos()
  dst = 0.;
  int ind_h, ind_w;
  for (int p=0; p<(int)m_height; ++p) {
    for (int q=0; q<(int)m_width; ++q) {
      for (int m=0; m<(int)m_height; ++m) {
        for (int n=0; n<(int)m_width; ++n) {
          ind_h = ((2*p+1)*m) % (4*(int)m_height);
          ind_w = ((2*q+1)*n) % (4*(int)m_width);
          dst(p,q) += src(m,n) * m_wsave_h(ind_h) * m_wsave_w(ind_w) *
            (m==0?m_sqrt_1h:m_sqrt_2h) * (n==0?m_sqrt_1w:m_sqrt_2w);
        }
      }
    }
  }
}
