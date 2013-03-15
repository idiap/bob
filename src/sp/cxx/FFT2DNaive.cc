/**
 * @file sp/cxx/FFT2DNaive.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Fast Fourier Transform
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

#include <bob/sp/FFT2DNaive.h>
#include <bob/core/assert.h>

bob::sp::detail::FFT2DNaiveAbstract::FFT2DNaiveAbstract(
    const size_t height, const size_t width):
  m_height(height), m_width(width)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::FFT2DNaiveAbstract::FFT2DNaiveAbstract(
    const bob::sp::detail::FFT2DNaiveAbstract& other):
  m_height(other.m_height), m_width(other.m_width)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::FFT2DNaiveAbstract::~FFT2DNaiveAbstract()
{
}

bob::sp::detail::FFT2DNaiveAbstract& 
bob::sp::detail::FFT2DNaiveAbstract::operator=(const FFT2DNaiveAbstract& other)
{
  if (this != &other) {
    reset(other.m_height, other.m_width);
  }
  return *this;
}

bool bob::sp::detail::FFT2DNaiveAbstract::operator==(const bob::sp::detail::FFT2DNaiveAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::detail::FFT2DNaiveAbstract::operator!=(const bob::sp::detail::FFT2DNaiveAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::detail::FFT2DNaiveAbstract::reset(const size_t height, const size_t width)
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

void bob::sp::detail::FFT2DNaiveAbstract::setHeight(const size_t height)
{
  reset(height, m_width);
}

void bob::sp::detail::FFT2DNaiveAbstract::setWidth(const size_t width)
{
  reset(m_height, width);
}
 
void bob::sp::detail::FFT2DNaiveAbstract::reset()
{
  // Precompute working array to save computation time
  initWorkingArrays();
}

void bob::sp::detail::FFT2DNaiveAbstract::initWorkingArrays() 
{
  blitz::firstIndex i;
  std::complex<double> J(0.,1.);
  if (m_wsave_h.extent(0) != (int)m_height)
    m_wsave_h.resize(m_height);
  m_wsave_h = exp(-(J * 2. * (M_PI * i))/(double)m_height);
  if (m_wsave_w.extent(0) != (int)m_width)
    m_wsave_w.resize(m_width);
  m_wsave_w = exp(-(J * 2. * (M_PI * i))/(double)m_width);
}

bob::sp::detail::FFT2DNaive::FFT2DNaive(const size_t height, const size_t width):
  bob::sp::detail::FFT2DNaiveAbstract::FFT2DNaiveAbstract(height,width)
{
}

bob::sp::detail::FFT2DNaive::FFT2DNaive(const bob::sp::detail::FFT2DNaive& other):
  bob::sp::detail::FFT2DNaiveAbstract(other)
{
}

bob::sp::detail::FFT2DNaive::~FFT2DNaive()
{
}

void bob::sp::detail::FFT2DNaive::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height,m_width);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape(dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::FFT2DNaive::processNoCheck(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Compute the FFT
  dst = 0.;
  int ind_yv, ind_xu; // indices in the working array using the periodicity of exp(J*x)
  for (int y=0; y<(int)m_height; ++y)
    for (int x=0; x<(int)m_width; ++x)
      for (int v=0; v<(int)m_height; ++v)
        for (int u=0; u<(int)m_width; ++u) {
          ind_yv = (y*v) % (int)m_height;
          ind_xu = (x*u) % (int)m_width;
          dst(y,x) += src(v,u) * m_wsave_h(ind_yv) * m_wsave_w(ind_xu);
        }
}


bob::sp::detail::IFFT2DNaive::IFFT2DNaive(const size_t height, const size_t width):
  bob::sp::detail::FFT2DNaiveAbstract::FFT2DNaiveAbstract(height,width)
{
}

bob::sp::detail::IFFT2DNaive::IFFT2DNaive(const bob::sp::detail::IFFT2DNaive& other):
  bob::sp::detail::FFT2DNaiveAbstract(other)
{
}

bob::sp::detail::IFFT2DNaive::~IFFT2DNaive()
{
}

void bob::sp::detail::IFFT2DNaive::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height,m_width);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::IFFT2DNaive::processNoCheck(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Compute the IFFT
  dst = 0.;
  int ind_yv, ind_xu; // indices in the working array using the periodicity of exp(J*x)
  for (int y=0; y<(int)m_height; ++y)
    for (int x=0; x<(int)m_width; ++x)
      for (int v=0; v<(int)m_height; ++v)
        for (int u=0; u<(int)m_width; ++u) {
          ind_yv = (((-y*v) % (int)m_height) + (int)m_height) % (int)m_height;
          ind_xu = (((-x*u) % (int)m_width) + (int)m_width) % (int)m_width;
          dst(y,x) += src(v,u) * m_wsave_h(ind_yv) * m_wsave_w(ind_xu);
        }
  dst /= (double)(m_height*m_width);
}
