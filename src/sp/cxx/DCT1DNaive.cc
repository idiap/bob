/**
 * @file sp/cxx/DCT1DNaive.cc
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Cosine Transform
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

#include <bob/sp/DCT1DNaive.h>
#include <bob/core/assert.h>

bob::sp::detail::DCT1DNaiveAbstract::DCT1DNaiveAbstract(const size_t length):
  m_length(length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::DCT1DNaiveAbstract::DCT1DNaiveAbstract(
    const bob::sp::detail::DCT1DNaiveAbstract& other):
  m_length(other.m_length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::DCT1DNaiveAbstract::~DCT1DNaiveAbstract()
{
}

bob::sp::detail::DCT1DNaiveAbstract& 
bob::sp::detail::DCT1DNaiveAbstract::operator=(const DCT1DNaiveAbstract& other)
{
  if (this != &other) {
    reset(other.m_length);
  }
  return *this;
}

bool bob::sp::detail::DCT1DNaiveAbstract::operator==(const bob::sp::detail::DCT1DNaiveAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::detail::DCT1DNaiveAbstract::operator!=(const bob::sp::detail::DCT1DNaiveAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::detail::DCT1DNaiveAbstract::reset(const size_t length)
{
  // Reset if required
  if (m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}
 
void bob::sp::detail::DCT1DNaiveAbstract::setLength(const size_t length)
{
  reset(length);
}
 
void bob::sp::detail::DCT1DNaiveAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
  // Precompute working array to save computation time
  initWorkingArray();
}

void bob::sp::detail::DCT1DNaiveAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1l = sqrt(1./(int)m_length);
  m_sqrt_2l = sqrt(2./(int)m_length);
}

void bob::sp::detail::DCT1DNaiveAbstract::initWorkingArray() 
{
  int n_wsave = 4*(int)m_length;
  m_wsave.resize(n_wsave);
  blitz::firstIndex i;
  m_wsave = cos(M_PI/(2*(int)m_length)*i);
}

bob::sp::detail::DCT1DNaive::DCT1DNaive(const size_t length):
  bob::sp::detail::DCT1DNaiveAbstract(length)
{
}

bob::sp::detail::DCT1DNaive::DCT1DNaive(const bob::sp::detail::DCT1DNaive& other):
  bob::sp::detail::DCT1DNaiveAbstract(other)
{
}

bob::sp::detail::DCT1DNaive::~DCT1DNaive()
{
}

void bob::sp::detail::DCT1DNaive::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertSameDimensionLength(src.extent(0), (int)m_length);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape(dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::DCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  dst = 0.;
  int ind; // index in the working array using the periodicity of cos()
  double val;
  for (int k=0; k<(int)m_length; ++k) {
    val = 0.;
    for (int n=0; n<(int)m_length; ++n) {
      ind = ((2*n+1)*k) % (4*(int)m_length);
      val += src(n) * m_wsave(ind);
    }
    dst(k) = val * (k==0?m_sqrt_1l:m_sqrt_2l);
  }
}


bob::sp::detail::IDCT1DNaive::IDCT1DNaive(const size_t length):
  bob::sp::detail::DCT1DNaiveAbstract(length)
{
}

bob::sp::detail::IDCT1DNaive::IDCT1DNaive(const bob::sp::detail::IDCT1DNaive& other):
  bob::sp::detail::DCT1DNaiveAbstract(other)
{
}

bob::sp::detail::IDCT1DNaive::~IDCT1DNaive()
{
}

void bob::sp::detail::IDCT1DNaive::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertSameDimensionLength(src.extent(0), (int)m_length);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape(dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::IDCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  // Process n==0 with different normalization factor separately
  dst = m_sqrt_1l * src(0) * m_wsave(0);
  // Process n==1 to length
  int ind;
  for (int k=0; k<(int)m_length; ++k) {
    for (int n=1; n<(int)m_length; ++n) {
      ind = ((2*k+1)*n) % (4*(int)m_length);
      dst(k) += m_sqrt_2l * src(n) * m_wsave(ind);
    }
  }
}
