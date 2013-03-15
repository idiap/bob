/**
 * @file sp/cxx/FFT1DNaive.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Fourier Transform
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

#include <bob/sp/FFT1DNaive.h>
#include <bob/core/assert.h>

bob::sp::detail::FFT1DNaiveAbstract::FFT1DNaiveAbstract(const size_t length):
  m_length(length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::FFT1DNaiveAbstract::FFT1DNaiveAbstract(
    const bob::sp::detail::FFT1DNaiveAbstract& other):
  m_length(other.m_length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::detail::FFT1DNaiveAbstract::~FFT1DNaiveAbstract()
{
}

bob::sp::detail::FFT1DNaiveAbstract& 
bob::sp::detail::FFT1DNaiveAbstract::operator=(const FFT1DNaiveAbstract& other)
{
  if (this != &other) {
    reset(other.m_length);
  }
  return *this;
}

bool bob::sp::detail::FFT1DNaiveAbstract::operator==(const bob::sp::detail::FFT1DNaiveAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::detail::FFT1DNaiveAbstract::operator!=(const bob::sp::detail::FFT1DNaiveAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::detail::FFT1DNaiveAbstract::reset(const size_t length)
{
  // Reset if required
  if (m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}

void bob::sp::detail::FFT1DNaiveAbstract::setLength(const size_t length)
{
  reset(length);
}
  
void bob::sp::detail::FFT1DNaiveAbstract::reset()
{
  // Precompute working array to save computation time
  initWorkingArray();
}

void bob::sp::detail::FFT1DNaiveAbstract::initWorkingArray() 
{
  m_wsave.resize(m_length);
  std::complex<double> J(0.,1.);
  blitz::firstIndex i;
  m_wsave = exp(-(J * 2. * (M_PI * i))/(double)m_length);
}

bob::sp::detail::FFT1DNaive::FFT1DNaive(const size_t length):
  bob::sp::detail::FFT1DNaiveAbstract(length)
{
}

bob::sp::detail::FFT1DNaive::FFT1DNaive(const bob::sp::detail::FFT1DNaive& other):
  bob::sp::detail::FFT1DNaiveAbstract(other)
{
}

bob::sp::detail::FFT1DNaive::~FFT1DNaive()
{
}

void bob::sp::detail::FFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::FFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the FFT
  dst = 0.;
  int ind; // index in the working array using the periodicity of exp(J*x)
  for (int k=0; k<(int)m_length; ++k) {
    for (int n=0; n<(int)m_length; ++n) {
      ind = (n*k) % (int)m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
}


bob::sp::detail::IFFT1DNaive::IFFT1DNaive(const size_t length):
  bob::sp::detail::FFT1DNaiveAbstract(length)
{
}

bob::sp::detail::IFFT1DNaive::IFFT1DNaive(const bob::sp::detail::IFFT1DNaive& other):
  bob::sp::detail::FFT1DNaiveAbstract(other)
{
}

bob::sp::detail::IFFT1DNaive::~IFFT1DNaive()
{
}

void bob::sp::detail::IFFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  bob::core::array::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  bob::core::array::assertSameShape(src, shape);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void bob::sp::detail::IFFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the IFFT
  dst = 0.;
  int ind;
  for (int k=0; k<(int)m_length; ++k) {
    for (int n=0; n<(int)m_length; ++n) {
      ind = (((-n*k) % (int)m_length ) + (int)m_length) % (int)m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
  dst /= (double)m_length;
}
