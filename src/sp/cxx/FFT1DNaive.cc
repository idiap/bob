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

#include "bob/sp/FFT1DNaive.h"
#include "bob/core/assert.h"

namespace ca = bob::core::array;
namespace spd = bob::sp::detail;

spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract( const int length):
  m_length(length), m_wsave(0)
{
  // Initialize working array and normalization factors
  reset();
}

void spd::FFT1DNaiveAbstract::reset(const int length)
{
  // Reset if required
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}
 
void spd::FFT1DNaiveAbstract::reset()
{
  // Precompute working array to save computation time
  initWorkingArray();
}

spd::FFT1DNaiveAbstract::~FFT1DNaiveAbstract()
{
}

void spd::FFT1DNaiveAbstract::initWorkingArray() 
{
  m_wsave.resize(m_length);
  std::complex<double> J(0.,1.);
  for( int i=0; i<m_length; ++i)
    m_wsave(i) = exp(-(J * 2. * (M_PI * i))/(double)m_length);
}

spd::FFT1DNaive::FFT1DNaive( const int length):
  spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract(length)
{
}

void spd::FFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  ca::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  ca::assertSameShape(src, shape);

  // Check output
  ca::assertCZeroBaseContiguous(dst);
  ca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::FFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the FFT
  dst = 0.;
  int ind; // index in the working array using the periodicity of exp(J*x)
  for( int k=0; k<m_length; ++k) {
    for( int n=0; n<m_length; ++n) {
      ind = (n*k) % m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
}


spd::IFFT1DNaive::IFFT1DNaive( const int length):
  spd::FFT1DNaiveAbstract::FFT1DNaiveAbstract(length)
{
}

void spd::IFFT1DNaive::operator()(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Check input, inclusive dimension
  ca::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  ca::assertSameShape(src, shape);

  // Check output
  ca::assertCZeroBaseContiguous(dst);
  ca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::IFFT1DNaive::processNoCheck(const blitz::Array<std::complex<double>,1>& src, 
  blitz::Array<std::complex<double>,1>& dst)
{
  // Compute the IFFT
  dst = 0.;
  int ind;
  for( int k=0; k<m_length; ++k) {
    for( int n=0; n<m_length; ++n) {
      ind = (((-n*k) % m_length ) + m_length) % m_length;
      dst(k) += src(n) * m_wsave(ind);
    }
  }
  dst /= (double)m_length;
}
