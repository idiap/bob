/**
 * @file cxx/sp/src/DCT1DNaive.cc
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Fast Cosine Transform
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

#include "sp/DCT1DNaive.h"
#include "core/array_assert.h"

namespace ca = bob::core::array;
namespace spd = bob::sp::detail;

spd::DCT1DNaiveAbstract::DCT1DNaiveAbstract( const int length):
  m_length(length), m_wsave(0)
{
  // Initialize working array and normalization factors
  reset();
}

void spd::DCT1DNaiveAbstract::reset(const int length)
{
  // Reset if required
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}
 
void spd::DCT1DNaiveAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
  // Precompute working array to save computation time
  initWorkingArray();
}

spd::DCT1DNaiveAbstract::~DCT1DNaiveAbstract()
{
}

void spd::DCT1DNaiveAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1l=sqrt(1./m_length);
  m_sqrt_2l=sqrt(2./m_length);
}

void spd::DCT1DNaiveAbstract::initWorkingArray() 
{
  int n_wsave = 4*m_length;
  m_wsave.resize(n_wsave);
  blitz::firstIndex i;
  m_wsave = cos(M_PI/(2*m_length)*i);
}

spd::DCT1DNaive::DCT1DNaive( const int length):
  spd::DCT1DNaiveAbstract::DCT1DNaiveAbstract(length)
{
}

void spd::DCT1DNaive::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
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

void spd::DCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  dst = 0.;
  int ind; // index in the working array using the periodicity of cos()
  double val;
  for( int k=0; k<m_length; ++k) {
    val = 0.;
    for( int n=0; n<m_length; ++n) {
      ind = ((2*n+1)*k) % (4*m_length);
      val += src(n) * m_wsave(ind);
    }
    dst(k) = val * (k==0?m_sqrt_1l:m_sqrt_2l);
  }
}


spd::IDCT1DNaive::IDCT1DNaive( const int length):
  spd::DCT1DNaiveAbstract::DCT1DNaiveAbstract(length)
{
}

void spd::IDCT1DNaive::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
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

void spd::IDCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  // Process n==0 with different normalization factor separately
  dst = m_sqrt_1l * src(0) * m_wsave(0);
  // Process n==1 to length
  int ind;
  for( int k=0; k<m_length; ++k) {
    for( int n=1; n<m_length; ++n) {
      ind = ((2*k+1)*n) % (4*m_length);
      dst(k) += m_sqrt_2l * src(n) * m_wsave(ind);
    }
  }
}
