/**
 * @file cxx/sp/src/DCT1D.cc
 * @date Wed Apr 6 14:02:12 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Cosine Transform using FFTPACK
 * functions
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

#include "sp/DCT1D.h"
#include "core/array_assert.h"

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);

namespace tca = Torch::core::array;
namespace sp = Torch::sp;

sp::DCT1DAbstract::DCT1DAbstract( const int length):
  m_length(length), m_wsave(0)
{
  // Initialize working array and normalization factors
  reset();
}

void sp::DCT1DAbstract::reset(const int length)
{
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Deallocate memory
    cleanup();
    // Reset given the new height and width
    reset();
  }
}
 
void sp::DCT1DAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();

  // Precompute working array to save computation time
  initWorkingArray();
}

sp::DCT1DAbstract::~DCT1DAbstract()
{
  cleanup();
}

void sp::DCT1DAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1byl=sqrt(1./m_length);
  m_sqrt_2byl=sqrt(2./m_length);
  m_sqrt_1l=sqrt(1.*m_length);
  m_sqrt_2l=sqrt(2.*m_length);
}

void sp::DCT1DAbstract::initWorkingArray() 
{
  int n_wsave = 3*m_length+15;
  m_wsave = new double[n_wsave];
  cosqi_( &m_length, m_wsave);
}

void sp::DCT1DAbstract::cleanup() {
  if(m_wsave)
    delete [] m_wsave;
}

sp::DCT1D::DCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::DCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Compute the FCT
  cosqb_( &m_length, dst.data(), m_wsave);

  // Normalize
  dst(0) *= m_sqrt_1byl/4.;
  blitz::Range r_dst(1, dst.ubound(0) );
  dst(r_dst) *= m_sqrt_2byl/4.;
}


sp::IDCT1D::IDCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::IDCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Normalize
  dst(0) /= m_sqrt_1l;
  blitz::Range r_dst(1, dst.ubound(0) );
  dst(r_dst) /= m_sqrt_2l;

  // Compute the FCT
  cosqf_( &m_length, dst.data(), m_wsave);
}

