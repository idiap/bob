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
#include <fftw3.h>

namespace ca = bob::core::array;
namespace sp = bob::sp;

sp::DCT1DAbstract::DCT1DAbstract( const int length):
  m_length(length)
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

void sp::DCT1DAbstract::cleanup() {
}

sp::DCT1D::DCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::DCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  ca::assertCZeroBaseContiguous(src);

  // Check output
  ca::assertCZeroBaseContiguous(dst);
  ca::assertSameShape( dst, src);

  // Reinterpret cast to fftw format
  double* src_ = const_cast<double*>(src.data());
  double* dst_ = dst.data();
  
  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized 
  // for large arrays
  p = fftw_plan_r2r_1d(src.extent(0), src_, dst_, FFTW_REDFT10, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Normalize
  dst(0) *= m_sqrt_1byl/2.;
  if(dst.extent(0)>1)
  {
    blitz::Range r_dst(1, dst.ubound(0) );
    dst(r_dst) *= m_sqrt_2byl/2.;
  }
}


sp::IDCT1D::IDCT1D( const int length):
  sp::DCT1DAbstract::DCT1DAbstract(length)
{
}

void sp::IDCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  ca::assertCZeroBaseContiguous(src);

  // Check output
  ca::assertCZeroBaseContiguous(dst);
  ca::assertSameShape( dst, src);

  // Copy content from src to dst
  dst = src;

  // Normalize
  dst(0) /= m_sqrt_1l;
  if(dst.extent(0)>1)
  {
    blitz::Range r_dst(1, dst.ubound(0) );
    dst(r_dst) /= m_sqrt_2l;
  }

  // Reinterpret cast to fftw format
  double* dst_ = dst.data();
 
  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized 
  // for large arrays
  p = fftw_plan_r2r_1d(src.extent(0), dst_, dst_, FFTW_REDFT01, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
}

