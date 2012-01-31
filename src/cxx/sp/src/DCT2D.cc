/**
 * @file cxx/sp/src/DCT2D.cc
 * @date Tue Apr 5 19:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
 * functions
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "sp/DCT2D.h"
#include "core/array_assert.h"
#include <fftw3.h>


namespace ca = bob::core::array;
namespace sp = bob::sp;

sp::DCT2DAbstract::DCT2DAbstract( const int height, const int width):
  m_height(height), m_width(width)
{
  reset();
}

sp::DCT2DAbstract::~DCT2DAbstract()
{
  cleanup();
}

void sp::DCT2DAbstract::reset(const int height, const int width)
{
  if( m_height != height && m_width != width) {
    // Update the height and width
    m_height = height;
    m_width = width;
    // Deallocate memory
    cleanup();
    // Reset given the new height and width
    reset();
  }
}
 
void sp::DCT2DAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
}

void sp::DCT2DAbstract::initNormFactors() 
{
  // Precompute multiplicative factors
  m_sqrt_1h=sqrt(1./m_height);
  m_sqrt_2h=sqrt(2./m_height);
  m_sqrt_1w=sqrt(1./m_width);
  m_sqrt_2w=sqrt(2./m_width);
}


void sp::DCT2DAbstract::cleanup() {
}



sp::DCT2D::DCT2D( const int height, const int width):
  sp::DCT2DAbstract::DCT2DAbstract(height, width)
{
}

void sp::DCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
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
  p = fftw_plan_r2r_2d(src.extent(0), src.extent(1), src_, dst_, FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);

  // Rescale the result
  for(int i=0; i<m_height; ++i)
    for(int j=0; j<m_width; ++j)
      dst(i,j) = dst(i,j)/4.*(i==0?m_sqrt_1h:m_sqrt_2h)*(j==0?m_sqrt_1w:m_sqrt_2w);
}


sp::IDCT2D::IDCT2D( const int height, const int width):
  sp::DCT2DAbstract::DCT2DAbstract(height, width)
{
}

void sp::IDCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // check input
  ca::assertCZeroBaseContiguous(src);

  // Check output
  ca::assertCZeroBaseContiguous(dst);
  ca::assertSameShape( dst, src);

  // Normalize
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array and normalize it
    for( int i=0; i<m_height; ++i)
      dst(i,j) = src(i,j)*4/(i==0?m_sqrt_1h:m_sqrt_2h)/(j==0?m_sqrt_1w:m_sqrt_2w);
  }

  // Reinterpret cast to fftw format
  double* dst_ = dst.data();
  
  fftw_plan p;
  // FFTW_ESTIMATE -> The planner is computed quickly but may not be optimized 
  // for large arrays
  p = fftw_plan_r2r_2d(src.extent(0), src.extent(1), dst_, dst_, FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  
  // Rescale the result by the size of the input 
  // (as this is not performed by FFTPACK)
  double norm_factor = 4*m_width*m_height;
  dst /= norm_factor;
}

