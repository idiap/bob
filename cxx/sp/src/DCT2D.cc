/**
 * @file cxx/sp/src/DCT2D.cc
 * @date Tue Apr 5 19:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
 * functions
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

#include "sp/DCT2D.h"
#include "core/array_assert.h"
#include <fftw3.h>


bob::sp::DCT2DAbstract::DCT2DAbstract( const size_t height, const size_t width):
  m_height(height), m_width(width)
{
  reset();
}

bob::sp::DCT2DAbstract::DCT2DAbstract( const bob::sp::DCT2DAbstract& other):
  m_height(other.m_height), m_width(other.m_width)
{
  reset();
}

bob::sp::DCT2DAbstract::~DCT2DAbstract()
{
}

const bob::sp::DCT2DAbstract& bob::sp::DCT2DAbstract::operator=(const DCT2DAbstract& other)
{
  if(this != &other)
  {
    reset(other.m_height, other.m_width);
  }
  return *this;
}

bool bob::sp::DCT2DAbstract::operator==(const bob::sp::DCT2DAbstract& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width);
}

bool bob::sp::DCT2DAbstract::operator!=(const bob::sp::DCT2DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT2DAbstract::reset(const size_t height, const size_t width)
{
  if( m_height != height && m_width != width) {
    // Update the height and width
    m_height = height;
    m_width = width;
    // Reset given the new height and width
    reset();
  }
}

void bob::sp::DCT2DAbstract::setHeight(const size_t height)
{
  m_height = height;
  reset();
}

void bob::sp::DCT2DAbstract::setWidth(const size_t width)
{
  m_width = width;
  reset();
}

void bob::sp::DCT2DAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
}

void bob::sp::DCT2DAbstract::initNormFactors() 
{
  // Precompute multiplicative factors
  m_sqrt_1h=sqrt(1./(double)m_height);
  m_sqrt_2h=sqrt(2./(double)m_height);
  m_sqrt_1w=sqrt(1./(double)m_width);
  m_sqrt_2w=sqrt(2./(double)m_width);
}


bob::sp::DCT2D::DCT2D( const size_t height, const size_t width):
  bob::sp::DCT2DAbstract(height, width)
{
}

bob::sp::DCT2D::DCT2D( const bob::sp::DCT2D& other):
  bob::sp::DCT2DAbstract(other)
{
}

bob::sp::DCT2D::~DCT2D()
{
}

const bob::sp::DCT2D& bob::sp::DCT2D::operator=(const DCT2D& other)
{
  if(this != &other)
  {
    bob::sp::DCT2DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::DCT2D::operator==(const bob::sp::DCT2D& b) const
{
  return (bob::sp::DCT2DAbstract::operator==(b));
}

bool bob::sp::DCT2D::operator!=(const bob::sp::DCT2D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // check input
  bob::core::array::assertCZeroBaseContiguous(src);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

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
  for(int i=0; i<(int)m_height; ++i)
    for(int j=0; j<(int)m_width; ++j)
      dst(i,j) = dst(i,j)/4.*(i==0?m_sqrt_1h:m_sqrt_2h)*(j==0?m_sqrt_1w:m_sqrt_2w);
}


bob::sp::IDCT2D::IDCT2D( const size_t height, const size_t width):
  bob::sp::DCT2DAbstract::DCT2DAbstract(height, width)
{
}

bob::sp::IDCT2D::IDCT2D( const bob::sp::IDCT2D& other):
  bob::sp::DCT2DAbstract(other)
{
}

bob::sp::IDCT2D::~IDCT2D()
{
}

const bob::sp::IDCT2D& bob::sp::IDCT2D::operator=(const IDCT2D& other)
{
  if(this != &other)
  {
    bob::sp::DCT2DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::IDCT2D::operator==(const bob::sp::IDCT2D& b) const
{
  return (bob::sp::DCT2DAbstract::operator==(b));
}

bool bob::sp::IDCT2D::operator!=(const bob::sp::IDCT2D& b) const
{
  return !(this->operator==(b));
}


void bob::sp::IDCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // check input
  bob::core::array::assertCZeroBaseContiguous(src);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

  // Normalize
  for(int j=0; j<(int)m_width; ++j)
  {
    // Copy the column into the C array and normalize it
    for( int i=0; i<(int)m_height; ++i)
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
  double norm_factor = 4.*(int)m_width*(int)m_height;
  dst /= norm_factor;
}

