/**
 * @file sp/cxx/DCT1D.cc
 * @date Wed Apr 6 14:02:12 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Cosine Transform using FFTPACK
 * functions
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

#include "bob/sp/DCT1D.h"
#include "bob/core/array_assert.h"
#include <fftw3.h>

bob::sp::DCT1DAbstract::DCT1DAbstract( const size_t length):
  m_length(length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::DCT1DAbstract::DCT1DAbstract( const bob::sp::DCT1DAbstract& other):
  m_length(other.m_length)
{
  // Initialize working array and normalization factors
  reset();
}

bob::sp::DCT1DAbstract::~DCT1DAbstract()
{
}

const bob::sp::DCT1DAbstract& bob::sp::DCT1DAbstract::operator=(const DCT1DAbstract& other)
{
  if(this != &other)
  {
    reset(other.m_length);
  }
  return *this;
}

bool bob::sp::DCT1DAbstract::operator==(const bob::sp::DCT1DAbstract& b) const
{
  return (this->m_length == b.m_length);
}

bool bob::sp::DCT1DAbstract::operator!=(const bob::sp::DCT1DAbstract& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT1DAbstract::reset(const size_t length)
{
  if( m_length != length) {
    // Update the length
    m_length = length;
    // Reset given the new height and width
    reset();
  }
}

void bob::sp::DCT1DAbstract::setLength(const size_t length)
{
  reset(length);
}
 
void bob::sp::DCT1DAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
}

void bob::sp::DCT1DAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1byl=sqrt(1./(double)m_length);
  m_sqrt_2byl=sqrt(2./(double)m_length);
  m_sqrt_1l=sqrt(1.*(double)m_length);
  m_sqrt_2l=sqrt(2.*(double)m_length);
}


bob::sp::DCT1D::DCT1D( const size_t length):
  bob::sp::DCT1DAbstract(length)
{
}

bob::sp::DCT1D::DCT1D( const bob::sp::DCT1D& other):
  bob::sp::DCT1DAbstract(other)
{
}

bob::sp::DCT1D::~DCT1D()
{
}

const bob::sp::DCT1D& bob::sp::DCT1D::operator=(const DCT1D& other)
{
  if(this != &other)
  {
    bob::sp::DCT1DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::DCT1D::operator==(const bob::sp::DCT1D& b) const
{
  return (bob::sp::DCT1DAbstract::operator==(b));
}

bool bob::sp::DCT1D::operator!=(const bob::sp::DCT1D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::DCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
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


bob::sp::IDCT1D::IDCT1D( const size_t length):
  bob::sp::DCT1DAbstract(length)
{
}

bob::sp::IDCT1D::IDCT1D( const bob::sp::IDCT1D& other):
  bob::sp::DCT1DAbstract(other)
{
}

bob::sp::IDCT1D::~IDCT1D()
{
}

const bob::sp::IDCT1D& bob::sp::IDCT1D::operator=(const IDCT1D& other)
{
  if(this != &other)
  {
    bob::sp::DCT1DAbstract::operator=(other);
  }
  return *this;
}

bool bob::sp::IDCT1D::operator==(const bob::sp::IDCT1D& b) const
{
  return (bob::sp::DCT1DAbstract::operator==(b));
}

bool bob::sp::IDCT1D::operator!=(const bob::sp::IDCT1D& b) const
{
  return !(this->operator==(b));
}

void bob::sp::IDCT1D::operator()(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // check input
  bob::core::array::assertCZeroBaseContiguous(src);

  // Check output
  bob::core::array::assertCZeroBaseContiguous(dst);
  bob::core::array::assertSameShape( dst, src);

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

