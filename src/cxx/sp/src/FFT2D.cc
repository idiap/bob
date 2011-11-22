/**
 * @file cxx/sp/src/FFT2D.cc
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
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

#include "sp/FFT2D.h"
#include "core/array_assert.h"

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cffti_( int *n, double *wsave);
extern "C" void cfftf_( int *n, std::complex<double> *x, double *wsave);
extern "C" void cfftb_( int *n, std::complex<double> *x, double *wsave);

namespace tca = Torch::core::array;
namespace sp = Torch::sp;

sp::FFT2DAbstract::FFT2DAbstract( const int height, const int width):
  m_height(height), m_width(width), m_wsave_w(0), m_wsave_h(0), m_col_tmp(0)
{
  reset();
}

sp::FFT2DAbstract::~FFT2DAbstract()
{
  cleanup();
}

void sp::FFT2DAbstract::reset(const int height, const int width)
{
  // Reset if required
  if( m_height != height || m_width != width) {
    // Update the height and width
    m_height = height;
    m_width = width;
    // Deallocate memory
    cleanup();
    // Reset given the new height and width
    reset();
  }
}
 
void sp::FFT2DAbstract::reset()
{
  // Precompute working arrays to save computation time
  initWorkingArrays();
}

void sp::FFT2DAbstract::initWorkingArrays() 
{
  int n_wsave_h = 4*m_height+15;
  m_wsave_h = new double[n_wsave_h];
  cffti_( &m_height, m_wsave_h);

  int n_wsave_w = 4*m_width+15;
  m_wsave_w = new double[n_wsave_w];
  cffti_( &m_width, m_wsave_w);

  m_col_tmp = new std::complex<double>[m_height];
}

void sp::FFT2DAbstract::cleanup() {
  if(m_wsave_w)
    delete [] m_wsave_w;
  if(m_wsave_h)
    delete [] m_wsave_h;
  if(m_col_tmp)
    delete [] m_col_tmp;
}



sp::FFT2D::FFT2D( const int height, const int width):
  sp::FFT2DAbstract::FFT2DAbstract(height, width)
{
}

void sp::FFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Apply 1D FFT to each column of the 2D array (array(id_row,id_column))
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array
    for( int i=0; i<m_height; ++i)
      m_col_tmp[i] = src(i,j);

    // Compute the FFT of one column
    cfftf_( &m_height, m_col_tmp, m_wsave_h);

    // Update the column
    for( int i=0; i<m_height; ++i)
      dst(i,j) = m_col_tmp[i];
  }

  // Apply 1D FFT to each row of the resulting matrix
  for(int i=0; i<m_height; ++i)
  {
    // Compute the FFT of one row
    cfftf_( &m_width, &(dst.data()[i*m_width]), m_wsave_w);
  }
}


sp::IFFT2D::IFFT2D( const int height, const int width):
  sp::FFT2DAbstract::FFT2DAbstract(height, width)
{
}

void sp::IFFT2D::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Apply 1D inverse FFT to each column of the 2D array (array(id_row,id_column))
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array
    for( int i=0; i<m_height; ++i)
      m_col_tmp[i] = src(i,j);

    // Compute the FFT of one column
    cfftb_( &m_height, m_col_tmp, m_wsave_h);

    // Update the column
    for( int i=0; i<m_height; ++i)
      dst(i,j) = m_col_tmp[i];
  }

  // Apply 1D FFT to each row of the resulting matrix
  for(int i=0; i<m_height; ++i)
  {
    // Compute the FFT of one row
    cfftb_( &m_width, &(dst.data()[i*m_width]), m_wsave_w);
  }

  // Rescale the result by the size of the input 
  // (as this is not performed by FFTPACK)
  dst /= static_cast<double>(m_width*m_height);
}

