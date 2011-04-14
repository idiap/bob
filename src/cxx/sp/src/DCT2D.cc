/**
 * @file src/cxx/sp/src/DCT2D.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK 
 * functions
 */

#include "sp/DCT2D.h"
#include "core/array_assert.h"

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);

namespace tca = Torch::core::array;
namespace sp = Torch::sp;

sp::DCT2DAbstract::DCT2DAbstract( const int height, const int width):
  m_height(height), m_width(width), m_wsave_w(0), m_wsave_h(0), m_col_tmp(0)
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

  // Precompute working arrays to save computation time
  initWorkingArrays();
}

void sp::DCT2DAbstract::initNormFactors() 
{
  // Precompute multiplicative factors
  m_sqrt_1h=sqrt(1./m_height);
  m_sqrt_2h=sqrt(2./m_height);
  m_sqrt_1w=sqrt(1./m_width);
  m_sqrt_2w=sqrt(2./m_width);
}

void sp::DCT2DAbstract::initWorkingArrays() 
{
  int n_wsave_h = 3*m_height+15;
  m_wsave_h = new double[n_wsave_h];
  cosqi_( &m_height, m_wsave_h);

  int n_wsave_w = 3*m_width+15;
  m_wsave_w = new double[n_wsave_w];
  cosqi_( &m_width, m_wsave_w);

  m_col_tmp = new double[m_height];
}

void sp::DCT2DAbstract::cleanup() {
  if(m_wsave_w)
    delete [] m_wsave_w;
  if(m_wsave_h)
    delete [] m_wsave_h;
  if(m_col_tmp)
    delete [] m_col_tmp;
}



sp::DCT2D::DCT2D( const int height, const int width):
  sp::DCT2DAbstract::DCT2DAbstract(height, width)
{
}

void sp::DCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Apply 1D FCT to each column of the 2D array (array(id_row,id_column))
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array
    for( int i=0; i<m_height; ++i)
      m_col_tmp[i] = src(i,j);

    // Compute the FCT of one column
    cosqb_( &m_height, m_col_tmp, m_wsave_h);

    // Update the column
    for( int i=0; i<m_height; ++i)
      dst(i,j) = m_col_tmp[i];
  }

  // Apply 1D FCT to each row of the resulting matrix
  for(int i=0; i<m_height; ++i)
  {
    // Compute the FCT of one row
    cosqb_( &m_width, &(dst.data()[i*m_width]), m_wsave_w);
  }

  // Rescale the result
  for(int i=0; i<m_height; ++i)
    for(int j=0; j<m_width; ++j)
      dst(i,j) = dst(i,j)/16.*(i==0?m_sqrt_1h:m_sqrt_2h)*(j==0?m_sqrt_1w:m_sqrt_2w);
}


sp::IDCT2D::IDCT2D( const int height, const int width):
  sp::DCT2DAbstract::DCT2DAbstract(height, width)
{
}

void sp::IDCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // check input
  tca::assertZeroBase(src);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Apply 1D inverse FCT to each column of the 2D array (array(id_row,id_column))
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array and normalize it
    for( int i=0; i<m_height; ++i)
      m_col_tmp[i] = src(i,j)*16/(i==0?m_sqrt_1h:m_sqrt_2h)/(j==0?m_sqrt_1w:m_sqrt_2w);

    // Compute the FCT of one column
    cosqf_( &m_height, m_col_tmp, m_wsave_h);

    // Update the column
    for( int i=0; i<m_height; ++i)
      dst(i,j) = m_col_tmp[i];
  }

  // Apply 1D FCT to each row of the resulting matrix
  for(int i=0; i<m_height; ++i)
  {
    // Compute the FCT of one row
    cosqf_( &m_width, &(dst.data()[i*m_width]), m_wsave_w);
  }

  // Rescale the result by the size of the input 
  // (as this is not performed by FFTPACK)
  double norm_factor = 16*m_width*m_height;
  dst /= norm_factor;
}

