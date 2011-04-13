/**
 * @file src/cxx/sp/src/FFT2DNaive.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 2D Fast Fourier Transform
 */

#include "sp/FFT2DNaive.h"
#include "core/array_assert.h"

namespace tca = Torch::core::array;
namespace spd = Torch::sp::detail;

spd::FFT2DNaiveAbstract::FFT2DNaiveAbstract( const int height, const int width):
  m_height(height), m_width(width), m_wsave_h(0), m_wsave_w(0)
{
  // Initialize working array and normalization factors
  reset();
}

void spd::FFT2DNaiveAbstract::reset(const int height, const int width)
{
  // Reset if required
  if( m_height != height || m_width != width) {
    // Update
    m_height = height;
    m_width = width;
    // Reset given the new height and width
    reset();
  }
}
 
void spd::FFT2DNaiveAbstract::reset()
{
  // Precompute working array to save computation time
  initWorkingArrays();
}

spd::FFT2DNaiveAbstract::~FFT2DNaiveAbstract()
{
}

void spd::FFT2DNaiveAbstract::initWorkingArrays() 
{
  std::complex<double> J(0.,1.);
  if( m_wsave_h.extent(0) != m_height)
    m_wsave_h.resize(m_height);
  for( int i=0; i<m_height; ++i)
    m_wsave_h(i) = exp(-(J * 2. * (M_PI * i))/(double)m_height);
  if( m_wsave_w.extent(0) != m_width)
    m_wsave_w.resize(m_width);
  for( int i=0; i<m_width; ++i)
    m_wsave_w(i) = exp(-(J * 2. * (M_PI * i))/(double)m_width);
}

spd::FFT2DNaive::FFT2DNaive( const int height, const int width):
  spd::FFT2DNaiveAbstract::FFT2DNaiveAbstract(height,width)
{
}

void spd::FFT2DNaive::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Check input, inclusive dimension
  tca::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height,m_width);
  tca::assertSameShape(src, shape);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::FFT2DNaive::processNoCheck(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Compute the FFT
  dst = 0.;
  int ind_yv, ind_xu; // indices in the working array using the periodicity of exp(J*x)
  for( int y=0; y<m_height; ++y)
    for( int x=0; x<m_width; ++x)
      for( int v=0; v<m_height; ++v)
        for( int u=0; u<m_width; ++u) {
          ind_yv = (y*v) % m_height;
          ind_xu = (x*u) % m_width;
          dst(y,x) += src(v,u) * m_wsave_h(ind_yv) * m_wsave_w(ind_xu);
        }
}


spd::IFFT2DNaive::IFFT2DNaive( const int height, const int width):
  spd::FFT2DNaiveAbstract::FFT2DNaiveAbstract(height,width)
{
}

void spd::IFFT2DNaive::operator()(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Check input, inclusive dimension
  tca::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height,m_width);
  tca::assertSameShape(src, shape);

  // Check output
  tca::assertCZeroBaseContiguous(dst);
  tca::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::IFFT2DNaive::processNoCheck(const blitz::Array<std::complex<double>,2>& src, 
  blitz::Array<std::complex<double>,2>& dst)
{
  // Compute the IFFT
  dst = 0.;
  int ind_yv, ind_xu; // indices in the working array using the periodicity of exp(J*x)
  for( int y=0; y<m_height; ++y)
    for( int x=0; x<m_width; ++x)
      for( int v=0; v<m_height; ++v)
        for( int u=0; u<m_width; ++u) {
          ind_yv = (((-y*v) % m_height) + m_height) % m_height;
          ind_xu = (((-x*u) % m_width) + m_width) % m_width;
          dst(y,x) += src(v,u) * m_wsave_h(ind_yv) * m_wsave_w(ind_xu);
        }
  dst /= (double)(m_height*m_width);
}
