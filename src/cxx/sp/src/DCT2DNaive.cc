/**
 * @file src/cxx/sp/src/DCT2DNaive.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 2D Discrete Cosine Transform
 */

#include "sp/DCT2DNaive.h"
#include "core/common.h"

namespace spd = Torch::sp::detail;

spd::DCT2DNaiveAbstract::DCT2DNaiveAbstract( const int height, 
  const int width): m_height(height), m_width(width), 
    m_wsave_h(0), m_wsave_w(0)
{
  // Initialize working arrays and normalization factors
  reset();
}

void spd::DCT2DNaiveAbstract::reset(const int height, const int width)
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
 
void spd::DCT2DNaiveAbstract::reset()
{
  // Precompute some normalization factors
  initNormFactors();
  // Precompute working array to save computation time
  initWorkingArrays();
}

spd::DCT2DNaiveAbstract::~DCT2DNaiveAbstract()
{
}

void spd::DCT2DNaiveAbstract::initNormFactors()
{
  // Precompute multiplicative factors
  m_sqrt_1h=sqrt(1./m_height);
  m_sqrt_2h=sqrt(2./m_height);
  m_sqrt_1w=sqrt(1./m_width);
  m_sqrt_2w=sqrt(2./m_width);
}

void spd::DCT2DNaiveAbstract::initWorkingArrays() 
{
  int n_wsave_h = 4*m_height;
  m_wsave_h.resize(n_wsave_h);
  blitz::firstIndex i;
  m_wsave_h = cos(M_PI/(2*m_height)*i);

  int n_wsave_w = 4*m_width;
  m_wsave_w.resize(n_wsave_w);
  blitz::firstIndex j;
  m_wsave_w = cos(M_PI/(2*m_width)*j);
}

spd::DCT2DNaive::DCT2DNaive( const int height, const int width):
  spd::DCT2DNaiveAbstract::DCT2DNaiveAbstract(height,width)
{
}

void spd::DCT2DNaive::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Check input, inclusive dimension
  Torch::core::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height, m_width);
  Torch::core::assertSameShape(src, shape);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::DCT2DNaive::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Compute the DCT
  dst = 0.;
  // index in the working arrays using the periodicity of cos()
  int ind_h, ind_w;
  for( int p=0; p<m_height; ++p) {
    for( int q=0; q<m_width; ++q) {
      for( int m=0; m<m_height; ++m) {
        for( int n=0; n<m_width; ++n) {
          ind_h = ((2*m+1)*p) % (4*m_height);
          ind_w = ((2*n+1)*q) % (4*m_width);
          dst(p,q) += src(m,n) * m_wsave_h(ind_h) * m_wsave_w(ind_w);
        }
      }
      dst(p,q) *= (p==0?m_sqrt_1h:m_sqrt_2h) * (q==0?m_sqrt_1w:m_sqrt_2w);
    }
  }
}


spd::IDCT2DNaive::IDCT2DNaive( const int height, const int width):
  spd::DCT2DNaiveAbstract::DCT2DNaiveAbstract(height, width)
{
}

void spd::IDCT2DNaive::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Check input, inclusive dimension
  Torch::core::assertZeroBase(src);
  const blitz::TinyVector<int,2> shape(m_height, m_width);
  Torch::core::assertSameShape(src, shape);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::IDCT2DNaive::processNoCheck(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // Compute the IDCT
  // index in the working arrays using the periodicity of cos()
  dst = 0.;
  int ind_h, ind_w;
  for( int p=0; p<m_height; ++p) {
    for( int q=0; q<m_width; ++q) {
      for( int m=0; m<m_height; ++m) {
        for( int n=0; n<m_width; ++n) {
          ind_h = ((2*p+1)*m) % (4*m_height);
          ind_w = ((2*q+1)*n) % (4*m_width);
          dst(p,q) += src(m,n) * m_wsave_h(ind_h) * m_wsave_w(ind_w) *
            (m==0?m_sqrt_1h:m_sqrt_2h) * (n==0?m_sqrt_1w:m_sqrt_2w);
        }
      }
    }
  }
}
