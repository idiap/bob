/**
 * @file src/cxx/sp/src/DCT1DNaive.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 1D Fast Cosine Transform
 */

#include "sp/DCT1DNaive.h"
#include "core/common.h"

namespace spd = Torch::sp::detail;

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
//  for(int i=0; i<n_wsave; ++i)
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
  Torch::core::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  Torch::core::assertSameShape(src, shape);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::DCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  dst = 0.;
  const int length = src.extent(0);
  int ind; // index in the working array using the periodicity of cos()
  for( int k=0; k<length; ++k) {
    for( int n=0; n<length; ++n) {
      ind = ((2*n+1)*k) % (4*length);
      dst(k) += src(n) * m_wsave(ind);
    }
    dst(k) *= (k==0?m_sqrt_1l:m_sqrt_2l);
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
  Torch::core::assertZeroBase(src);
  const blitz::TinyVector<int,1> shape(m_length);
  Torch::core::assertSameShape(src, shape);

  // Check output
  Torch::core::assertCZeroBaseContiguous(dst);
  Torch::core::assertSameShape( dst, src);

  // Process
  processNoCheck(src, dst);
}

void spd::IDCT1DNaive::processNoCheck(const blitz::Array<double,1>& src, 
  blitz::Array<double,1>& dst)
{
  // Compute the DCT
  const int length = src.extent(0);
  // Process n==0 with different normalization factor separately
  dst = m_sqrt_1l * src(0) * m_wsave(0);
  // Process n==1 to length
  int ind;
  for( int k=0; k<length; ++k) {
    for( int n=1; n<length; ++n) {
      ind = ((2*k+1)*n) % (4*length);
      dst(k) += m_sqrt_2l * src(n) * m_wsave(ind);
    }
  }
}
