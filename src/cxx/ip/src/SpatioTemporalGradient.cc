/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 01 Sep 2011 08:45:53 CEST
 *
 * @brief Implementation of Spatio-Temporal Gradients as indicated on the
 * equivalent header file.
 */

#include <cmath>
#include "ip/SpatioTemporalGradient.h"
#include "sp/convolution.h"
#include "core/array_assert.h"

namespace ip = Torch::ip;
namespace sp = Torch::sp;

static inline void fastconv(const blitz::Array<double,2>& image,
    const blitz::Array<double,1>& kernel,
    blitz::Array<double,2>& result, int dimension) {
  sp::convolveSep(image, kernel, result, dimension, sp::Convolution::Same,
    sp::Convolution::Mirror);
}

ip::ForwardGradient::ForwardGradient(const blitz::TinyVector<int,2>& shape) :
  m_buffer1(shape),
  m_buffer2(shape),
  m_kernel1(2),
  m_kernel2(2)
{
  m_kernel1 = +1, -1; //normalization factor: sqrt(2)
  m_kernel2 = +1, +1; //normalization factor: sqrt(2)
}

ip::ForwardGradient::ForwardGradient(const ip::ForwardGradient& other) :
  m_buffer1(other.m_buffer1.shape()),
  m_buffer2(other.m_buffer2.shape()),
  m_kernel1(other.m_kernel1.copy()),
  m_kernel2(other.m_kernel2.copy())
{
}

ip::ForwardGradient::~ForwardGradient() { }

ip::ForwardGradient& ip::ForwardGradient::operator= (const ip::ForwardGradient& other) {
  m_buffer1.resize(other.m_buffer1.shape());
  m_buffer2.resize(other.m_buffer2.shape());
  m_kernel1.reference(other.m_kernel1.copy());
  m_kernel2.reference(other.m_kernel2.copy());
  return *this;
}

void ip::ForwardGradient::setShape(const blitz::TinyVector<int,2>& shape) {
  m_buffer1.resize(shape);
  m_buffer2.resize(shape);
}

void ip::ForwardGradient::operator()(const blitz::Array<double,2>& i1,
    const blitz::Array<double,2>& i2, blitz::Array<double,2>& u, 
    blitz::Array<double,2>& v) const {

  // all arrays have to have the same shape
  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(i1, u);
  Torch::core::array::assertSameShape(m_buffer1, i1);

  // Movement along the X direction (extent 1) => u matrix
  fastconv(i1, m_kernel1, m_buffer1, 1); // [-1 +1] * i1
  fastconv(m_buffer1, m_kernel2, u, 0); // [+1 +1]^T([-1 +1]*(i1))
  
  fastconv(i2, m_kernel1, m_buffer1, 1); // [-1 +1] * i2
  fastconv(m_buffer1, m_kernel2, m_buffer2, 0); // [+1 +1]^T * ([-1 +1]*(i2))
  
  u += m_buffer2;
  u /= 2*std::sqrt(2); //normalization factor for unit length

  // Movement along the Y direction (extent 0) => v matrix
  fastconv(i1, m_kernel2, m_buffer1, 1); // [+1 +1] * i1
  fastconv(m_buffer1, m_kernel1, v, 0); // [+1 -1]^T * ([+1 +1]*(i1))
  
  fastconv(i2, m_kernel2, m_buffer1, 1); // [+1 +1] * i2
  fastconv(m_buffer1, m_kernel1, m_buffer2, 0); // [+1 -1]^T * ([+1 +1]*(i2))

  v += m_buffer2;
  v /= 2*std::sqrt(2); //normalization factor for unit length
}

ip::CentralGradient::CentralGradient(const blitz::TinyVector<int,2>& shape) :
  m_buffer1(shape),
  m_buffer2(shape),
  m_kernel1(3),
  m_kernel2(3)
{
  m_kernel1 = +1,  0, -1; //normalization factor: sqrt(2)
  m_kernel2 = +1, +2, +1; //normalization factor: sqrt(6)
}

ip::CentralGradient::CentralGradient(const ip::CentralGradient& other) :
  m_buffer1(other.m_buffer1.shape()),
  m_buffer2(other.m_buffer2.shape()),
  m_kernel1(other.m_kernel1.copy()),
  m_kernel2(other.m_kernel2.copy())
{
}

ip::CentralGradient::~CentralGradient() { }

ip::CentralGradient& ip::CentralGradient::operator= (const ip::CentralGradient& other) {
  m_buffer1.resize(other.m_buffer1.shape());
  m_buffer2.resize(other.m_buffer2.shape());
  m_kernel1.reference(other.m_kernel1.copy());
  m_kernel2.reference(other.m_kernel2.copy());
  return *this;
}

void ip::CentralGradient::setShape(const blitz::TinyVector<int,2>& shape) {
  m_buffer1.resize(shape);
  m_buffer2.resize(shape);
}

void ip::CentralGradient::operator() (const blitz::Array<double,2>& i_prev,
    const blitz::Array<double,2>& i, const blitz::Array<double,2>& i_after,
    blitz::Array<double,2>& u, blitz::Array<double,2>& v) const {
  
  // all arrays have to have the same shape
  Torch::core::array::assertSameShape(i_prev, i);
  Torch::core::array::assertSameShape(i_after, i);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(i, u);
  Torch::core::array::assertSameShape(m_buffer1, i);

  // Movement along the Y direction (extent 0) => v matrix
  fastconv(i_prev, m_kernel2, m_buffer1, 1); // [+1 +2 +1] * i_prev
  fastconv(m_buffer1, m_kernel1, v, 0); // [-1 0 +1]^T * ([+1 +2 +1]*(i_prev))
  
  fastconv(i, m_kernel2, m_buffer1, 1); // [+1 +2 +1] * i
  fastconv(m_buffer1, m_kernel1, m_buffer2, 0); // [-1 0 +1]^T * ([+1 +2 +1]*(i))

  v += 2*m_buffer2;
  
  fastconv(i_after, m_kernel2, m_buffer1, 1); // [+1 +2 +1] * i_after
  fastconv(m_buffer1, m_kernel1, m_buffer2, 0); // [-1 0 +1]^T * ([+1 +2 +1]*(i_after))

  v += m_buffer2;
  v /= 6*std::sqrt(2);

  // Movement along the X direction (extent 1) => u matrix
  fastconv(i_prev, m_kernel1, m_buffer1, 1); // [-1 0 +1] * i_prev
  fastconv(m_buffer1, m_kernel2, u, 0); // [+1 +2 +1]^T * ([-1 0 +1]*(i_prev))
  
  fastconv(i, m_kernel1, m_buffer1, 1); // [-1 0 +1] * i
  fastconv(m_buffer1, m_kernel2, m_buffer2, 0); // [+1 +2 +1]^T * ([-1 0 +1]*(i))
 
  u += 2*m_buffer2;

  fastconv(i_after, m_kernel1, m_buffer1, 1); // [-1 0 +1] * i_after
  fastconv(m_buffer1, m_kernel2, m_buffer2, 0); // [+1 +2 +1]^T * ([-1 0 +1]*(i_after))
  
  u += m_buffer2;
  u /= 6*std::sqrt(2);
}
