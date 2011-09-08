/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 08 Sep 2011 14:28:02 CEST
 *
 * @brief Defines the HornAndSchunckFlow methods
 */

#include "ip/HornAndSchunckFlow.h"
#include "core/array_assert.h"

namespace of = Torch::ip::optflow;

of::VanillaHornAndSchunckFlow::VanillaHornAndSchunckFlow
(const blitz::TinyVector<int,2>& shape) :
  m_gradient(shape),
  m_ex(shape),
  m_ey(shape),
  m_et(shape),
  m_u(shape),
  m_v(shape),
  m_cterm(shape)
{
}

of::VanillaHornAndSchunckFlow::~VanillaHornAndSchunckFlow() { }

void of::VanillaHornAndSchunckFlow::setShape
(const blitz::TinyVector<int,2>& shape) {
  m_gradient.setShape(shape);
  m_ex.resize(shape);
  m_ey.resize(shape);
  m_et.resize(shape);
  m_u.resize(shape);
  m_v.resize(shape);
  m_cterm.resize(shape);
}

void of::VanillaHornAndSchunckFlow::operator() (double alpha,
    size_t iterations, const blitz::Array<double,2>& i1, 
    const blitz::Array<double,2>& i2, blitz::Array<double,2>& u0,
    blitz::Array<double,2>& v0) const {

  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(i1, m_ex);
  Torch::core::array::assertSameShape(u0, m_u);
  Torch::core::array::assertSameShape(v0, m_v);

  m_gradient(i1, i2, m_ex, m_ey, m_et);
  m_ex *= 0.25;
  m_ey *= 0.25;
  m_et *= 0.25;
  double a2 = std::pow(alpha, 2);
  for (size_t i=0; i<iterations; ++i) {
    ip::laplacian_12(u0, m_u); m_u /= -12.;
    ip::laplacian_12(v0, m_v); m_v /= -12.;
    m_cterm = (m_ex*m_u + m_ey*m_v + m_et) / 
      (blitz::pow2(m_ex) + blitz::pow2(m_ey) + a2);
    u0 = m_u - m_ex*m_cterm;
    v0 = m_v - m_ey*m_cterm;
  }
}

void of::VanillaHornAndSchunckFlow::evalEc2
(const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(u, m_u);

  laplacian_12(u, m_u); m_u /= -12.;
  laplacian_12(v, m_u); m_v /= -12.;
  error = blitz::pow2(m_u - u) + blitz::pow2(m_v - v);

}

void of::VanillaHornAndSchunckFlow::evalEb
(const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
 const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(error, m_u);
  m_gradient(i1, i2, m_ex, m_ey, m_et);
  m_ex *= 0.25;
  m_ey *= 0.25;
  m_et *= 0.25;
  error = m_ex*u + m_ey*v + m_et;

}

of::HornAndSchunckFlow::HornAndSchunckFlow
(const blitz::TinyVector<int,2>& shape) :
  m_gradient(shape),
  m_ex(shape),
  m_ey(shape),
  m_et(shape),
  m_u(shape),
  m_v(shape),
  m_cterm(shape)
{
}

of::HornAndSchunckFlow::~HornAndSchunckFlow() { }

void of::HornAndSchunckFlow::setShape
(const blitz::TinyVector<int,2>& shape) {
  m_gradient.setShape(shape);
  m_ex.resize(shape);
  m_ey.resize(shape);
  m_et.resize(shape);
  m_u.resize(shape);
  m_v.resize(shape);
  m_cterm.resize(shape);
}

void of::HornAndSchunckFlow::operator() (double alpha,
    size_t iterations, const blitz::Array<double,2>& i1, 
    const blitz::Array<double,2>& i2, const blitz::Array<double,2>& i3,
    blitz::Array<double,2>& u0, blitz::Array<double,2>& v0) const {

  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(i2, i3);
  Torch::core::array::assertSameShape(i1, m_ex);
  Torch::core::array::assertSameShape(u0, m_u);
  Torch::core::array::assertSameShape(v0, m_v);

  m_gradient(i1, i2, i3, m_ex, m_ey, m_et);
  double a2 = std::pow(alpha, 2);
  for (size_t i=0; i<iterations; ++i) {
    ip::laplacian_014(u0, m_u);
    ip::laplacian_014(v0, m_v);
    m_cterm = (m_ex*m_u + m_ey*m_v + m_et) / 
      (blitz::pow2(m_ex) + blitz::pow2(m_ey) + a2);
    u0 = m_u - m_ex*m_cterm;
    v0 = m_v - m_ey*m_cterm;
  }
}

void of::HornAndSchunckFlow::evalEc2
(const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(u, m_u);

  laplacian_014(u, m_u);
  laplacian_014(v, m_u);
  error = blitz::pow2(m_u - u) + blitz::pow2(m_v - v);

}

void of::HornAndSchunckFlow::evalEb
(const blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
 const blitz::Array<double,2>& i3, const blitz::Array<double,2>& u,
 const blitz::Array<double,2>& v, blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(i2, i3);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(error, m_u);

  m_gradient(i1, i2, i3, m_ex, m_ey, m_et);
  error = m_ex*u + m_ey*v + m_et;

}

void of::flowError (const blitz::Array<double,2>& i1,
    const blitz::Array<double,2>& i2, const blitz::Array<double,2>& u, 
    const blitz::Array<double,2>& v, blitz::Array<double,2>& error) {
  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(i1, u);
  Torch::core::array::assertSameShape(i1, error);
  error = 0;
  for (int i=0; i<i1.extent(1); ++i) {
    for (int j=0; j<i1.extent(0); ++j) {
      int i_ = i - u(i,j); //flow adjustment
      if (i_ >= i1.extent(1)) continue; //cannot project
      int j_ = j - v(i,j); //flow adjustment
      if (j_ >= i1.extent(0)) continue; //cannot project
      error(i,j) = i2(i_,j_) - i1(i,j);
    }
  }
}
