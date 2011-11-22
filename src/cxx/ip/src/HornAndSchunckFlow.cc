/**
 * @file cxx/ip/src/HornAndSchunckFlow.cc
 * @date Wed Mar 16 15:01:13 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Defines the HornAndSchunckFlow methods
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

#include "core/array_assert.h"
#include "sp/convolution.h"
#include "ip/HornAndSchunckFlow.h"

namespace of = Torch::ip::optflow;

static const double LAPLACIAN_014_KERNEL_DATA[] = {0,.25,0,.25,0,.25,0,.25,0};
static const blitz::Array<double,2> LAPLACIAN_014_KERNEL(const_cast<double*>(LAPLACIAN_014_KERNEL_DATA), blitz::shape(3,3), blitz::neverDeleteData);

void of::laplacian_avg_hs_opencv(const blitz::Array<double,2>& input,
    blitz::Array<double,2>& output) {
  Torch::sp::convolve(input, LAPLACIAN_014_KERNEL, output,
      sp::Convolution::Same, sp::Convolution::Mirror);
}

static const double _12 = 1./12.;
static const double _6 = 1./6.;
static const double LAPLACIAN_12_KERNEL_DATA[] = {_12,_6,_12,_6,0,_6,_12,_6,_12};
static const blitz::Array<double,2> LAPLACIAN_12_KERNEL(const_cast<double*>(LAPLACIAN_12_KERNEL_DATA), blitz::shape(3,3), blitz::neverDeleteData);

void of::laplacian_avg_hs(const blitz::Array<double,2>& input,
    blitz::Array<double,2>& output) {
  Torch::sp::convolve(input, LAPLACIAN_12_KERNEL, output,
      sp::Convolution::Same, sp::Convolution::Mirror);
}
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
  double a2 = std::pow(alpha, 2);
  for (size_t i=0; i<iterations; ++i) {
    of::laplacian_avg_hs(u0, m_u);
    of::laplacian_avg_hs(v0, m_v);
    m_cterm = (m_ex*m_u + m_ey*m_v + m_et) / 
      (blitz::pow2(m_ex) + blitz::pow2(m_ey) + a2);
    u0 = m_u - m_ex*m_cterm;
    v0 = m_v - m_ey*m_cterm;
  }
}

void of::VanillaHornAndSchunckFlow::evalEc2
(const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(u, m_u);

  laplacian_avg_hs(u, m_u);
  laplacian_avg_hs(v, m_u);
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
    of::laplacian_avg_hs_opencv(u0, m_u);
    of::laplacian_avg_hs_opencv(v0, m_v);
    m_cterm = (m_ex*m_u + m_ey*m_v + m_et) / 
      (blitz::pow2(m_ex) + blitz::pow2(m_ey) + a2);
    u0 = m_u - m_ex*m_cterm;
    v0 = m_v - m_ey*m_cterm;
  }
}

void of::HornAndSchunckFlow::evalEc2
(const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) const {
  
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(u, error);
  Torch::core::array::assertSameShape(u, m_u);

  laplacian_avg_hs_opencv(u, m_u);
  laplacian_avg_hs_opencv(v, m_u);
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
      int i_ = i - u(j,i); //flow adjustment
      if (i_ >= i1.extent(1)) continue; //cannot project
      int j_ = j - v(j,i); //flow adjustment
      if (j_ >= i1.extent(0)) continue; //cannot project
      error(j,i) = i2(j_,i_) - i1(j,i);
    }
  }
}
