/**
 * @file machine/cxx/IVectorMachine.cc
 * @date Sat Mar 30 21:00:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/machine/IVectorMachine.h>
#include <bob/core/array_copy.h>
#include <bob/core/check.h>
#include <bob/math/linear.h>
#include <bob/math/linsolve.h>

bob::machine::IVectorMachine::IVectorMachine()
{
}

bob::machine::IVectorMachine::IVectorMachine(const boost::shared_ptr<bob::machine::GMMMachine> ubm, 
    const size_t rt, const double variance_threshold):
  m_ubm(ubm), m_rt(rt),
  m_T(getDimCD(),rt), m_sigma(getDimCD()), 
  m_variance_threshold(variance_threshold)
{
  resizePrecompute();
}

bob::machine::IVectorMachine::IVectorMachine(const bob::machine::IVectorMachine& other):
  m_ubm(other.m_ubm), m_rt(other.m_rt), 
  m_T(bob::core::array::ccopy(other.m_T)),
  m_sigma(bob::core::array::ccopy(other.m_sigma)),
  m_variance_threshold(other.m_variance_threshold)
{
  resizePrecompute();
}

bob::machine::IVectorMachine::IVectorMachine(bob::io::HDF5File& config)
{
  load(config);
}

bob::machine::IVectorMachine::~IVectorMachine() {
}

void bob::machine::IVectorMachine::save(bob::io::HDF5File& config) const
{
  config.setArray("m_T", m_T);
  config.setArray("m_sigma", m_sigma);
  config.set("m_variance_threshold", m_variance_threshold);
}

void bob::machine::IVectorMachine::load(bob::io::HDF5File& config) 
{
  //reads all data directly into the member variables 
  m_T.reference(config.readArray<double,2>("m_T"));
  m_rt = m_T.extent(1);
  m_sigma.reference(config.readArray<double,1>("m_sigma"));
  m_variance_threshold = config.read<double>("m_variance_threshold");
  resizePrecompute();
}

void bob::machine::IVectorMachine::resize(const size_t rt)
{
  m_rt = rt;
  m_T.resizeAndPreserve(m_T.extent(0), rt);
  resizePrecompute();
}

bob::machine::IVectorMachine& 
bob::machine::IVectorMachine::operator=(const bob::machine::IVectorMachine& other) 
{
  if (this != &other)
  {
    m_ubm = other.m_ubm;
    m_rt = other.m_rt;
    m_T.reference(bob::core::array::ccopy(other.m_T));
    m_sigma.reference(bob::core::array::ccopy(other.m_sigma));
    m_variance_threshold = other.m_variance_threshold;
    resizePrecompute();
  }
  return *this;
}

bool bob::machine::IVectorMachine::operator==(const IVectorMachine& b) const
{
  return (((m_ubm && b.m_ubm) && *m_ubm == *(b.m_ubm)) || (!m_ubm && !b.m_ubm)) &&
         m_rt == b.m_rt && 
         bob::core::array::isEqual(m_T, b.m_T) &&
         bob::core::array::isEqual(m_sigma, b.m_sigma) &&
         m_variance_threshold == b.m_variance_threshold;
}

bool bob::machine::IVectorMachine::operator!=(const bob::machine::IVectorMachine& b) const 
{
  return !(this->operator==(b));
}

bool bob::machine::IVectorMachine::is_similar_to(const IVectorMachine& b,
  const double r_epsilon, const double a_epsilon) const
{
  // TODO: update with new is_similar_to method
  return (((m_ubm && b.m_ubm) && m_ubm->is_similar_to(*(b.m_ubm), r_epsilon)) || (!m_ubm && !b.m_ubm)) &&
          m_rt == b.m_rt && 
          bob::core::array::isClose(m_T, b.m_T, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_sigma, b.m_sigma, r_epsilon, a_epsilon) &&
          bob::core::isClose(m_variance_threshold, b.m_variance_threshold, r_epsilon, a_epsilon);
}

void bob::machine::IVectorMachine::setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm)
{
  m_ubm = ubm;
  resizePrecompute();
}

void bob::machine::IVectorMachine::setT(const blitz::Array<double,2>& T)
{
  bob::core::array::assertSameShape(m_T, T);
  m_T = T;
  // Update cache
  precompute();
}

void bob::machine::IVectorMachine::setSigma(const blitz::Array<double,1>& sigma)
{
  bob::core::array::assertSameShape(m_sigma, sigma);
  m_sigma = sigma;
  // Update cache
  precompute();
}


void bob::machine::IVectorMachine::setVarianceThreshold(const double thd)
{
  m_variance_threshold = thd;
  // Update cache
  precompute();
}

void bob::machine::IVectorMachine::applyVarianceThreshold()
{
  // Apply variance flooring threshold
  m_sigma = blitz::where(m_sigma < m_variance_threshold, m_variance_threshold, m_sigma);
}

void bob::machine::IVectorMachine::precompute()
{
  if (m_ubm)
  {
    // Apply variance threshold
    applyVarianceThreshold();

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::Range rall = blitz::Range::all();
    const int C = (int)m_ubm->getNGaussians();
    const int D = (int)m_ubm->getNInputs();
    // T_{c}^{T}.sigma_{c}^{-1}
    for (int c=0; c<C; ++c)
    {
      blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
      blitz::Array<double,2> Tc = m_T(blitz::Range(c*D,(c+1)*D-1), rall);
      blitz::Array<double,2> Tct = Tc.transpose(1,0);
      blitz::Array<double,1> sigma_c = m_sigma(blitz::Range(c*D,(c+1)*D-1));
      Tct_sigmacInv = Tct(i,j) / sigma_c(j);
    }

    // T_{c}^{T}.sigma_{c}^{-1}.T_{c}
    for (int c=0; c<C; ++c)
    {
      blitz::Array<double,2> Tc = m_T(blitz::Range(c*D,(c+1)*D-1), rall);
      blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
      blitz::Array<double,2> Tct_sigmacInv_Tc = m_cache_Tct_sigmacInv_Tc(c, rall, rall);
      bob::math::prod(Tct_sigmacInv, Tc, Tct_sigmacInv_Tc);
    }
  }
}

void bob::machine::IVectorMachine::resizePrecompute()
{
  resizeCache();
  resizeTmp();
  precompute();
}

void bob::machine::IVectorMachine::resizeCache()
{
  if (m_ubm)
  {
    const int C = (int)m_ubm->getNGaussians();
    const int D = (int)m_ubm->getNInputs();
    m_cache_Tct_sigmacInv.resize(C, (int)m_rt, D); 
    m_cache_Tct_sigmacInv_Tc.resize(C, (int)m_rt, (int)m_rt);
  }
}

void bob::machine::IVectorMachine::resizeTmp()
{
  m_tmp_d.resize(m_ubm->getNInputs());
  m_tmp_t1.resize(m_rt);
  m_tmp_t2.resize(m_rt);
  m_tmp_tt.resize(m_rt, m_rt);
}

void bob::machine::IVectorMachine::forward(const bob::machine::GMMStats& gs,
  blitz::Array<double,1>& ivector) const
{
  bob::core::array::assertSameDimensionLength(ivector.extent(0), (int)m_rt);
  forward_(gs, ivector);
}

void bob::machine::IVectorMachine::computeIdTtSigmaInvT(
  const bob::machine::GMMStats& gs, blitz::Array<double,2>& output) const
{ 
  // Computes \f$(Id + \sum_{c=1}^{C} N_{i,j,c} T^{T} \Sigma_{c}^{-1} T)\f$
  blitz::Range rall = blitz::Range::all();
  bob::math::eye(output);
  for (int c=0; c<(int)getDimC(); ++c)
    output += gs.n(c) * m_cache_Tct_sigmacInv_Tc(c, rall, rall);
}

void bob::machine::IVectorMachine::computeTtSigmaInvFnorm(
  const bob::machine::GMMStats& gs, blitz::Array<double,1>& output) const
{
  // Computes \f$T^{T} \Sigma^{-1} \sum_{c=1}^{C} (F_c - N_c ubmmean_{c})\f$
  blitz::Range rall = blitz::Range::all();
  output = 0;
  for (int c=0; c<(int)getDimC(); ++c)
  {
    m_tmp_d = gs.sumPx(c,rall) - gs.n(c) * m_ubm->getGaussian(c)->getMean();
    blitz::Array<double,2> Tct_sigmacInv = m_cache_Tct_sigmacInv(c, rall, rall);
    bob::math::prod(Tct_sigmacInv, m_tmp_d, m_tmp_t2);
    output += m_tmp_t2;
  }
}

void bob::machine::IVectorMachine::forward_(const bob::machine::GMMStats& gs, 
  blitz::Array<double,1>& ivector) const
{
  // Computes \f$(Id + \sum_{c=1}^{C} N_{i,j,c} T^{T} \Sigma_{c}^{-1} T)\f$
  computeIdTtSigmaInvT(gs, m_tmp_tt);

  // Computes \f$T^{T} \Sigma^{-1} \sum_{c=1}^{C} (F_c - N_c ubmmean_{c})\f$
  computeTtSigmaInvFnorm(gs, m_tmp_t1);

  // Solves m_tmp_tt.ivector = m_tmp_t1
  bob::math::linsolve(m_tmp_tt, ivector, m_tmp_t1);
}

