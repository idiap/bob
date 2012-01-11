/**
 * @file cxx/machine/src/JFAMachine.cc
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Machines that implements the Joint Factor Analysis model
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

#include <cmath>

#include "core/array_copy.h"
#include "core/repmat.h"
#include "io/Arrayset.h"
#include "math/linear.h"
#include "math/lu_det.h"
#include "machine/Exception.h"
#include "machine/JFAMachine.h"
#include "machine/LinearScoring.h"


namespace mach = bob::machine;
namespace math = bob::math;

mach::JFABaseMachine::JFABaseMachine():
  m_ubm(boost::shared_ptr<bob::machine::GMMMachine>()), m_ru(0), m_rv(0),
  m_U(0,0),
  m_V(0,0),
  m_d(0)
{
}

mach::JFABaseMachine::JFABaseMachine(const boost::shared_ptr<bob::machine::GMMMachine> ubm, 
    int ru, int rv):
  m_ubm(ubm), m_ru(ru), m_rv(rv),
  m_U(getDimC()*getDimD(),ru),
  m_V(getDimC()*getDimD(),rv),
  m_d(getDimC()*getDimD())
{
}


mach::JFABaseMachine::JFABaseMachine(const mach::JFABaseMachine& other):
  m_ubm(other.m_ubm), m_ru(other.m_ru), m_rv(other.m_rv),
  m_U(bob::core::array::ccopy(other.m_U)),
  m_V(bob::core::array::ccopy(other.m_V)),
  m_d(bob::core::array::ccopy(other.m_d))
{
}

mach::JFABaseMachine::JFABaseMachine(bob::io::HDF5File& config) {
  load(config);
}

mach::JFABaseMachine::~JFABaseMachine() {
}

mach::JFABaseMachine& mach::JFABaseMachine::operator=
(const mach::JFABaseMachine& other) {
  m_ubm = other.m_ubm;
  m_ru = other.m_ru;
  m_rv = other.m_rv;
  m_U.reference(bob::core::array::ccopy(other.m_U));
  m_V.reference(bob::core::array::ccopy(other.m_V));
  m_d.reference(bob::core::array::ccopy(other.m_d));
  return *this;
}

void mach::JFABaseMachine::load(bob::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_U.reference(config.readArray<double,2>("U"));
  m_V.reference(config.readArray<double,2>("V"));
  m_d.reference(config.readArray<double,1>("d"));
  m_ru = m_U.extent(1);
  m_rv = m_V.extent(1);
}

void mach::JFABaseMachine::save(bob::io::HDF5File& config) const {
  config.setArray("U", m_U);
  config.setArray("V", m_V);
  config.setArray("d", m_d);
}

void mach::JFABaseMachine::setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm) {
  m_ubm = ubm;
}

void mach::JFABaseMachine::setU(const blitz::Array<double,2>& U) {
  if(U.extent(0) != m_U.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(0), m_U.extent(0));
  }
  if(U.extent(1) != m_U.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(1), m_U.extent(1));
  }
  m_U.reference(bob::core::array::ccopy(U));
}

void mach::JFABaseMachine::setV(const blitz::Array<double,2>& V) {
  if(V.extent(0) != m_V.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(0), m_V.extent(0));
  }
  if(V.extent(1) != m_V.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(1), m_V.extent(1));
  }
  m_V.reference(bob::core::array::ccopy(V));
}

void mach::JFABaseMachine::setD(const blitz::Array<double,1>& d) {
  if(d.extent(0) != m_d.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(d.extent(0), m_d.extent(0));
  }
  m_d.reference(bob::core::array::ccopy(d));
}



mach::JFAMachine::JFAMachine():
  m_jfa_base(boost::shared_ptr<bob::machine::JFABaseMachine>()),
  m_y(0),
  m_z(0),
  m_y_for_x(0),
  m_z_for_x(0),
  m_x(0)
{
}

mach::JFAMachine::JFAMachine(const boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base): 
  m_jfa_base(jfa_base),
  m_y(jfa_base->getDimRv()),
  m_z(jfa_base->getDimCD()),
  m_y_for_x(jfa_base->getDimRv()),
  m_z_for_x(jfa_base->getDimCD()),
  m_x(jfa_base->getDimRu())
{
}


mach::JFAMachine::JFAMachine(const mach::JFAMachine& other):
  m_jfa_base(other.m_jfa_base),
  m_y(bob::core::array::ccopy(other.m_y)),
  m_z(bob::core::array::ccopy(other.m_z)),
  m_y_for_x(bob::core::array::ccopy(other.m_y_for_x)),
  m_z_for_x(bob::core::array::ccopy(other.m_z_for_x)),
  m_x(bob::core::array::ccopy(other.m_x))
{
}

mach::JFAMachine::JFAMachine(bob::io::HDF5File& config) {
  load(config);
}

mach::JFAMachine::~JFAMachine() {
}

mach::JFAMachine& mach::JFAMachine::operator=
(const mach::JFAMachine& other) {
  m_jfa_base = other.m_jfa_base;
  m_y.reference(bob::core::array::ccopy(other.m_y));
  m_z.reference(bob::core::array::ccopy(other.m_z));
  m_y_for_x.reference(bob::core::array::ccopy(other.m_y_for_x));
  m_z_for_x.reference(bob::core::array::ccopy(other.m_z_for_x));
  m_x.reference(bob::core::array::ccopy(other.m_x));
  return *this;
}

void mach::JFAMachine::load(bob::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_y.reference(config.readArray<double,1>("y"));
  m_z.reference(config.readArray<double,1>("z"));
}

void mach::JFAMachine::save(bob::io::HDF5File& config) const {
  config.setArray("y", m_y);
  config.setArray("z", m_z);
}

void mach::JFAMachine::setJFABase(const boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base) {
  m_jfa_base = jfa_base; 
  m_y.resize(jfa_base->getDimRv());
  m_z.resize(jfa_base->getDimCD());
  m_y_for_x.resize(jfa_base->getDimRv());
  m_z_for_x.resize(jfa_base->getDimCD());
  m_x.resize(jfa_base->getDimRu());
}

void mach::JFAMachine::setY(const blitz::Array<double,1>& y) {
  if(y.extent(0) != m_y.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(y.extent(0), m_y.extent(0));
  }
  m_y.reference(bob::core::array::ccopy(y));
}

void mach::JFAMachine::setZ(const blitz::Array<double,1>& z) {
  if(z.extent(0) != m_z.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(z.extent(0), m_z.extent(0));
  }
  m_z.reference(bob::core::array::ccopy(z));
}

void mach::JFAMachine::updateX(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F)
{
  // initialize variables (y=0 and z=0)
  m_x.resize(getDimRu());
  m_y_for_x.resize(getDimRv());
  m_y_for_x = 0;
  m_z_for_x.resize(getDimCD());
  m_z_for_x = 0;
  // Precompute Ut*diag(sigma)^-1
  computeUtSigmaInv();
  computeUProd();
  computeIdPlusUProd(N,F);
  computeFn_x(N,F);
  updateX_fromCache();
}

void mach::JFAMachine::computeUtSigmaInv()
{
  m_cache_UtSigmaInv.resize(getDimRu(), getDimCD());
  const blitz::Array<double,2>& U = m_jfa_base->getU();
  blitz::Array<double,2> Uu = U(blitz::Range::all(), blitz::Range::all()); // Blitz compatibility
  blitz::Array<double,2> Ut = Uu.transpose(1,0);
  m_cache_sigma.resize(getDimCD());
  m_jfa_base->getUbm()->getVarianceSupervector(m_cache_sigma);
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_UtSigmaInv = Ut(i,j) / m_cache_sigma(j); // Ut * diag(sigma)^-1
}

void mach::JFAMachine::computeUProd()
{
  m_cache_UProd.resizeAndPreserve(getDimC(),getDimRu(),getDimRu());
  m_tmp_ruD.resize(getDimRu(),getDimD());
  m_cache_sigma.resizeAndPreserve(getDimCD());
  m_jfa_base->getUbm()->getVarianceSupervector(m_cache_sigma);
  blitz::firstIndex i;
  blitz::secondIndex j;
  for(int c=0; c<getDimC(); ++c)
  {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c, blitz::Range::all(), blitz::Range::all());
    const blitz::Array<double,2>& U = m_jfa_base->getU();
    blitz::Array<double,2> Uu_c = U(blitz::Range(c*getDimD(),(c+1)*getDimD()-1), blitz::Range::all());
    blitz::Array<double,2> Ut_c = Uu_c.transpose(1,0);
    blitz::Array<double,1> sigma_c = m_cache_sigma(blitz::Range(c*getDimD(),(c+1)*getDimD()-1));
    m_tmp_ruD = Ut_c(i,j) / sigma_c(j); // Ut_c * diag(sigma)^-1 
    bob::math::prod(m_tmp_ruD, Uu_c, UProd_c);
  }
}

void mach::JFAMachine::computeIdPlusUProd(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F) 
{
  m_cache_IdPlusUProd.resizeAndPreserve(getDimRu(),getDimRu());
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_tmp_ruru.resize(getDimRu(), getDimRu());
  bob::math::eye(m_tmp_ruru); // m_tmp_ruru = I
  for(int c=0; c<getDimC(); ++c) {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c,blitz::Range::all(),blitz::Range::all());
    m_tmp_ruru += UProd_c * N(c);
  }
  bob::math::inv(m_tmp_ruru, m_cache_IdPlusUProd); // m_cache_IdPlusUProdh = ( I+Ut*diag(sigma)^-1*N*U)^-1
}

void mach::JFAMachine::computeFn_x(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F)
{
  // Compute Fn_x = sum_{sessions h}(N*(o - m - D*z - V*y) (Normalised first order statistics)
  m_cache_Fn_x.resize(getDimCD());
  m_cache_mean.resize(getDimCD());
  m_jfa_base->getUbm()->getMeanSupervector(m_cache_mean);
  const blitz::Array<double,1>& d = m_jfa_base->getD();
  m_tmp_CD.resize(getDimCD());
  bob::core::repelem(N, m_tmp_CD);
  m_cache_Fn_x = F - m_tmp_CD * (m_cache_mean + d * m_z_for_x); // Fn_x = N*(o - m - D*z) 

  const blitz::Array<double,2>& V = m_jfa_base->getV();
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_tmp_CD_b.resize(getDimCD());
  bob::math::prod(V, m_y_for_x, m_tmp_CD_b);
  m_cache_Fn_x -= m_tmp_CD * m_tmp_CD_b; // Fn_x = N*(o - m - D*z - V*y)
}

void mach::JFAMachine::updateX_fromCache()
{
  // Compute x = Ax * Cus * Fn_x
  m_tmp_ru.resize(getDimRu());
  // m_tmp_ru = m_cache_UtSigmaInv * m_cache_Fn_x = Ut*diag(sigma)^-1 * N*(o - m - D*z - V*y)
  bob::math::prod(m_cache_UtSigmaInv, m_cache_Fn_x, m_tmp_ru); 
  bob::math::prod(m_cache_IdPlusUProd, m_tmp_ru, m_x);
}

void mach::JFAMachine::estimateX(const mach::GMMStats* gmm_stats) {
  boost::shared_ptr<bob::machine::GMMMachine> ubm(getJFABase()->getUbm());
  m_cache_gmmstats.resize(ubm->getNGaussians(),ubm->getNInputs()); 
  blitz::Array<double,1> N(ubm->getNGaussians());
  blitz::Array<double,1> F(ubm->getNGaussians()*ubm->getNInputs());
  // TODO: check type/dimensions?
  m_cache_gmmstats = *gmm_stats;
  N = m_cache_gmmstats.n;
  for(int g=0; g<ubm->getNGaussians(); ++g) {
    blitz::Array<double,1> F_g = F(blitz::Range(g*ubm->getNInputs(),(g+1)*ubm->getNInputs()-1));
    F_g = m_cache_gmmstats.sumPx(g,blitz::Range::all());
  }
  updateX(N, F);
}


void mach::JFAMachine::forward(const mach::GMMStats* gmm_stats, double& score)
{
  // Ux and GMMStats
  estimateX(gmm_stats);
  std::vector<const bob::machine::GMMStats*> stats;
  stats.push_back(&m_cache_gmmstats);
  m_cache_Ux.resize(getDimCD());
  bob::math::prod(m_jfa_base->getU(), m_x, m_cache_Ux);
  std::vector<blitz::Array<double,1> > channelOffset;
  channelOffset.push_back(m_cache_Ux);

  // m + Vy + Dz
  m_cache_mVyDz.resize(getDimCD());
  bob::math::prod(m_jfa_base->getV(), m_y, m_cache_mVyDz);
  m_cache_mVyDz += m_jfa_base->getD()*m_z + m_jfa_base->getUbm()->getMeanSupervector();
  std::vector<blitz::Array<double,1> > models;
  models.push_back(m_cache_mVyDz);

  // Linear scoring
  blitz::Array<double,2> scores(1,1);
  mach::linearScoring(models, 
    m_jfa_base->getUbm()->getMeanSupervector(), m_jfa_base->getUbm()->getVarianceSupervector(),
    stats, channelOffset, true, scores);
  score = scores(0,0);
}

void mach::JFAMachine::forward(const std::vector<const mach::GMMStats*>& samples, blitz::Array<double,1>& score)
{
  std::vector<blitz::Array<double,1> > channelOffset;
  m_cache_Ux.resize(getDimCD());
  for(size_t i=0; i<samples.size(); ++i)
  {
    // Ux and GMMStats
    estimateX(samples[i]);
    bob::math::prod(m_jfa_base->getU(), m_x, m_cache_Ux);
    channelOffset.push_back(bob::core::array::ccopy(m_cache_Ux));
  }

  // m + Vy + Dz
  m_cache_mVyDz.resize(getDimCD());
  bob::math::prod(m_jfa_base->getV(), m_y, m_cache_mVyDz);
  m_cache_mVyDz += m_jfa_base->getD()*m_z + m_jfa_base->getUbm()->getMeanSupervector();
  std::vector<blitz::Array<double,1> > models;
  models.push_back(m_cache_mVyDz);

  // Linear scoring
  blitz::Array<double,2> scores(samples.size(),1);
  mach::linearScoring(models, 
    m_jfa_base->getUbm()->getMeanSupervector(), m_jfa_base->getUbm()->getVarianceSupervector(),
    samples, channelOffset, true, scores);
  blitz::Array<double,1> scores_sl = scores(blitz::Range::all(), 0);
  score = scores_sl;  
}
