/**
 * @file machine/cxx/JFAMachine.cc
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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


#include "bob/machine/JFAMachine.h"
#include "bob/core/array_copy.h"
#include "bob/core/repmat.h"
#include "bob/math/linear.h"
#include "bob/math/inv.h"
#include "bob/machine/Exception.h"
#include "bob/machine/LinearScoring.h"
#include <cmath>

namespace mach = bob::machine;
namespace core = bob::core;
namespace ca = bob::core::array;
namespace io = bob::io;
namespace math = bob::math;


mach::JFABaseMachine::JFABaseMachine():
  m_ubm(boost::shared_ptr<mach::GMMMachine>()), m_ru(0), m_rv(0),
  m_U(0,0), m_V(0,0), m_d(0)
{
}

mach::JFABaseMachine::JFABaseMachine(const boost::shared_ptr<bob::machine::GMMMachine> ubm, 
    const size_t ru, const size_t rv):
  m_ubm(ubm), m_ru(ru), m_rv(rv),
  m_U(getDimCD(),ru), m_V(getDimCD(),rv), m_d(getDimCD())
{
}


mach::JFABaseMachine::JFABaseMachine(const mach::JFABaseMachine& other):
  m_ubm(other.m_ubm), m_ru(other.m_ru), m_rv(other.m_rv),
  m_U(ca::ccopy(other.m_U)), m_V(ca::ccopy(other.m_V)), m_d(ca::ccopy(other.m_d))
{
}

mach::JFABaseMachine::JFABaseMachine(io::HDF5File& config):
  m_ubm(boost::shared_ptr<mach::GMMMachine>()) 
{
  load(config);
}

mach::JFABaseMachine::~JFABaseMachine() {
}

mach::JFABaseMachine& mach::JFABaseMachine::operator=
(const mach::JFABaseMachine& other) {
  m_ubm = other.m_ubm;
  m_ru = other.m_ru;
  m_rv = other.m_rv;
  m_U.reference(ca::ccopy(other.m_U));
  m_V.reference(ca::ccopy(other.m_V));
  m_d.reference(ca::ccopy(other.m_d));
  return *this;
}

bool mach::JFABaseMachine::operator==(const mach::JFABaseMachine& b) const {
  return (((m_ubm && b.m_ubm) && *m_ubm == *(b.m_ubm)) || (!m_ubm && !b.m_ubm)) && 
         m_ru == b.m_ru && m_rv == b.m_rv &&
         blitz::all(m_U == b.m_U) && blitz::all(m_V == b.m_V) &&
         blitz::all(m_d == b.m_d);
}

void mach::JFABaseMachine::load(io::HDF5File& config) {
  //reads all data directly into the member variables
  m_U.reference(config.readArray<double,2>("U"));
  m_V.reference(config.readArray<double,2>("V"));
  m_d.reference(config.readArray<double,1>("d"));
  m_ru = static_cast<size_t>(m_U.extent(1));
  m_rv = static_cast<size_t>(m_V.extent(1));
}

void mach::JFABaseMachine::save(io::HDF5File& config) const {
  config.setArray("U", m_U);
  config.setArray("V", m_V);
  config.setArray("d", m_d);
}

void mach::JFABaseMachine::resize(const size_t ru, const size_t rv) {
  m_ru = ru;
  m_rv = rv;
  m_U.resizeAndPreserve(m_U.extent(0), ru);
  m_V.resizeAndPreserve(m_V.extent(0), rv);
}

void mach::JFABaseMachine::setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm) {
  m_ubm = ubm;
  m_U.resizeAndPreserve(getDimCD(), m_ru);
  m_V.resizeAndPreserve(getDimCD(), m_rv);
  m_d.resizeAndPreserve(getDimCD());
}

void mach::JFABaseMachine::setU(const blitz::Array<double,2>& U) {
  if(U.extent(0) != m_U.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(0), m_U.extent(0));
  }
  if(U.extent(1) != m_U.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(U.extent(1), m_U.extent(1));
  }
  m_U.reference(ca::ccopy(U));
}

void mach::JFABaseMachine::setV(const blitz::Array<double,2>& V) {
  if(V.extent(0) != m_V.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(0), m_V.extent(0));
  }
  if(V.extent(1) != m_V.extent(1)) { //checks dimension
    throw mach::NInputsMismatch(V.extent(1), m_V.extent(1));
  }
  m_V.reference(ca::ccopy(V));
}

void mach::JFABaseMachine::setD(const blitz::Array<double,1>& d) {
  if(d.extent(0) != m_d.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(d.extent(0), m_d.extent(0));
  }
  m_d.reference(ca::ccopy(d));
}



mach::JFAMachine::JFAMachine():
  m_jfa_base(boost::shared_ptr<mach::JFABaseMachine>()),
  m_y(0), m_z(0),
  m_y_for_x(0), m_z_for_x(0),
  m_x(0)
{
}

mach::JFAMachine::JFAMachine(const boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base): 
  m_jfa_base(jfa_base),
  m_y(jfa_base->getDimRv()), m_z(jfa_base->getDimCD()),
  m_y_for_x(jfa_base->getDimRv()), m_z_for_x(jfa_base->getDimCD()),
  m_x(jfa_base->getDimRu())
{
  resizeCache();
}


mach::JFAMachine::JFAMachine(const mach::JFAMachine& other):
  m_jfa_base(other.m_jfa_base),
  m_y(ca::ccopy(other.m_y)), m_z(ca::ccopy(other.m_z)),
  m_y_for_x(ca::ccopy(other.m_y_for_x)), m_z_for_x(ca::ccopy(other.m_z_for_x)),
  m_x(ca::ccopy(other.m_x))
{
  resizeCache();
}

mach::JFAMachine::JFAMachine(io::HDF5File& config):
  m_jfa_base(boost::shared_ptr<mach::JFABaseMachine>())
{
  load(config);
}

mach::JFAMachine::~JFAMachine() {
}

mach::JFAMachine& mach::JFAMachine::operator=
(const mach::JFAMachine& other) {
  m_jfa_base = other.m_jfa_base;
  m_y.reference(ca::ccopy(other.m_y));
  m_z.reference(ca::ccopy(other.m_z));
  m_y_for_x.reference(ca::ccopy(other.m_y_for_x));
  m_z_for_x.reference(ca::ccopy(other.m_z_for_x));
  m_x.reference(ca::ccopy(other.m_x));
  return *this;
}

bool mach::JFAMachine::operator==(const mach::JFAMachine& m) const {
  return (((m_jfa_base && m.m_jfa_base) && *m_jfa_base == *(m.m_jfa_base)) ||
         (!m_jfa_base && !m.m_jfa_base)) &&
         blitz::all(m_y == m.m_y) && blitz::all(m_z == m.m_z);
}

void mach::JFAMachine::load(io::HDF5File& config) {
  //reads all data directly into the member variables
  m_y.reference(config.readArray<double,1>("y"));
  m_z.reference(config.readArray<double,1>("z"));
}

void mach::JFAMachine::save(io::HDF5File& config) const {
  config.setArray("y", m_y);
  config.setArray("z", m_z);
}

void mach::JFAMachine::setJFABase(const boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base) {
  m_jfa_base = jfa_base; 
  // Resize variables
  resize();
}

void mach::JFAMachine::resize()
{
  m_y.resizeAndPreserve(getDimRv());
  m_z.resizeAndPreserve(getDimCD());
  m_y_for_x.resize(getDimRv());
  m_z_for_x.resize(getDimCD());
  m_x.resize(getDimRu());

  resizeCache();
}

void mach::JFAMachine::resizeCache()
{
  m_cache_mean.resize(getDimCD());
  m_cache_sigma.resize(getDimCD());
  m_cache_UtSigmaInv.resize(getDimRu(), getDimCD());
  m_cache_IdPlusUSProdInv.resize(getDimRu(),getDimRu());
  m_cache_Fn_x.resize(getDimCD());

  m_tmp_ru.resize(getDimRu());
  m_tmp_ruD.resize(getDimRu(), getDimD());
  m_tmp_ruCD.resize(getDimRu(), getDimCD());
  m_tmp_ruru.resize(getDimRu(), getDimRu());
}

void mach::JFAMachine::setY(const blitz::Array<double,1>& y) {
  if(y.extent(0) != m_y.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(y.extent(0), m_y.extent(0));
  }
  m_y.reference(ca::ccopy(y));
}

void mach::JFAMachine::setZ(const blitz::Array<double,1>& z) {
  if(z.extent(0) != m_z.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(z.extent(0), m_z.extent(0));
  }
  m_z.reference(ca::ccopy(z));
}


void mach::JFAMachine::cacheSupervectors()
{ 
  // Put supervectors in cache
  m_jfa_base->getUbm()->getMeanSupervector(m_cache_mean);
  m_jfa_base->getUbm()->getVarianceSupervector(m_cache_sigma);
}

void mach::JFAMachine::computeUtSigmaInv()
{
  const blitz::Array<double,2>& U = m_jfa_base->getU();
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_UtSigmaInv = U(j,i) / m_cache_sigma(j); // Ut * diag(sigma)^-1
}

void mach::JFAMachine::computeIdPlusUSProdInv(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats)
{
  // Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 = (Id + sum_{c=1..C} N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c})^-1
  const blitz::Array<double,2>& U = m_jfa_base->getU();
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not 
  // provide a non-const version of transpose())
  blitz::Array<double,2> Ut = const_cast<blitz::Array<double,2>&>(U).transpose(1,0);

  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range rall = blitz::Range::all();

  math::eye(m_tmp_ruru); // m_tmp_ruru = Id
  // Loop and add N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c} to m_tmp_ruru at each iteration
  for(size_t c=0; c<getDimC(); ++c) {
    blitz::Range rc(c*getDimD(),(c+1)*getDimD()-1);
    blitz::Array<double,2> Ut_c = Ut(rall,rc);
    blitz::Array<double,1> sigma_c = m_cache_sigma(rc);
    m_tmp_ruD = Ut_c(i,j) / sigma_c(j); // U_{c}^T.Sigma_{c}^-1
    blitz::Array<double,2> U_c = U(rc,rall);
    // Use m_cache_IdPlusUSProdInv as an intermediate array
    math::prod(m_tmp_ruD, U_c, m_cache_IdPlusUSProdInv); // U_{c}^T.Sigma_{c}^-1.U_{c}
    // Finally, add N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c} to m_tmp_ruru
    m_tmp_ruru += m_cache_IdPlusUSProdInv * gmm_stats->n(c);
  }
  // Computes the inverse
  math::inv(m_tmp_ruru, m_cache_IdPlusUSProdInv);
}

void mach::JFAMachine::computeFn_x(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats)
{
  // Compute Fn_x = sum_{sessions h}(N*(o - m) (Normalised first order statistics)
  m_jfa_base->getUbm()->getMeanSupervector(m_cache_mean);

  blitz::Range rall = blitz::Range::all();
  for(size_t c=0; c<getDimC(); ++c) {
    blitz::Range rc(c*getDimD(),(c+1)*getDimD()-1);
    blitz::Array<double,1> Fn_x_c = m_cache_Fn_x(rc);
    blitz::Array<double,1> mean_c = m_cache_mean(rc);
    Fn_x_c = gmm_stats->sumPx(c,rall) - mean_c*gmm_stats->n(c);
  }
}

void mach::JFAMachine::updateX_fromCache()
{
  // m_tmp_ru = m_cache_UtSigmaInv * m_cache_Fn_x = Ut*diag(sigma)^-1 * N*(o - m)
  math::prod(m_cache_UtSigmaInv, m_cache_Fn_x, m_tmp_ru); 
  // x = m_cache_IdPlusUSProdInv * m_cache_UtSigmaInv * m_cache_Fn_x 
  math::prod(m_cache_IdPlusUSProdInv, m_tmp_ru, m_x);
}

void mach::JFAMachine::estimateX(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats) 
{
  cacheSupervectors(); // Put supervector in cache
  computeUtSigmaInv(); // Computes U^T.Sigma^-1
  computeIdPlusUSProdInv(gmm_stats); // Computes first term
  computeFn_x(gmm_stats); // Computes last term
  updateX_fromCache(); // Estimates the value of x using the current cache
}

void mach::JFAMachine::forward(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats, double& score)
{
  // Checks that a Base machine has been set
  if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet();

  // Ux and GMMStats
  estimateX(gmm_stats);

  std::vector<boost::shared_ptr<const mach::GMMStats> > stats;
  stats.push_back(gmm_stats);
  m_cache_Ux.resize(getDimCD());
  math::prod(m_jfa_base->getU(), m_x, m_cache_Ux);
  std::vector<blitz::Array<double,1> > channelOffset;
  channelOffset.push_back(m_cache_Ux);

  // m + Vy + Dz
  m_cache_mVyDz.resize(getDimCD());
  math::prod(m_jfa_base->getV(), m_y, m_cache_mVyDz);
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

void mach::JFAMachine::forward(const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& samples, blitz::Array<double,1>& score)
{
  // Checks that a Base machine has been set
  if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet();

  std::vector<blitz::Array<double,1> > channelOffset;
  m_cache_Ux.resize(getDimCD());
  for(size_t i=0; i<samples.size(); ++i)
  {
    // Ux and GMMStats
    estimateX(samples[i]);
    math::prod(m_jfa_base->getU(), m_x, m_cache_Ux);
    channelOffset.push_back(ca::ccopy(m_cache_Ux));
  }

  // m + Vy + Dz
  m_cache_mVyDz.resize(getDimCD());
  math::prod(m_jfa_base->getV(), m_y, m_cache_mVyDz);
  m_cache_mVyDz += m_jfa_base->getD()*m_z + m_jfa_base->getUbm()->getMeanSupervector();
  std::vector<blitz::Array<double,1> > models;
  models.push_back(m_cache_mVyDz);

  // Linear scoring
  // TODO: try to avoid this 2D array allocation or put in cache
  blitz::Array<double,2> scores(1,samples.size());
  mach::linearScoring(models, 
    m_jfa_base->getUbm()->getMeanSupervector(), m_jfa_base->getUbm()->getVarianceSupervector(),
    samples, channelOffset, true, scores);
  score = scores(0,blitz::Range::all());
}
