/**
 * @file cxx/trainer/src/JFATrainer.cc
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Joint Factor Analysis Trainer
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "trainer/JFATrainer.h"
#include "math/inv.h"
#include "math/linear.h"
#include "core/array_check.h"
#include "core/Exception.h"
#include "core/repmat.h"
#include <algorithm>
#include <random/normal.h>


namespace train = bob::trainer;
namespace mach = bob::machine;
namespace math = bob::math;
namespace core = bob::core;

void train::jfa::updateEigen(const blitz::Array<double,3> &A, 
  const blitz::Array<double,2> &C, blitz::Array<double,2> &uv)
{
  // Check dimensions: ru
  int ru = A.extent(1);
  if(A.extent(2) != ru || C.extent(0) != ru || uv.extent(0) != ru)
    throw core::Exception();

  // Supervector length
  if(C.extent(1) != uv.extent(1))
    throw core::Exception();

  // Number of Gaussian components
  int Ng = A.extent(0); // Number of Gaussians
  if(C.extent(1) % Ng != 0)
    throw core::Exception();

  // Dimensionality, where cd = 
  //   c (Number of Gaussians) * d (Dimensionality of each Gaussian)
  int D = C.extent(1) / A.extent(0);
  
  // Intermediate array
  blitz::Array<double,2> tmp(ru, ru);
  // Initialize to 0
  uv = 0.;
  // Update U
  for(int c=0; c<Ng; ++c)
  {
    blitz::Array<double,2> c_elements = C(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,2> uv_elements = uv(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,2> A_c = A(c,blitz::Range::all(),blitz::Range::all());
    // Compute inverse of A
    math::inv(A_c, tmp);
    math::prod(tmp, c_elements, uv_elements);
  }
}

void train::jfa::estimateXandU(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E,
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v,
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z,
  const blitz::Array<double,2> &y, blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);

  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw core::Exception();
  int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw core::Exception();
  
  //index_map = reshape(repmat(1:size(N,2), dim,1),size(F,2),1);
  //x = zeros(size(spk_ids,1), size(u,1));

  // 2/ Initialize uEuT
  blitz::Array<double,3> uEuT(C, ru, ru);
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> tmp1(ru, D);
  for(int c=0; c<C; ++c)
  {
    blitz::Array<double,2> uEuT_c = uEuT(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> u_elements = u(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,1> e_elements = E(blitz::Range(c*D, (c+1)*D-1));

    tmp1 = u_elements(i,j) / e_elements(j);
    blitz::Array<double,2> u_transposed = u_elements.transpose(1,0);
    
    math::prod(tmp1, u_transposed, uEuT_c);
  }

  // 3/ Determine the vector of speaker ids
  //    Assume that samples from the same speaker are consecutive
  std::vector<uint32_t> ids;
  uint32_t last_elem=0;
  for(int ind=0; ind<T; ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // 4/ Main computation
  // Allocate working arrays
  blitz::Array<double,1> spk_shift(CD);
  blitz::Array<double,1> tmp2(CD);
  blitz::Array<double,1> tmp3(ru);
  blitz::Array<double,1> Fh(CD);
  blitz::Array<double,2> L(ru, ru);
  blitz::Array<double,2> Linv(ru, ru);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // a/ Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind = 0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<T; ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // b/ Compute speaker shift
    spk_shift = m;
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    math::prod(y_ii, v, tmp2);
    spk_shift += tmp2;
    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    spk_shift += z_ii * d;
   
    // c/ Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      blitz::Array<double,1> Nhint = N(jj, blitz::Range::all());
      core::repelem(Nhint, tmp2);  
      Fh = F(jj, blitz::Range::all()) - tmp2 * spk_shift;
      
      // L=Identity
      L = 0.;
      for(int k=0; k<ru; ++k)
        L(k,k) = 1.;
    
      for(int c=0; c<C; ++c)
      {
        blitz::Array<double,2> uEuT_c = uEuT(c, blitz::Range::all(), blitz::Range::all());
        L += uEuT_c * N(jj,c);
      }

      // inverse L
      math::inv(L, Linv);

      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      Fh /= E;
      blitz::Array<double,2> uu = u(blitz::Range::all(), blitz::Range::all());
      blitz::Array<double,2> u_t = uu.transpose(1,0);
      math::prod(Fh, u_t, tmp3);
      math::prod(tmp3, Linv, x_jj);
    }
  }
}


void train::jfa::estimateYandV(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N, 
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  blitz::Array<double,2> &y, const blitz::Array<double,2> &x, 
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);
  
  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw core::Exception();
  int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw core::Exception();


  blitz::Array<double,3> vEvT(C, rv, rv);
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> tmp1(rv,D);
  for(int c=0; c<C; ++c)
  {
    blitz::Array<double,2> vEvT_c = vEvT(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> v_elements = v(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,1> e_elements = E(blitz::Range(c*D, (c+1)*D-1));

    tmp1 = v_elements(i,j) / e_elements(j);
    blitz::Array<double,2> v_transposed = v_elements.transpose(1,0);
    math::prod(tmp1, v_transposed, vEvT_c);
  }

  // Determine a vector of speaker ids
  std::vector<uint32_t> ids;
  uint32_t last_elem=0;
  for(int ind=0; ind<spk_ids.extent(0); ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // Allocate working arrays
  blitz::Array<double,1> Fs(CD);
  blitz::Array<double,1> Nss(C);
  blitz::Array<double,1> Ns(CD);
  blitz::Array<double,1> tmp2(CD);
  blitz::Array<double,2> L(rv, rv);
  blitz::Array<double,2> Linv(rv, rv);
  blitz::Array<double,1> tmp3(rv);
  blitz::Array<double,1> tmp4(CD);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind=0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<spk_ids.extent(0); ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // Extract speaker sessions
    blitz::Array<double,2> Fs_sessions = F(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());
    blitz::Array<double,2> Nss_sessions = N(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());

    blitz::firstIndex i;
    blitz::secondIndex j;
    Fs = blitz::sum(Fs_sessions(j,i), j);
    Nss = blitz::sum(Nss_sessions(j,i), j);
    core::repelem(Nss, Ns);

    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    Fs -= ((m + z_ii * d) * Ns);

    // Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      math::prod(x_jj, u, tmp2);
      blitz::Array<double,1> N_jj = N(jj,blitz::Range::all());
      core::repelem(N_jj, tmp4);
      Fs -= tmp2 * tmp4;
    }

    // L=Identity
    L = 0.;
    for(int k=0; k<rv; ++k)
      L(k,k) = 1.;

    for(int c=0; c<C; ++c)
    {
      blitz::Array<double,2> vEvT_c = vEvT(c, blitz::Range::all(), blitz::Range::all());
      L += vEvT_c * Nss(c);
    }

    // inverse L
    math::inv(L, Linv);

    // update y
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    Fs /= E;
    blitz::Array<double,2> vv = v(blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> v_t = vv.transpose(1,0);
    math::prod(Fs, v_t, tmp3);
    math::prod(tmp3, Linv, y_ii);
  }    
}




void train::jfa::estimateZandD(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, const blitz::Array<double,2> &x, 
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);
  
  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw core::Exception();
  //int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw core::Exception();

  // Determine a vector of speaker ids
  std::vector<uint32_t> ids;
  uint32_t last_elem=0;
  for(int ind=0; ind<T; ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // Allocate arrays
  blitz::Array<double,1> shift(CD);
  blitz::Array<double,1> tmp1(CD);
  blitz::Array<double,1> Fs(CD);
  blitz::Array<double,1> Nss(C);
  blitz::Array<double,1> Ns(CD);
  blitz::Array<double,1> L(CD);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind=0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<T; ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // Extract speaker sessions
    blitz::Array<double,2> Fs_sessions = F(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());
    blitz::Array<double,2> Nss_sessions = N(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());

    blitz::firstIndex i;
    blitz::secondIndex j;
    Fs = blitz::sum(Fs_sessions(j,i), j);
    Nss = blitz::sum(Nss_sessions(j,i), j);
    core::repelem(Nss, Ns);

    // Compute shift
    shift = m;
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    math::prod(y_ii, v, tmp1);
    shift += tmp1;
    Fs -= shift * Ns;

    // Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      math::prod(x_jj, u, shift);
      blitz::Array<double,1> N_jj = N(jj,blitz::Range::all());
      core::repelem(N_jj, tmp1);
      Fs -= shift * tmp1;
    }

    L = 1.;
    L += Ns / E * blitz::pow2(d);

    // Update z   
    // z(ii,:) = Fs ./ E .* d ./L;
    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    z_ii = Fs / E * d / L;
  }
}



////////////////////////////////////////
////// JFATrainer class methods ////////
////////////////////////////////////////

train::JFABaseTrainerBase::JFABaseTrainerBase(mach::JFABaseMachine& m): 
  m_jfa_machine(m), 
  m_Nid(0), m_x(0), m_y(0), m_z(0), m_Nacc(0), m_Facc(0),
  m_cache_ubm_mean(0), m_cache_ubm_var(0)
{
  m_cache_ubm_mean.resize(m_jfa_machine.getDimCD());
  m_cache_ubm_var.resize(m_jfa_machine.getDimCD());
  m_jfa_machine.getUbm()->getMeanSupervector(m_cache_ubm_mean);
  m_jfa_machine.getUbm()->getVarianceSupervector(m_cache_ubm_var);
}

void train::JFABaseTrainerBase::initNid(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Number of people
  m_Nid = stats.size();
}

void train::JFABaseTrainerBase::initNid(const size_t Nid)
{
  // Number of people
  m_Nid = Nid;
}

void train::JFABaseTrainerBase::precomputeSumStatisticsN(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  m_Nacc.clear();
  blitz::Array<double,1> Nsum(m_jfa_machine.getDimC());
  for(size_t id=0; id<stats.size(); ++id) {
    Nsum = 0.;
    for(size_t s=0; s<stats[id].size(); ++s) {
      Nsum += stats[id][s]->n;
    }
    m_Nacc.push_back(core::array::ccopy(Nsum));
  }
}

void train::JFABaseTrainerBase::precomputeSumStatisticsF(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  m_Facc.clear();
  boost::shared_ptr<mach::GMMMachine> ubm(m_jfa_machine.getUbm());
  blitz::Array<double,1> Fsum(m_jfa_machine.getDimCD());
  for(size_t id=0; id<stats.size(); ++id) {
    Fsum = 0.;
    for(size_t s=0; s<stats[id].size(); ++s) {
      for(size_t g=0; g<ubm->getNGaussians(); ++g) {
        blitz::Array<double,1> Fsum_g = Fsum(blitz::Range(g*ubm->getNInputs(),(g+1)*ubm->getNInputs()-1));
        Fsum_g += stats[id][s]->sumPx(g,blitz::Range::all());
      }
    }
    m_Facc.push_back(core::array::ccopy(Fsum));
  }
}

void train::JFABaseTrainerBase::setSpeakerFactors(const std::vector<blitz::Array<double,2> >& x, 
  const std::vector<blitz::Array<double,1> >& y, 
  const std::vector<blitz::Array<double,1> >& z)
{
  // Number of people
  if(y.size() != m_Nid || z.size() != m_Nid)
    throw core::Exception();
  m_x.resize(x.size());
  m_y.resize(y.size());
  m_z.resize(z.size());


  for(size_t i=0; i<m_x.size(); ++i) 
    if(x[i].extent(0) != static_cast<int>(m_jfa_machine.getDimRu()))
      throw core::Exception();
    
  for(size_t i=0; i<m_Nid; ++i) {
    if(y[i].extent(0) != static_cast<int>(m_jfa_machine.getDimRv()))
      throw core::Exception();
    if(z[i].extent(0) != static_cast<int>(m_jfa_machine.getDimCD()))
      throw core::Exception();
  }

  // Copy the vectors
  for(size_t i=0; i<m_x.size(); ++i) 
    m_x[i].reference(core::array::ccopy(x[i]));
  for(size_t i=0; i<m_Nid; ++i) {
    m_y[i].reference(core::array::ccopy(y[i]));
    m_z[i].reference(core::array::ccopy(z[i]));
  }
}

void train::JFABaseTrainerBase::initializeRandom(blitz::Array<double,1>& vector)
{
  ranlib::Normal<double> normalGen(0., 1.);
  for(int i=0; i<vector.extent(0); ++i) 
    vector(i) = normalGen.random();    // normal random number 
}

void train::JFABaseTrainerBase::initializeRandom(blitz::Array<double,2>& matrix)
{
  ranlib::Normal<double> normalGen(0., 1.);
  for(int i=0; i<matrix.extent(0); ++i) {
    blitz::Array<double,1> vec = matrix(i, blitz::Range::all());
    initializeRandom(vec);
  }
}

void train::JFABaseTrainerBase::initializeRandomU()
{
  blitz::Array<double,2> U = m_jfa_machine.updateU();
  initializeRandom(U);
}

void train::JFABaseTrainerBase::initializeRandomV()
{
  blitz::Array<double,2> V = m_jfa_machine.updateV();
  initializeRandom(V);
}

void train::JFABaseTrainerBase::initializeRandomD()
{
  blitz::Array<double,1> d = m_jfa_machine.updateD();
  initializeRandom(d);
}

void train::JFABaseTrainerBase::initializeUVD()
{
  initializeRandomV();
  initializeRandomU();
  initializeRandomD();
}

void train::JFABaseTrainerBase::initializeXYZ(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& vec)
{
  std::vector<blitz::Array<double,1> > z;
  std::vector<blitz::Array<double,1> > y;
  std::vector<blitz::Array<double,2> > x;

  blitz::Array<double,1> z0(m_jfa_machine.getDimCD());
  z0 = 0;
  blitz::Array<double,1> y0(m_jfa_machine.getDimRv());
  y0 = 0;
  blitz::Array<double,2> x0(m_jfa_machine.getDimRu(),0);
  x0 = 0;
  for(size_t i=0; i<vec.size(); ++i)
  {
    z.push_back(core::array::ccopy(z0));
    y.push_back(core::array::ccopy(y0));
    x0.resize(m_jfa_machine.getDimRu(),vec[i].size());
    x0 = 0;
    x.push_back(core::array::ccopy(x0));
  }
  setSpeakerFactors(x,y,z);
}



train::JFABaseTrainer::JFABaseTrainer(mach::JFABaseMachine& m): 
  JFABaseTrainerBase(m),
  m_cache_VtSigmaInv(0), m_cache_VProd(0), m_cache_IdPlusVProd_i(0), 
  m_cache_Fn_y_i(0), m_cache_A1_y(0), m_cache_A2_y(0),
  m_cache_UtSigmaInv(0), m_cache_UProd(0), m_cache_IdPlusUProd_ih(0),
  m_cache_Fn_x_ih(0), m_cache_A1_x(0), m_cache_A2_x(0),
  m_cache_DtSigmaInv(0), m_cache_DProd(0), m_cache_IdPlusDProd_i(0),
  m_cache_Fn_z_i(0), m_cache_A1_z(0), m_cache_A2_z(0),
  m_tmp_rvrv(0), m_tmp_rvCD(0), m_tmp_rvD(0), m_tmp_ruD(0),
  m_tmp_ruru(0), m_tmp_rv(0), m_tmp_ru(0), m_tmp_CD(0), m_tmp_CD_b(0),
  m_tmp_D(0), m_tmp_CDCD(0)
{
  initCache();
}

void train::JFABaseTrainer::initCache()
{
  // U
  m_cache_UtSigmaInv.resize(m_jfa_machine.getDimRu(), m_jfa_machine.getDimCD());
  m_cache_UProd.resize(m_jfa_machine.getDimC(),m_jfa_machine.getDimRu(),m_jfa_machine.getDimRu());
  m_cache_IdPlusUProd_ih.resize(m_jfa_machine.getDimRu(),m_jfa_machine.getDimRu());
  m_cache_Fn_x_ih.resize(m_jfa_machine.getDimCD());
  m_cache_A1_x.resize(m_jfa_machine.getDimC(),m_jfa_machine.getDimRu(),m_jfa_machine.getDimRu());
  m_cache_A2_x.resize(m_jfa_machine.getDimCD(),m_jfa_machine.getDimRu());
  // V
  m_cache_VtSigmaInv.resize(m_jfa_machine.getDimRv(), m_jfa_machine.getDimCD());
  m_cache_VProd.resize(m_jfa_machine.getDimC(),m_jfa_machine.getDimRv(),m_jfa_machine.getDimRv());
  m_cache_IdPlusVProd_i.resize(m_jfa_machine.getDimRv(),m_jfa_machine.getDimRv());
  m_cache_Fn_y_i.resize(m_jfa_machine.getDimCD());
  m_cache_A1_y.resize(m_jfa_machine.getDimC(),m_jfa_machine.getDimRv(),m_jfa_machine.getDimRv());
  m_cache_A2_y.resize(m_jfa_machine.getDimCD(),m_jfa_machine.getDimRv());
  // D
  m_cache_DtSigmaInv.resize(m_jfa_machine.getDimCD());
  m_cache_DProd.resize(m_jfa_machine.getDimCD());
  m_cache_IdPlusDProd_i.resize(m_jfa_machine.getDimCD());
  m_cache_Fn_z_i.resize(m_jfa_machine.getDimCD());
  m_cache_A1_z.resize(m_jfa_machine.getDimCD());
  m_cache_A2_z.resize(m_jfa_machine.getDimCD());

  // tmp
  m_tmp_CD.resize(m_jfa_machine.getDimCD());
  m_tmp_CD_b.resize(m_jfa_machine.getDimCD());

  m_tmp_ru.resize(m_jfa_machine.getDimRu());
  m_tmp_ruD.resize(m_jfa_machine.getDimRu(),m_jfa_machine.getDimD());
  m_tmp_ruru.resize(m_jfa_machine.getDimRu(),m_jfa_machine.getDimRu());

  m_tmp_rv.resize(m_jfa_machine.getDimRv());
  m_tmp_rvD.resize(m_jfa_machine.getDimRv(),m_jfa_machine.getDimD());
  m_tmp_rvrv.resize(m_jfa_machine.getDimRv(), m_jfa_machine.getDimRv());
}

void train::JFABaseTrainer::computeVtSigmaInv()
{
  const blitz::Array<double,2>& V = m_jfa_machine.getV();
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not 
  // provide a non-const version of transpose())
  const blitz::Array<double,2> Vt = const_cast<blitz::Array<double,2>&>(V).transpose(1,0);
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_VtSigmaInv = Vt(i,j) / sigma(j); // Vt * diag(sigma)^-1
}

void train::JFABaseTrainer::computeVProd() 
{
  blitz::firstIndex i;
  blitz::secondIndex j;
  const blitz::Array<double,2>& V = m_jfa_machine.getV();
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c)
  {
    blitz::Array<double,2> VProd_c = m_cache_VProd(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> Vv_c = V(blitz::Range(c*m_jfa_machine.getDimD(),(c+1)*m_jfa_machine.getDimD()-1), blitz::Range::all());
    blitz::Array<double,2> Vt_c = Vv_c.transpose(1,0);
    blitz::Array<double,1> sigma_c = sigma(blitz::Range(c*m_jfa_machine.getDimD(),(c+1)*m_jfa_machine.getDimD()-1));
    m_tmp_rvD = Vt_c(i,j) / sigma_c(j); // Vt_c * diag(sigma)^-1 
    math::prod(m_tmp_rvD, Vv_c, VProd_c);
  }
}

void train::JFABaseTrainer::computeIdPlusVProd_i(const size_t id) 
{
  const blitz::Array<double,1>& Ni = m_Nacc[id];
  math::eye(m_tmp_rvrv); // m_tmp_rvrv = I
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c) {
    blitz::Array<double,2> VProd_c = m_cache_VProd(c,blitz::Range::all(),blitz::Range::all());
    m_tmp_rvrv += VProd_c * Ni(c);
  }
  math::inv(m_tmp_rvrv, m_cache_IdPlusVProd_i); // m_cache_IdPlusVProd_i = ( I+Vt*diag(sigma)^-1*Ni*V)^-1
}

void train::JFABaseTrainer::computeFn_y_i(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats, const size_t id)
{
  // Compute Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) (Normalised first order statistics)
  const blitz::Array<double,1>& Fi = m_Facc[id];
  const blitz::Array<double,1>& m = m_cache_ubm_mean;
  const blitz::Array<double,1>& d = m_jfa_machine.getD();
  const blitz::Array<double,1>& z = m_z[id];
  core::repelem(m_Nacc[id], m_tmp_CD);
  m_cache_Fn_y_i = Fi - m_tmp_CD * (m + d * z); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i}) 
  const blitz::Array<double,2>& X = m_x[id];
  const blitz::Array<double,2>& U = m_jfa_machine.getU();
  for(int h=0; h<X.extent(1); ++h) // Loops over the sessions
  {
    blitz::Array<double,1> Xh = X(blitz::Range::all(), h); // Xh = x_{i,h} (length: ru)
    math::prod(U, Xh, m_tmp_CD_b); // m_tmp_CD_b = U*x_{i,h}
    const blitz::Array<double,1>& Nih = stats[id][h]->n;
    core::repelem(Nih, m_tmp_CD);
    m_cache_Fn_y_i -= m_tmp_CD * m_tmp_CD_b; // N_{i,h} * U * x_{i,h}
  }
  // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
}

void train::JFABaseTrainer::updateY_i(const size_t id)
{
  // Computes yi = Ayi * Cvs * Fn_yi
  blitz::Array<double,1>& y = m_y[id];
  // m_tmp_rv = m_cache_VtSigmaInv * m_cache_Fn_y_i = Vt*diag(sigma)^-1 * sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
  math::prod(m_cache_VtSigmaInv, m_cache_Fn_y_i, m_tmp_rv); 
  math::prod(m_cache_IdPlusVProd_i, m_tmp_rv, y);
}

void train::JFABaseTrainer::updateY(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Precomputation
  computeVtSigmaInv();
  computeVProd();
  // Loops over all people
  for(size_t id=0; id<m_Nacc.size(); ++id) {
    computeIdPlusVProd_i(id);
    computeFn_y_i(stats, id);
    updateY_i(id);
  }
}

void train::JFABaseTrainer::updateV(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{  
  // Initializes the cache accumulator
  m_cache_A1_y = 0.;
  m_cache_A2_y = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j;
  for(size_t id=0; id<m_Nacc.size(); ++id) {
    computeIdPlusVProd_i(id);
    computeFn_y_i(stats, id);

    // Needs to return values to be accumulated for estimating V
    const blitz::Array<double,1>& y = m_y[id];
    m_tmp_rvrv = m_cache_IdPlusVProd_i;
    m_tmp_rvrv += y(i) * y(j); 
    for(size_t c=0; c<m_jfa_machine.getDimC(); ++c)
    {
      blitz::Array<double,2> A1_y_c = m_cache_A1_y(c,blitz::Range::all(),blitz::Range::all());
      A1_y_c += m_tmp_rvrv * m_Nacc[id](c);
    }
    m_cache_A2_y += m_cache_Fn_y_i(i) * y(j);
  }
 
  const size_t dim = m_jfa_machine.getDimD();
  blitz::Array<double,2>& V = m_jfa_machine.updateV();
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c)
  {
    const blitz::Array<double,2> A1 = m_cache_A1_y(c,blitz::Range::all(),blitz::Range::all());
    math::inv(A1, m_tmp_rvrv);
    const blitz::Array<double,2> A2 = m_cache_A2_y(blitz::Range(c*dim,(c+1)*dim-1), blitz::Range::all());
    blitz::Array<double,2> V_c = V(blitz::Range(c*dim,(c+1)*dim-1), blitz::Range::all());
    math::prod(A2, m_tmp_rvrv, V_c);
  }
}


void train::JFABaseTrainer::computeUtSigmaInv()
{
  const blitz::Array<double,2>& U = m_jfa_machine.getU();
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not 
  // provide a non-const version of transpose())
  const blitz::Array<double,2> Ut = const_cast<blitz::Array<double,2>&>(U).transpose(1,0);
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_UtSigmaInv = Ut(i,j) / sigma(j); // Ut * diag(sigma)^-1
}

void train::JFABaseTrainer::computeUProd() 
{
  blitz::firstIndex i;
  blitz::secondIndex j;
  const blitz::Array<double,2>& U = m_jfa_machine.getU();
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c)
  {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> Uu_c = U(blitz::Range(c*m_jfa_machine.getDimD(),(c+1)*m_jfa_machine.getDimD()-1), blitz::Range::all());
    blitz::Array<double,2> Ut_c = Uu_c.transpose(1,0);
    blitz::Array<double,1> sigma_c = sigma(blitz::Range(c*m_jfa_machine.getDimD(),(c+1)*m_jfa_machine.getDimD()-1));
    m_tmp_ruD = Ut_c(i,j) / sigma_c(j); // Ut_c * diag(sigma)^-1 
    math::prod(m_tmp_ruD, Uu_c, UProd_c);
  }
}

void train::JFABaseTrainer::computeIdPlusUProd_ih(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats, const size_t id, const size_t h) 
{
  const blitz::Array<double,1>& Nih = stats[id][h]->n;
  math::eye(m_tmp_ruru); // m_tmp_ruru = I
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c) {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c,blitz::Range::all(),blitz::Range::all());
    m_tmp_ruru += UProd_c * Nih(c);
  }
  math::inv(m_tmp_ruru, m_cache_IdPlusUProd_ih); // m_cache_IdPlusUProd_ih = ( I+Ut*diag(sigma)^-1*Ni*U)^-1
}

void train::JFABaseTrainer::computeFn_x_ih(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats, const size_t id, const size_t h)
{
  // Compute Fn_x_ih = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i}) (Normalised first order statistics)
  const blitz::Array<double,2>& Fih = stats[id][h]->sumPx;
  const blitz::Array<double,1>& m = m_cache_ubm_mean;
  const blitz::Array<double,1>& d = m_jfa_machine.getD();
  const blitz::Array<double,1>& z = m_z[id];
  const blitz::Array<double,1>& Nih = stats[id][h]->n;
  core::repelem(Nih, m_tmp_CD); 
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c) {
    blitz::Array<double,1> Fn_x_ih_c = m_cache_Fn_x_ih(blitz::Range(c*m_jfa_machine.getDimD(),(c+1)*m_jfa_machine.getDimD()-1));
    Fn_x_ih_c = Fih(c,blitz::Range::all());
  } 
  m_cache_Fn_x_ih -= m_tmp_CD * (m + d * z); // Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i}) 

  const blitz::Array<double,1>& y = m_y[id];
  const blitz::Array<double,2>& V = m_jfa_machine.getV();
  math::prod(V, y, m_tmp_CD_b);
  m_cache_Fn_x_ih -= m_tmp_CD * m_tmp_CD_b;
  // Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
}

void train::JFABaseTrainer::updateX_ih(const size_t id, const size_t h)
{
  // Computes xih = Axih * Cus * Fn_x_ih
  blitz::Array<double,1> x = m_x[id](blitz::Range::all(), h);
  // m_tmp_ru = m_cache_UtSigmaInv * m_cache_Fn_x_ih = Ut*diag(sigma)^-1 * N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
  math::prod(m_cache_UtSigmaInv, m_cache_Fn_x_ih, m_tmp_ru); 
  math::prod(m_cache_IdPlusUProd_ih, m_tmp_ru, x);
}

void train::JFABaseTrainer::updateX(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Precomputation
  computeUtSigmaInv();
  computeUProd();
  // Loops over all people
  for(size_t id=0; id<stats.size(); ++id) {
    int n_session_i = m_x[id].extent(1);
    for(int s=0; s<n_session_i; ++s) {
      computeIdPlusUProd_ih(stats, id, s);
      computeFn_x_ih(stats, id, s);
      updateX_ih(id, s);
    }
  }
}

void train::JFABaseTrainer::updateU(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Initializes the cache accumulator
  m_cache_A1_x = 0.;
  m_cache_A2_x = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j; 
  for(size_t id=0; id<stats.size(); ++id) {
    int n_session_i = m_x[id].extent(1);
    for(int h=0; h<n_session_i; ++h) {
      computeIdPlusUProd_ih(stats, id, h);
      computeFn_x_ih(stats,id, h);

      // Needs to return values to be accumulated for estimating U
      blitz::Array<double,1> x = m_x[id](blitz::Range::all(), h);
      m_tmp_ruru = m_cache_IdPlusUProd_ih;
      m_tmp_ruru += x(i) * x(j); 
      for(int c=0; c<static_cast<int>(m_jfa_machine.getDimC()); ++c)
      {
        blitz::Array<double,2> A1_x_c = m_cache_A1_x(c,blitz::Range::all(),blitz::Range::all());
        A1_x_c += m_tmp_ruru * stats[id][h]->n(c);
      }
      m_cache_A2_x += m_cache_Fn_x_ih(i) * x(j);
    }
  }

  const size_t dim = m_jfa_machine.getDimD();
  for(size_t c=0; c<m_jfa_machine.getDimC(); ++c)
  {
    const blitz::Array<double,2> A1 = m_cache_A1_x(c,blitz::Range::all(),blitz::Range::all());
    math::inv(A1, m_tmp_ruru);
    const blitz::Array<double,2> A2 = m_cache_A2_x(blitz::Range(c*dim,(c+1)*dim-1),blitz::Range::all());
    blitz::Array<double,2>& U = m_jfa_machine.updateU();
    blitz::Array<double,2> U_c = U(blitz::Range(c*dim,(c+1)*dim-1),blitz::Range::all());
    math::prod(A2, m_tmp_ruru, U_c);
  }
}

void train::JFABaseTrainer::computeDtSigmaInv()
{
  const blitz::Array<double,1>& d = m_jfa_machine.getD();
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  m_cache_DtSigmaInv = d / sigma; // Dt * diag(sigma)^-1
}

void train::JFABaseTrainer::computeDProd() 
{
  const blitz::Array<double,1>& d = m_jfa_machine.getD();
  const blitz::Array<double,1>& sigma = m_cache_ubm_var;
  m_cache_DProd = d / sigma * d; // Dt * diag(sigma)^-1 * D
}

void train::JFABaseTrainer::computeIdPlusDProd_i(const size_t id)
{
  const blitz::Array<double,1>& Ni = m_Nacc[id];
  core::repelem(Ni, m_tmp_CD); // m_tmp_CD = Ni 'repmat'
  m_cache_IdPlusDProd_i = 1.; // m_cache_IdPlusDProd_i = Id
  m_cache_IdPlusDProd_i += m_cache_DProd * m_tmp_CD; // m_cache_IdPlusDProd_i = I+Dt*diag(sigma)^-1*Ni*D
  m_cache_IdPlusDProd_i = 1 / m_cache_IdPlusDProd_i; // m_cache_IdPlusVProd_i = (I+Dt*diag(sigma)^-1*Ni*D)^-1
}

void train::JFABaseTrainer::computeFn_z_i(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats, const size_t id)
{
  // Compute Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h}) (Normalised first order statistics)
  const blitz::Array<double,1>& Fi = m_Facc[id];
  const blitz::Array<double,1>& m = m_cache_ubm_mean;
  const blitz::Array<double,2>& V = m_jfa_machine.getV();
  const blitz::Array<double,1>& y = m_y[id];
  core::repelem(m_Nacc[id], m_tmp_CD);
  math::prod(V, y, m_tmp_CD_b); // m_tmp_CD_b = V * y
  m_cache_Fn_z_i = Fi - m_tmp_CD * (m + m_tmp_CD_b); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i}) 

  const blitz::Array<double,2>& X = m_x[id];
  const blitz::Array<double,2>& U = m_jfa_machine.getU();
  for(int h=0; h<X.extent(1); ++h) // Loops over the sessions
  {
    const blitz::Array<double,1>& Nh = stats[id][h]->n; // Nh = N_{i,h} (length: C)
    core::repelem(Nh, m_tmp_CD);
    blitz::Array<double,1> Xh = X(blitz::Range::all(), h); // Xh = x_{i,h} (length: ru)
    math::prod(U, Xh, m_tmp_CD_b);
    m_cache_Fn_z_i -= m_tmp_CD * m_tmp_CD_b;
  }
  // Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
}

void train::JFABaseTrainer::updateZ_i(const size_t id)
{
  // Computes zi = Azi * D^T.Sigma^-1 * Fn_zi
  blitz::Array<double,1>& z = m_z[id];
  // m_tmp_CD = m_cache_DtSigmaInv * m_cache_Fn_z_i = Dt*diag(sigma)^-1 * sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
  z = m_cache_IdPlusDProd_i * m_cache_DtSigmaInv * m_cache_Fn_z_i; 
}

void train::JFABaseTrainer::updateZ(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Precomputation
  computeDtSigmaInv();
  computeDProd();
  // Loops over all people
  for(size_t id=0; id<m_Nacc.size(); ++id) {
    computeIdPlusDProd_i(id);
    computeFn_z_i(stats, id);
    updateZ_i(id);
  }
}

void train::JFABaseTrainer::updateD(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& stats)
{
  // Initializes the cache accumulator
  m_cache_A1_z = 0.;
  m_cache_A2_z = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j; 
  for(size_t id=0; id<m_Nacc.size(); ++id) {
    computeIdPlusDProd_i(id);
    computeFn_z_i(stats, id);

    // Needs to return values to be accumulated for estimating D
    blitz::Array<double,1> z = m_z[id];
    core::repelem(m_Nacc[id], m_tmp_CD);
    m_cache_A1_z += (m_cache_IdPlusDProd_i + z * z) * m_tmp_CD;
    m_cache_A2_z += m_cache_Fn_z_i * z;
  } 

  blitz::Array<double,1>& d = m_jfa_machine.updateD();
  d = m_cache_A2_z / m_cache_A1_z;
}


void train::JFABaseTrainer::train(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& vec,
  const size_t n_iter)
{
  initNid(vec);
  precomputeSumStatisticsN(vec);
  precomputeSumStatisticsF(vec);

  initializeUVD();
  initializeXYZ(vec);

  for(size_t i=0; i<n_iter; ++i) {
    updateY(vec);
    updateV(vec);
  }
  updateY(vec);
  for(size_t i=0; i<n_iter; ++i) {
    updateX(vec);
    updateU(vec);
  }
  updateX(vec);
  for(size_t i=0; i<n_iter; ++i) {
    updateZ(vec);
    updateD(vec);
  }
}

void train::JFABaseTrainer::trainNoInit(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& vec,
  const size_t n_iter)
{
  initNid(vec);
  precomputeSumStatisticsN(vec);
  precomputeSumStatisticsF(vec);

  initializeXYZ(vec);

  for(size_t i=0; i<n_iter; ++i) {
    updateY(vec);
    updateV(vec);
  }
  updateY(vec);
  for(size_t i=0; i<n_iter; ++i) {
    updateX(vec);
    updateU(vec);
  }
  updateX(vec);
  for(size_t i=0; i<n_iter; ++i) {
    updateZ(vec);
    updateD(vec);
  }
}


void train::JFABaseTrainer::initializeVD_ISV(const double relevance_factor)
{
  // V = 0
  blitz::Array<double,2> V = m_jfa_machine.updateV();
  V = 0.;
  // D = sqrt(variance(UBM) / relevance_factor)
  blitz::Array<double,1> d = m_jfa_machine.updateD();
  m_jfa_machine.getUbm()->getVarianceSupervector(m_cache_ubm_var);
  d = sqrt(m_cache_ubm_var / relevance_factor);
}

void train::JFABaseTrainer::trainISV(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& vec,
  const size_t n_iter, const double relevance_factor)
{
  initNid(vec);
  precomputeSumStatisticsN(vec);
  precomputeSumStatisticsF(vec);

  initializeRandomU();
  initializeVD_ISV(relevance_factor);

  for(size_t i=0; i<n_iter; ++i) {
    initializeXYZ(vec);
    updateX(vec);
    updateZ(vec);
    updateU(vec);
  }
}

void train::JFABaseTrainer::trainISVNoInit(const std::vector<std::vector<boost::shared_ptr<const mach::GMMStats> > >& vec,
  const size_t n_iter, const double relevance_factor)
{
  initNid(vec);
  precomputeSumStatisticsN(vec);
  precomputeSumStatisticsF(vec);

  initializeVD_ISV(relevance_factor);

  for(size_t i=0; i<n_iter; ++i) {
    initializeXYZ(vec);
    updateX(vec);
    updateZ(vec);
    updateU(vec);
  }
}



train::JFATrainer::JFATrainer(mach::JFAMachine& jfa_machine, train::JFABaseTrainer& base_trainer): 
  m_jfa_machine(jfa_machine),
  m_base_trainer(base_trainer)
{
}

void train::JFATrainer::enrol(const std::vector<boost::shared_ptr<const mach::GMMStats> >& vec,
  const size_t n_iter)
{
  std::vector< std::vector<boost::shared_ptr<const mach::GMMStats> > > vvec;
  vvec.push_back(vec);
  m_base_trainer.initNid(1);
  m_base_trainer.precomputeSumStatisticsN(vvec);
  m_base_trainer.precomputeSumStatisticsF(vvec);

  m_base_trainer.initializeXYZ(vvec);
  
  for(size_t i=0; i<n_iter; ++i) {
    m_base_trainer.updateY(vvec);
    m_base_trainer.updateX(vvec);
    m_base_trainer.updateZ(vvec);
  }

  const blitz::Array<double,1> y(m_base_trainer.getY()[0]);
  const blitz::Array<double,1> z(m_base_trainer.getZ()[0]);
  m_jfa_machine.setY(y);
  m_jfa_machine.setZ(z);
}
