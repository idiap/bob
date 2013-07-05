/**
 * @file trainer/cxx/JFATrainer.cc
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Joint Factor Analysis Trainer
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

#include <bob/trainer/JFATrainer.h>
#include <bob/core/check.h>
#include <bob/core/array_copy.h>

#include <bob/math/inv.h>
#include <bob/math/linear.h>
#include <bob/core/check.h>
#include <bob/core/array_repmat.h>
#include <algorithm>

#include <random/normal.h>


bob::trainer::FABaseTrainer::FABaseTrainer():
  m_Nid(0), m_dim_C(0), m_dim_D(0), m_dim_ru(0), m_dim_rv(0),
  m_x(0), m_y(0), m_z(0), m_Nacc(0), m_Facc(0)
{
}

bob::trainer::FABaseTrainer::FABaseTrainer(const bob::trainer::FABaseTrainer& other)
{
}

bob::trainer::FABaseTrainer::~FABaseTrainer()
{
}

void bob::trainer::FABaseTrainer::checkStatistics(
  const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  for (size_t id=0; id<stats.size(); ++id) {
    for (size_t s=0; s<stats[id].size(); ++s) {
      if (stats[id][s]->sumPx.extent(0) != (int)m_dim_C) {
        boost::format m("GMMStats C dimension parameter = %d is different than the expected value of %d");
        m % stats[id][s]->sumPx.extent(0) % (int)m_dim_C;
        throw std::runtime_error(m.str());
      }
      if (stats[id][s]->sumPx.extent(1) != (int)m_dim_D) {
        boost::format m("GMMStats D dimension parameter = %d is different than the expected value of %d");
        m % stats[id][s]->sumPx.extent(1) % (int)m_dim_D;
        throw std::runtime_error(m.str());
      }
    }
  }
}


void bob::trainer::FABaseTrainer::initUbmNidSumStatistics(
  const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  m_Nid = stats.size();
  boost::shared_ptr<bob::machine::GMMMachine> ubm = m.getUbm();
  // Put UBM in cache
  m_dim_C = ubm->getNGaussians();
  m_dim_D = ubm->getNInputs();
  m_dim_ru = m.getDimRu();
  m_dim_rv = m.getDimRv();
  // Check statistics dimensionality
  checkStatistics(m, stats);
  // Precomputes the sum of the statistics for each client/identity
  precomputeSumStatisticsN(stats);
  precomputeSumStatisticsF(stats);
  // Cache and working arrays
  initCache();
}

void bob::trainer::FABaseTrainer::precomputeSumStatisticsN(
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  m_Nacc.clear();
  blitz::Array<double,1> Nsum(m_dim_C);
  for (size_t id=0; id<stats.size(); ++id) {
    Nsum = 0.;
    for (size_t s=0; s<stats[id].size(); ++s) {
      Nsum += stats[id][s]->n;
    }
    m_Nacc.push_back(bob::core::array::ccopy(Nsum));
  }
}

void bob::trainer::FABaseTrainer::precomputeSumStatisticsF(
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  m_Facc.clear();
  blitz::Array<double,1> Fsum(m_dim_C*m_dim_D);
  for (size_t id=0; id<stats.size(); ++id) {
    Fsum = 0.;
    for (size_t s=0; s<stats[id].size(); ++s) {
      for (size_t c=0; c<m_dim_C; ++c) {
        blitz::Array<double,1> Fsum_c = Fsum(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1));
        Fsum_c += stats[id][s]->sumPx(c,blitz::Range::all());
      }
    }
    m_Facc.push_back(bob::core::array::ccopy(Fsum));
  }
}

void bob::trainer::FABaseTrainer::initializeXYZ(const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& vec)
{
  m_x.clear();
  m_y.clear();
  m_z.clear();

  blitz::Array<double,1> z0(m_dim_C*m_dim_D);
  z0 = 0;
  blitz::Array<double,1> y0(m_dim_rv);
  y0 = 0;
  blitz::Array<double,2> x0(m_dim_ru,0);
  x0 = 0;
  for (size_t i=0; i<vec.size(); ++i)
  {
    m_z.push_back(bob::core::array::ccopy(z0));
    m_y.push_back(bob::core::array::ccopy(y0));
    x0.resize(m_dim_ru, vec[i].size());
    x0 = 0;
    m_x.push_back(bob::core::array::ccopy(x0));
  }
}

void bob::trainer::FABaseTrainer::resetXYZ()
{
  for (size_t i=0; i<m_x.size(); ++i)
  {
    m_x[i] = 0.;
    m_y[i] = 0.;
    m_z[i] = 0.;
  }
}

void bob::trainer::FABaseTrainer::initCache()
{
  const size_t dim_CD = m_dim_C*m_dim_D;
  // U
  m_cache_UtSigmaInv.resize(m_dim_ru, dim_CD);
  m_cache_UProd.resize(m_dim_C, m_dim_ru, m_dim_ru);
  m_cache_IdPlusUProd_ih.resize(m_dim_ru, m_dim_ru);
  m_cache_Fn_x_ih.resize(dim_CD);
  m_acc_U_A1.resize(m_dim_C, m_dim_ru, m_dim_ru);
  m_acc_U_A2.resize(dim_CD, m_dim_ru);
  // V
  m_cache_VtSigmaInv.resize(m_dim_rv, dim_CD);
  m_cache_VProd.resize(m_dim_C, m_dim_rv, m_dim_rv);
  m_cache_IdPlusVProd_i.resize(m_dim_rv, m_dim_rv);
  m_cache_Fn_y_i.resize(dim_CD);
  m_acc_V_A1.resize(m_dim_C, m_dim_rv, m_dim_rv);
  m_acc_V_A2.resize(dim_CD, m_dim_rv);
  // D
  m_cache_DtSigmaInv.resize(dim_CD);
  m_cache_DProd.resize(dim_CD);
  m_cache_IdPlusDProd_i.resize(dim_CD);
  m_cache_Fn_z_i.resize(dim_CD);
  m_acc_D_A1.resize(dim_CD);
  m_acc_D_A2.resize(dim_CD);

  // tmp
  m_tmp_CD.resize(dim_CD);
  m_tmp_CD_b.resize(dim_CD);

  m_tmp_ru.resize(m_dim_ru);
  m_tmp_ruD.resize(m_dim_ru, m_dim_D);
  m_tmp_ruru.resize(m_dim_ru, m_dim_ru);

  m_tmp_rv.resize(m_dim_rv);
  m_tmp_rvD.resize(m_dim_rv, m_dim_D);
  m_tmp_rvrv.resize(m_dim_rv, m_dim_rv);
}



//////////////////////////// V ///////////////////////////
void bob::trainer::FABaseTrainer::computeVtSigmaInv(const bob::machine::FABase& m)
{
  const blitz::Array<double,2>& V = m.getV();
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not
  // provide a non-const version of transpose())
  const blitz::Array<double,2> Vt = const_cast<blitz::Array<double,2>&>(V).transpose(1,0);
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_VtSigmaInv = Vt(i,j) / sigma(j); // Vt * diag(sigma)^-1
}

void bob::trainer::FABaseTrainer::computeVProd(const bob::machine::FABase& m)
{
  const blitz::Array<double,2>& V = m.getV();
  blitz::firstIndex i;
  blitz::secondIndex j;
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  blitz::Range rall = blitz::Range::all();
  for (size_t c=0; c<m_dim_C; ++c)
  {
    blitz::Array<double,2> VProd_c = m_cache_VProd(c, rall, rall);
    blitz::Array<double,2> Vv_c = V(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1), rall);
    blitz::Array<double,2> Vt_c = Vv_c.transpose(1,0);
    blitz::Array<double,1> sigma_c = sigma(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1));
    m_tmp_rvD = Vt_c(i,j) / sigma_c(j); // Vt_c * diag(sigma)^-1
    bob::math::prod(m_tmp_rvD, Vv_c, VProd_c);
  }
}

void bob::trainer::FABaseTrainer::computeIdPlusVProd_i(const size_t id)
{
  const blitz::Array<double,1>& Ni = m_Nacc[id];
  bob::math::eye(m_tmp_rvrv); // m_tmp_rvrv = I
  blitz::Range rall = blitz::Range::all();
  for (size_t c=0; c<m_dim_C; ++c) {
    blitz::Array<double,2> VProd_c = m_cache_VProd(c, rall, rall);
    m_tmp_rvrv += VProd_c * Ni(c);
  }
  bob::math::inv(m_tmp_rvrv, m_cache_IdPlusVProd_i); // m_cache_IdPlusVProd_i = ( I+Vt*diag(sigma)^-1*Ni*V)^-1
}

void bob::trainer::FABaseTrainer::computeFn_y_i(const bob::machine::FABase& mb,
  const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& stats, const size_t id)
{
  const blitz::Array<double,2>& U = mb.getU();
  const blitz::Array<double,1>& d = mb.getD();
  // Compute Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) (Normalised first order statistics)
  const blitz::Array<double,1>& Fi = m_Facc[id];
  const blitz::Array<double,1>& m = mb.getUbmMean();
  const blitz::Array<double,1>& z = m_z[id];
  bob::core::array::repelem(m_Nacc[id], m_tmp_CD);
  m_cache_Fn_y_i = Fi - m_tmp_CD * (m + d * z); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i})
  const blitz::Array<double,2>& X = m_x[id];
  blitz::Range rall = blitz::Range::all();
  for (int h=0; h<X.extent(1); ++h) // Loops over the sessions
  {
    blitz::Array<double,1> Xh = X(rall, h); // Xh = x_{i,h} (length: ru)
    bob::math::prod(U, Xh, m_tmp_CD_b); // m_tmp_CD_b = U*x_{i,h}
    const blitz::Array<double,1>& Nih = stats[h]->n;
    bob::core::array::repelem(Nih, m_tmp_CD);
    m_cache_Fn_y_i -= m_tmp_CD * m_tmp_CD_b; // N_{i,h} * U * x_{i,h}
  }
  // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
}

void bob::trainer::FABaseTrainer::updateY_i(const size_t id)
{
  // Computes yi = Ayi * Cvs * Fn_yi
  blitz::Array<double,1>& y = m_y[id];
  // m_tmp_rv = m_cache_VtSigmaInv * m_cache_Fn_y_i = Vt*diag(sigma)^-1 * sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
  bob::math::prod(m_cache_VtSigmaInv, m_cache_Fn_y_i, m_tmp_rv);
  bob::math::prod(m_cache_IdPlusVProd_i, m_tmp_rv, y);
}

void bob::trainer::FABaseTrainer::updateY(const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Precomputation
  computeVtSigmaInv(m);
  computeVProd(m);
  // Loops over all people
  for (size_t id=0; id<stats.size(); ++id) {
    computeIdPlusVProd_i(id);
    computeFn_y_i(m, stats[id], id);
    updateY_i(id);
  }
}

void bob::trainer::FABaseTrainer::computeAccumulatorsV(
  const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Initializes the cache accumulator
  m_acc_V_A1 = 0.;
  m_acc_V_A2 = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range rall = blitz::Range::all();
  for (size_t id=0; id<stats.size(); ++id) {
    computeIdPlusVProd_i(id);
    computeFn_y_i(m, stats[id], id);

    // Needs to return values to be accumulated for estimating V
    const blitz::Array<double,1>& y = m_y[id];
    m_tmp_rvrv = m_cache_IdPlusVProd_i;
    m_tmp_rvrv += y(i) * y(j);
    for (size_t c=0; c<m_dim_C; ++c)
    {
      blitz::Array<double,2> A1_y_c = m_acc_V_A1(c, rall, rall);
      A1_y_c += m_tmp_rvrv * m_Nacc[id](c);
    }
    m_acc_V_A2 += m_cache_Fn_y_i(i) * y(j);
  }
}

void bob::trainer::FABaseTrainer::updateV(blitz::Array<double,2>& V)
{
  blitz::Range rall = blitz::Range::all();
  for (size_t c=0; c<m_dim_C; ++c)
  {
    const blitz::Array<double,2> A1 = m_acc_V_A1(c, rall, rall);
    bob::math::inv(A1, m_tmp_rvrv);
    const blitz::Array<double,2> A2 = m_acc_V_A2(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1), rall);
    blitz::Array<double,2> V_c = V(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1), rall);
    bob::math::prod(A2, m_tmp_rvrv, V_c);
  }
}


//////////////////////////// U ///////////////////////////
void bob::trainer::FABaseTrainer::computeUtSigmaInv(const bob::machine::FABase& m)
{
  const blitz::Array<double,2>& U = m.getU();
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not
  // provide a non-const version of transpose())
  const blitz::Array<double,2> Ut = const_cast<blitz::Array<double,2>&>(U).transpose(1,0);
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  blitz::firstIndex i;
  blitz::secondIndex j;
  m_cache_UtSigmaInv = Ut(i,j) / sigma(j); // Ut * diag(sigma)^-1
}

void bob::trainer::FABaseTrainer::computeUProd(const bob::machine::FABase& m)
{
  const blitz::Array<double,2>& U = m.getU();
  blitz::firstIndex i;
  blitz::secondIndex j;
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  for (size_t c=0; c<m_dim_C; ++c)
  {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> Uu_c = U(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1), blitz::Range::all());
    blitz::Array<double,2> Ut_c = Uu_c.transpose(1,0);
    blitz::Array<double,1> sigma_c = sigma(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1));
    m_tmp_ruD = Ut_c(i,j) / sigma_c(j); // Ut_c * diag(sigma)^-1
    bob::math::prod(m_tmp_ruD, Uu_c, UProd_c);
  }
}

void bob::trainer::FABaseTrainer::computeIdPlusUProd_ih(
  const boost::shared_ptr<bob::machine::GMMStats>& stats)
{
  const blitz::Array<double,1>& Nih = stats->n;
  bob::math::eye(m_tmp_ruru); // m_tmp_ruru = I
  for (size_t c=0; c<m_dim_C; ++c) {
    blitz::Array<double,2> UProd_c = m_cache_UProd(c,blitz::Range::all(),blitz::Range::all());
    m_tmp_ruru += UProd_c * Nih(c);
  }
  bob::math::inv(m_tmp_ruru, m_cache_IdPlusUProd_ih); // m_cache_IdPlusUProd_ih = ( I+Ut*diag(sigma)^-1*Ni*U)^-1
}

void bob::trainer::FABaseTrainer::computeFn_x_ih(const bob::machine::FABase& mb,
  const boost::shared_ptr<bob::machine::GMMStats>& stats, const size_t id)
{
  const blitz::Array<double,2>& V = mb.getV();
  const blitz::Array<double,1>& d =  mb.getD();
  // Compute Fn_x_ih = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i}) (Normalised first order statistics)
  const blitz::Array<double,2>& Fih = stats->sumPx;
  const blitz::Array<double,1>& m = mb.getUbmMean();
  const blitz::Array<double,1>& z = m_z[id];
  const blitz::Array<double,1>& Nih = stats->n;
  bob::core::array::repelem(Nih, m_tmp_CD);
  for (size_t c=0; c<m_dim_C; ++c) {
    blitz::Array<double,1> Fn_x_ih_c = m_cache_Fn_x_ih(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1));
    Fn_x_ih_c = Fih(c,blitz::Range::all());
  }
  m_cache_Fn_x_ih -= m_tmp_CD * (m + d * z); // Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i})

  const blitz::Array<double,1>& y = m_y[id];
  bob::math::prod(V, y, m_tmp_CD_b);
  m_cache_Fn_x_ih -= m_tmp_CD * m_tmp_CD_b;
  // Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
}

void bob::trainer::FABaseTrainer::updateX_ih(const size_t id, const size_t h)
{
  // Computes xih = Axih * Cus * Fn_x_ih
  blitz::Array<double,1> x = m_x[id](blitz::Range::all(), h);
  // m_tmp_ru = m_cache_UtSigmaInv * m_cache_Fn_x_ih = Ut*diag(sigma)^-1 * N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
  bob::math::prod(m_cache_UtSigmaInv, m_cache_Fn_x_ih, m_tmp_ru);
  bob::math::prod(m_cache_IdPlusUProd_ih, m_tmp_ru, x);
}

void bob::trainer::FABaseTrainer::updateX(const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Precomputation
  computeUtSigmaInv(m);
  computeUProd(m);
  // Loops over all people
  for (size_t id=0; id<stats.size(); ++id) {
    int n_session_i = stats[id].size();
    for (int s=0; s<n_session_i; ++s) {
      computeIdPlusUProd_ih(stats[id][s]);
      computeFn_x_ih(m, stats[id][s], id);
      updateX_ih(id, s);
    }
  }
}

void bob::trainer::FABaseTrainer::computeAccumulatorsU(
  const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Initializes the cache accumulator
  m_acc_U_A1 = 0.;
  m_acc_U_A2 = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range rall = blitz::Range::all();
  for (size_t id=0; id<stats.size(); ++id) {
    int n_session_i = stats[id].size();
    for (int h=0; h<n_session_i; ++h) {
      computeIdPlusUProd_ih(stats[id][h]);
      computeFn_x_ih(m, stats[id][h], id);

      // Needs to return values to be accumulated for estimating U
      blitz::Array<double,1> x = m_x[id](rall, h);
      m_tmp_ruru = m_cache_IdPlusUProd_ih;
      m_tmp_ruru += x(i) * x(j);
      for (int c=0; c<(int)m_dim_C; ++c)
      {
        blitz::Array<double,2> A1_x_c = m_acc_U_A1(c,rall,rall);
        A1_x_c += m_tmp_ruru * stats[id][h]->n(c);
      }
      m_acc_U_A2 += m_cache_Fn_x_ih(i) * x(j);
    }
  }
}

void bob::trainer::FABaseTrainer::updateU(blitz::Array<double,2>& U)
{
  for (size_t c=0; c<m_dim_C; ++c)
  {
    const blitz::Array<double,2> A1 = m_acc_U_A1(c,blitz::Range::all(),blitz::Range::all());
    bob::math::inv(A1, m_tmp_ruru);
    const blitz::Array<double,2> A2 = m_acc_U_A2(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1),blitz::Range::all());
    blitz::Array<double,2> U_c = U(blitz::Range(c*m_dim_D,(c+1)*m_dim_D-1),blitz::Range::all());
    bob::math::prod(A2, m_tmp_ruru, U_c);
  }
}


//////////////////////////// D ///////////////////////////
void bob::trainer::FABaseTrainer::computeDtSigmaInv(const bob::machine::FABase& m)
{
  const blitz::Array<double,1>& d = m.getD();
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  m_cache_DtSigmaInv = d / sigma; // Dt * diag(sigma)^-1
}

void bob::trainer::FABaseTrainer::computeDProd(const bob::machine::FABase& m)
{
  const blitz::Array<double,1>& d = m.getD();
  const blitz::Array<double,1>& sigma = m.getUbmVariance();
  m_cache_DProd = d / sigma * d; // Dt * diag(sigma)^-1 * D
}

void bob::trainer::FABaseTrainer::computeIdPlusDProd_i(const size_t id)
{
  const blitz::Array<double,1>& Ni = m_Nacc[id];
  bob::core::array::repelem(Ni, m_tmp_CD); // m_tmp_CD = Ni 'repmat'
  m_cache_IdPlusDProd_i = 1.; // m_cache_IdPlusDProd_i = Id
  m_cache_IdPlusDProd_i += m_cache_DProd * m_tmp_CD; // m_cache_IdPlusDProd_i = I+Dt*diag(sigma)^-1*Ni*D
  m_cache_IdPlusDProd_i = 1 / m_cache_IdPlusDProd_i; // m_cache_IdPlusVProd_i = (I+Dt*diag(sigma)^-1*Ni*D)^-1
}

void bob::trainer::FABaseTrainer::computeFn_z_i(
  const bob::machine::FABase& mb,
  const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& stats, const size_t id)
{
  const blitz::Array<double,2>& U = mb.getU();
  const blitz::Array<double,2>& V = mb.getV();
  // Compute Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h}) (Normalised first order statistics)
  const blitz::Array<double,1>& Fi = m_Facc[id];
  const blitz::Array<double,1>& m = mb.getUbmMean();
  const blitz::Array<double,1>& y = m_y[id];
  bob::core::array::repelem(m_Nacc[id], m_tmp_CD);
  bob::math::prod(V, y, m_tmp_CD_b); // m_tmp_CD_b = V * y
  m_cache_Fn_z_i = Fi - m_tmp_CD * (m + m_tmp_CD_b); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i})

  const blitz::Array<double,2>& X = m_x[id];
  blitz::Range rall = blitz::Range::all();
  for (int h=0; h<X.extent(1); ++h) // Loops over the sessions
  {
    const blitz::Array<double,1>& Nh = stats[h]->n; // Nh = N_{i,h} (length: C)
    bob::core::array::repelem(Nh, m_tmp_CD);
    blitz::Array<double,1> Xh = X(rall, h); // Xh = x_{i,h} (length: ru)
    bob::math::prod(U, Xh, m_tmp_CD_b);
    m_cache_Fn_z_i -= m_tmp_CD * m_tmp_CD_b;
  }
  // Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
}

void bob::trainer::FABaseTrainer::updateZ_i(const size_t id)
{
  // Computes zi = Azi * D^T.Sigma^-1 * Fn_zi
  blitz::Array<double,1>& z = m_z[id];
  // m_tmp_CD = m_cache_DtSigmaInv * m_cache_Fn_z_i = Dt*diag(sigma)^-1 * sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
  z = m_cache_IdPlusDProd_i * m_cache_DtSigmaInv * m_cache_Fn_z_i;
}

void bob::trainer::FABaseTrainer::updateZ(const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Precomputation
  computeDtSigmaInv(m);
  computeDProd(m);
  // Loops over all people
  for (size_t id=0; id<m_Nid; ++id) {
    computeIdPlusDProd_i(id);
    computeFn_z_i(m, stats[id], id);
    updateZ_i(id);
  }
}

void bob::trainer::FABaseTrainer::computeAccumulatorsD(
  const bob::machine::FABase& m,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats)
{
  // Initializes the cache accumulator
  m_acc_D_A1 = 0.;
  m_acc_D_A2 = 0.;
  // Loops over all people
  blitz::firstIndex i;
  blitz::secondIndex j;
  for (size_t id=0; id<stats.size(); ++id) {
    computeIdPlusDProd_i(id);
    computeFn_z_i(m, stats[id], id);

    // Needs to return values to be accumulated for estimating D
    blitz::Array<double,1> z = m_z[id];
    bob::core::array::repelem(m_Nacc[id], m_tmp_CD);
    m_acc_D_A1 += (m_cache_IdPlusDProd_i + z * z) * m_tmp_CD;
    m_acc_D_A2 += m_cache_Fn_z_i * z;
  }
}

void bob::trainer::FABaseTrainer::updateD(blitz::Array<double,1>& d)
{
  d = m_acc_D_A2 / m_acc_D_A1;
}

void bob::trainer::FABaseTrainer::initializeRandom(blitz::Array<double,1>& vector)
{
  ranlib::Normal<double> normalGen(0., 1.);
  for (int i=0; i<vector.extent(0); ++i)
    vector(i) = normalGen.random();    // normal random number
}

void bob::trainer::FABaseTrainer::initializeRandom(blitz::Array<double,2>& matrix)
{
  ranlib::Normal<double> normalGen(0., 1.);
  for (int i=0; i<matrix.extent(0); ++i) {
    blitz::Array<double,1> vec = matrix(i, blitz::Range::all());
    initializeRandom(vec);
  }
}




//////////////////////////// ISVTrainer ///////////////////////////
bob::trainer::ISVTrainer::ISVTrainer(const size_t max_iterations, const double relevance_factor):
  EMTrainer<bob::machine::ISVBase, std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >
    (0.001, max_iterations, false),
  m_relevance_factor(relevance_factor)
{
}

bob::trainer::ISVTrainer::ISVTrainer(const bob::trainer::ISVTrainer& other):
  EMTrainer<bob::machine::ISVBase, std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >
    (other.m_convergence_threshold, other.m_max_iterations,
     other.m_compute_likelihood),
  m_relevance_factor(other.m_relevance_factor)
{
}

bob::trainer::ISVTrainer::~ISVTrainer()
{
}

bob::trainer::ISVTrainer& bob::trainer::ISVTrainer::operator=
(const bob::trainer::ISVTrainer& other)
{
  if (this != &other)
  {
    bob::trainer::EMTrainer<bob::machine::ISVBase,
      std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >::operator=(other);
    m_relevance_factor = other.m_relevance_factor;
  }
  return *this;
}

bool bob::trainer::ISVTrainer::operator==(const bob::trainer::ISVTrainer& b) const
{
  return bob::trainer::EMTrainer<bob::machine::ISVBase,
            std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >::operator==(b) &&
          m_relevance_factor == b.m_relevance_factor;
}

bool bob::trainer::ISVTrainer::operator!=(const bob::trainer::ISVTrainer& b) const
{
  return !(this->operator==(b));
}

bool bob::trainer::ISVTrainer::is_similar_to(const bob::trainer::ISVTrainer& b,
  const double r_epsilon, const double a_epsilon) const
{
  return bob::trainer::EMTrainer<bob::machine::ISVBase,
            std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >::is_similar_to(b, r_epsilon, a_epsilon) &&
          m_relevance_factor == b.m_relevance_factor;
}

void bob::trainer::ISVTrainer::initialize(bob::machine::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  m_base_trainer.initUbmNidSumStatistics(machine.getBase(), ar);
  m_base_trainer.initializeXYZ(ar);

  m_base_trainer.initializeRandom(machine.updateU());
  initializeD(machine);
  machine.precompute();
}

void bob::trainer::ISVTrainer::initializeD(bob::machine::ISVBase& machine) const
{
  // D = sqrt(variance(UBM) / relevance_factor)
  blitz::Array<double,1> d = machine.updateD();
  d = sqrt(machine.getBase().getUbmVariance() / m_relevance_factor);
}

void bob::trainer::ISVTrainer::finalize(bob::machine::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
}

void bob::trainer::ISVTrainer::eStep(bob::machine::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  m_base_trainer.resetXYZ();

  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
  m_base_trainer.updateZ(base, ar);
  m_base_trainer.computeAccumulatorsU(base, ar);
}

void bob::trainer::ISVTrainer::mStep(bob::machine::ISVBase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  blitz::Array<double,2>& U = machine.updateU();
  m_base_trainer.updateU(U);
  machine.precompute();
}

double bob::trainer::ISVTrainer::computeLikelihood(bob::machine::ISVBase& machine)
{
  // TODO
  return 0;
}

void bob::trainer::ISVTrainer::enrol(bob::machine::ISVMachine& machine,
  const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& ar,
  const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > vvec;
  vvec.push_back(ar);

  const bob::machine::FABase& fb = machine.getISVBase()->getBase();

  m_base_trainer.initUbmNidSumStatistics(fb, vvec);
  m_base_trainer.initializeXYZ(vvec);

  for (size_t i=0; i<n_iter; ++i) {
    m_base_trainer.updateX(fb, vvec);
    m_base_trainer.updateZ(fb, vvec);
  }

  const blitz::Array<double,1> z(m_base_trainer.getZ()[0]);
  machine.setZ(z);
}



//////////////////////////// JFATrainer ///////////////////////////
bob::trainer::JFATrainer::JFATrainer(const size_t max_iterations):
  m_max_iterations(max_iterations), m_rng(new boost::mt19937())
{
}

bob::trainer::JFATrainer::JFATrainer(const bob::trainer::JFATrainer& other):
  m_max_iterations(other.m_max_iterations), m_rng(other.m_rng)
{
}

bob::trainer::JFATrainer::~JFATrainer()
{
}

bob::trainer::JFATrainer& bob::trainer::JFATrainer::operator=
(const bob::trainer::JFATrainer& other)
{
  if (this != &other)
  {
    m_max_iterations = other.m_max_iterations;
    m_rng = other.m_rng;
  }
  return *this;
}

bool bob::trainer::JFATrainer::operator==(const bob::trainer::JFATrainer& b) const
{
  return m_max_iterations == b.m_max_iterations && *m_rng == *(b.m_rng);
}

bool bob::trainer::JFATrainer::operator!=(const bob::trainer::JFATrainer& b) const
{
  return !(this->operator==(b));
}

bool bob::trainer::JFATrainer::is_similar_to(const bob::trainer::JFATrainer& b,
  const double r_epsilon, const double a_epsilon) const
{
  return m_max_iterations == b.m_max_iterations && *m_rng == *(b.m_rng);
}

void bob::trainer::JFATrainer::initialize(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  m_base_trainer.initUbmNidSumStatistics(machine.getBase(), ar);
  m_base_trainer.initializeXYZ(ar);

  m_base_trainer.initializeRandom(machine.updateU());
  m_base_trainer.initializeRandom(machine.updateV());
  m_base_trainer.initializeRandom(machine.updateD());
  machine.precompute();
}

void bob::trainer::JFATrainer::eStep1(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateY(base, ar);
  m_base_trainer.computeAccumulatorsV(base, ar);
}

void bob::trainer::JFATrainer::mStep1(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  blitz::Array<double,2>& V = machine.updateV();
  m_base_trainer.updateV(V);
}

void bob::trainer::JFATrainer::finalize1(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateY(base, ar);
}


void bob::trainer::JFATrainer::eStep2(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
  m_base_trainer.computeAccumulatorsU(base, ar);
}

void bob::trainer::JFATrainer::mStep2(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  blitz::Array<double,2>& U = machine.updateU();
  m_base_trainer.updateU(U);
  machine.precompute();
}

void bob::trainer::JFATrainer::finalize2(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateX(base, ar);
}


void bob::trainer::JFATrainer::eStep3(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  const bob::machine::FABase& base = machine.getBase();
  m_base_trainer.updateZ(base, ar);
  m_base_trainer.computeAccumulatorsD(base, ar);
}

void bob::trainer::JFATrainer::mStep3(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  blitz::Array<double,1>& d = machine.updateD();
  m_base_trainer.updateD(d);
}

void bob::trainer::JFATrainer::finalize3(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
}

void bob::trainer::JFATrainer::train_loop(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  // V subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep1(machine, ar);
    mStep1(machine, ar);
  }
  finalize1(machine, ar);
  // U subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep2(machine, ar);
    mStep2(machine, ar);
  }
  finalize2(machine, ar);
  // d subspace
  for (size_t i=0; i<m_max_iterations; ++i) {
    eStep3(machine, ar);
    mStep3(machine, ar);
  }
  finalize3(machine, ar);
}

void bob::trainer::JFATrainer::train(bob::machine::JFABase& machine,
  const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar)
{
  initialize(machine, ar);
  train_loop(machine, ar);
}

void bob::trainer::JFATrainer::enrol(bob::machine::JFAMachine& machine,
  const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& ar,
  const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > vvec;
  vvec.push_back(ar);

  const bob::machine::FABase& fb = machine.getJFABase()->getBase();

  m_base_trainer.initUbmNidSumStatistics(fb, vvec);
  m_base_trainer.initializeXYZ(vvec);

  for (size_t i=0; i<n_iter; ++i) {
    m_base_trainer.updateY(fb, vvec);
    m_base_trainer.updateX(fb, vvec);
    m_base_trainer.updateZ(fb, vvec);
  }

  const blitz::Array<double,1> y(m_base_trainer.getY()[0]);
  const blitz::Array<double,1> z(m_base_trainer.getZ()[0]);
  machine.setY(y);
  machine.setZ(z);
}

