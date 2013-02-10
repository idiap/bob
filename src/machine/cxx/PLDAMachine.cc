/**
 * @file machine/cxx/PLDAMachine.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Machines that implements the PLDA model
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

#include "bob/core/assert.h"
#include "bob/core/array_copy.h"
#include "bob/machine/Exception.h"
#include "bob/machine/PLDAMachine.h"
#include "bob/math/linear.h"
#include "bob/math/det.h"
#include "bob/math/inv.h"

#include <cmath>
#include <boost/lexical_cast.hpp>
#include <string>

#include "bob/core/logging.h"

bob::machine::PLDABaseMachine::PLDABaseMachine()
{
  resizeNoInit(0, 0, 0);
}

bob::machine::PLDABaseMachine::PLDABaseMachine(const size_t dim_d, 
  const size_t dim_f, const size_t dim_g, const double variance_threshold)
{
  resize(dim_d, dim_f, dim_g, variance_threshold);
}


bob::machine::PLDABaseMachine::PLDABaseMachine(const bob::machine::PLDABaseMachine& other):
  m_dim_d(other.m_dim_d),
  m_dim_f(other.m_dim_f),
  m_dim_g(other.m_dim_g),
  m_F(bob::core::array::ccopy(other.m_F)), 
  m_G(bob::core::array::ccopy(other.m_G)), 
  m_sigma(bob::core::array::ccopy(other.m_sigma)), 
  m_mu(bob::core::array::ccopy(other.m_mu)),
  m_variance_thresholds(bob::core::array::ccopy(other.m_variance_thresholds)),
  m_isigma(bob::core::array::ccopy(other.m_isigma)), 
  m_alpha(bob::core::array::ccopy(other.m_alpha)),
  m_beta(bob::core::array::ccopy(other.m_beta)),
  m_gamma(),
  m_Ft_beta(bob::core::array::ccopy(other.m_Ft_beta)),
  m_Gt_isigma(bob::core::array::ccopy(other.m_Gt_isigma)), 
  m_logdet_alpha(other.m_logdet_alpha),
  m_logdet_sigma(other.m_logdet_sigma),
  m_loglike_constterm(other.m_loglike_constterm),
  m_cache_d_1(bob::core::array::ccopy(other.m_cache_d_1)),
  m_cache_d_2(bob::core::array::ccopy(other.m_cache_d_2)),
  m_cache_d_ng_1(bob::core::array::ccopy(other.m_cache_d_ng_1)), 
  m_cache_nf_nf_1(bob::core::array::ccopy(other.m_cache_nf_nf_1)), 
  m_cache_ng_ng_1(bob::core::array::ccopy(other.m_cache_ng_ng_1))
{
  bob::core::array::ccopy(other.m_gamma, m_gamma);
}

bob::machine::PLDABaseMachine::PLDABaseMachine(bob::io::HDF5File& config) {
  load(config);
}

bob::machine::PLDABaseMachine::~PLDABaseMachine() {
}

bob::machine::PLDABaseMachine& bob::machine::PLDABaseMachine::operator=
    (const bob::machine::PLDABaseMachine& other) 
{
  if(this!=&other)
  {
    m_dim_d = other.m_dim_d;
    m_dim_f = other.m_dim_f;
    m_dim_g = other.m_dim_g;
    m_F.reference(bob::core::array::ccopy(other.m_F));
    m_G.reference(bob::core::array::ccopy(other.m_G));
    m_sigma.reference(bob::core::array::ccopy(other.m_sigma));
    m_mu.reference(bob::core::array::ccopy(other.m_mu));
    m_variance_thresholds.reference(bob::core::array::ccopy(other.m_variance_thresholds));
    m_isigma.reference(bob::core::array::ccopy(other.m_isigma));
    m_alpha.reference(bob::core::array::ccopy(other.m_alpha));
    m_beta.reference(bob::core::array::ccopy(other.m_beta));
    bob::core::array::ccopy(other.m_gamma, m_gamma);
    m_Ft_beta.reference(bob::core::array::ccopy(other.m_Ft_beta));
    m_Gt_isigma.reference(bob::core::array::ccopy(other.m_Gt_isigma));
    m_logdet_alpha = other.m_logdet_alpha;
    m_logdet_sigma = other.m_logdet_sigma;
    m_loglike_constterm = other.m_loglike_constterm;
    m_cache_d_1.reference(bob::core::array::ccopy(other.m_cache_d_1));
    m_cache_d_2.reference(bob::core::array::ccopy(other.m_cache_d_2));
    m_cache_d_ng_1.reference(bob::core::array::ccopy(other.m_cache_d_ng_1));
    m_cache_nf_nf_1.reference(bob::core::array::ccopy(other.m_cache_nf_nf_1));
    m_cache_ng_ng_1.reference(bob::core::array::ccopy(other.m_cache_ng_ng_1));
  }
  return *this;
}

bool bob::machine::PLDABaseMachine::operator==
    (const bob::machine::PLDABaseMachine& b) const
{
  // Check dimensions
  if(this->m_dim_d != b.m_dim_d || this->m_dim_f != b.m_dim_f ||
     this->m_dim_g != b.m_dim_g)
    return false;

  // Check content
  if(blitz::any(this->m_F != b.m_F) || blitz::any(this->m_G != b.m_G) || 
     blitz::any(this->m_sigma != b.m_sigma) || blitz::any(this->m_mu != b.m_mu) ||
     blitz::any(this->m_variance_thresholds != b.m_variance_thresholds))
    return false;

  // Check additional content
  if(blitz::any(this->m_isigma != b.m_isigma) || blitz::any(this->m_alpha != b.m_alpha) || 
     blitz::any(this->m_beta != b.m_beta) || blitz::any(this->m_Ft_beta != b.m_Ft_beta) ||
     blitz::any(this->m_Gt_isigma != b.m_Gt_isigma) ||
     this->m_logdet_alpha != b.m_logdet_alpha || this->m_logdet_sigma != b.m_logdet_sigma)
    return false;

  // Check additional maps
  // 1. m_gamma
  {
    if(this->m_gamma.size() != b.m_gamma.size())
      return false;  // differing sizes, they are not the same
    std::map<size_t, blitz::Array<double,2> >::const_iterator i, j;
    for(i = this->m_gamma.begin(), j = b.m_gamma.begin(); 
      i != this->m_gamma.end(); ++i, ++j)
    {
      if(i->first != j->first || blitz::any(i->second != j->second))
        return false;
    }
  }
  // 2. m_loglike_constterm
  {
    if(this->m_loglike_constterm.size() != b.m_loglike_constterm.size())
      return false;  // differing sizes, they are not the same
    std::map<size_t, double>::const_iterator i, j;
    for(i = this->m_loglike_constterm.begin(), j = b.m_loglike_constterm.begin(); 
      i != this->m_loglike_constterm.end(); ++i, ++j)
    {
      if(i->first != j->first || i->second != j->second)
        return false;
    }
  }

  return true;
}

bool bob::machine::PLDABaseMachine::operator!=
    (const bob::machine::PLDABaseMachine& b) const 
{
  return !(this->operator==(b));
}

void bob::machine::PLDABaseMachine::load(bob::io::HDF5File& config) 
{
  if(!config.contains("dim_d"))
  {
    // Then the model was saved using bob < 1.2.0
    //reads all data directly into the member variables
    m_F.reference(config.readArray<double,2>("F"));
    m_G.reference(config.readArray<double,2>("G"));
    m_dim_d = m_F.extent(0);
    m_dim_f = m_F.extent(1);
    m_dim_g = m_G.extent(1);
    m_sigma.reference(config.readArray<double,1>("sigma"));
    m_mu.reference(config.readArray<double,1>("mu"));
    m_isigma.resize(m_dim_d);
    precomputeISigma();
    m_variance_thresholds.resize(m_dim_d);
    m_variance_thresholds = 0.; 
    m_alpha.reference(config.readArray<double,2>("alpha"));
    m_beta.reference(config.readArray<double,2>("beta"));
    // gamma and log like constant term (a-dependent terms)
    if(config.contains("a_indices"))
    {
      blitz::Array<uint32_t, 1> a_indices;
      a_indices.reference(config.readArray<uint32_t,1>("a_indices"));
      for(int i=0; i<a_indices.extent(0); ++i)
      {
        std::string str1 = "gamma_" + boost::lexical_cast<std::string>(a_indices(i));
        m_gamma[a_indices(i)].reference(config.readArray<double,2>(str1));
        std::string str2 = "loglikeconstterm_" + boost::lexical_cast<std::string>(a_indices(i));
        m_loglike_constterm[a_indices(i)] = config.read<double>(str2);
      }
    }
    m_Ft_beta.reference(config.readArray<double,2>("Ft_beta"));
    m_Gt_isigma.reference(config.readArray<double,2>("Gt_isigma"));
    m_logdet_alpha = config.read<double>("logdet_alpha");
    m_logdet_sigma = config.read<double>("logdet_sigma");
  }
  else
  {
    // Then the model was saved using bob >= 1.2.0
    //reads all data directly into the member variables
    m_F.reference(config.readArray<double,2>("F"));
    m_G.reference(config.readArray<double,2>("G"));
    // Conditional because previous versions had not these variables
    m_dim_d = config.read<uint64_t>("dim_d");
    m_dim_f = config.read<uint64_t>("dim_f");
    m_dim_g = config.read<uint64_t>("dim_g");
    m_sigma.reference(config.readArray<double,1>("sigma"));
    m_mu.reference(config.readArray<double,1>("mu"));
    m_isigma.resize(m_dim_d);
    precomputeISigma();
    m_variance_thresholds.reference(config.readArray<double,1>("variance_thresholds"));
    m_alpha.reference(config.readArray<double,2>("alpha"));
    m_beta.reference(config.readArray<double,2>("beta"));
    // gamma's (a-dependent terms)
    if(config.contains("a_indices_gamma"))
    {
      blitz::Array<uint32_t, 1> a_indices;
      a_indices.reference(config.readArray<uint32_t,1>("a_indices_gamma"));
      for(int i=0; i<a_indices.extent(0); ++i)
      {
        std::string str = "gamma_" + boost::lexical_cast<std::string>(a_indices(i));
        m_gamma[a_indices(i)].reference(config.readArray<double,2>(str));
      }
    }
    // log likelihood constant term's (a-dependent terms)
    if(config.contains("a_indices_loglikeconstterm"))
    {
      blitz::Array<uint32_t, 1> a_indices;
      a_indices.reference(config.readArray<uint32_t,1>("a_indices_loglikeconstterm"));
      for(int i=0; i<a_indices.extent(0); ++i)
      {
        std::string str = "loglikeconstterm_" + boost::lexical_cast<std::string>(a_indices(i));
        m_loglike_constterm[a_indices(i)] = config.read<double>(str);
      }
    }
    m_Ft_beta.reference(config.readArray<double,2>("Ft_beta"));
    m_Gt_isigma.reference(config.readArray<double,2>("Gt_isigma"));
    m_logdet_alpha = config.read<double>("logdet_alpha");
    m_logdet_sigma = config.read<double>("logdet_sigma");
  }
  m_cache_d_1.resize(m_dim_d);
  m_cache_d_2.resize(m_dim_d);
  m_cache_d_ng_1.resize(m_dim_d,m_dim_g);
  m_cache_nf_nf_1.resize(m_dim_f,m_dim_f);
  m_cache_ng_ng_1.resize(m_dim_g,m_dim_g);
}

void bob::machine::PLDABaseMachine::save(bob::io::HDF5File& config) const 
{
  config.set("dim_d", (uint64_t)m_dim_d);
  config.set("dim_f", (uint64_t)m_dim_f);
  config.set("dim_g", (uint64_t)m_dim_g);
  config.setArray("F", m_F);
  config.setArray("G", m_G);
  config.setArray("sigma", m_sigma);
  config.setArray("mu", m_mu);
  config.setArray("variance_thresholds", m_variance_thresholds);
  config.setArray("alpha", m_alpha);
  config.setArray("beta", m_beta);
  // gamma's
  if(m_gamma.size() > 0)
  {
    blitz::Array<uint32_t, 1> a_indices(m_gamma.size());
    int i = 0;
    for(std::map<size_t,blitz::Array<double,2> >::const_iterator 
        it=m_gamma.begin(); it!=m_gamma.end(); ++it)
    {
      a_indices(i) = it->first;
      std::string str = "gamma_" + boost::lexical_cast<std::string>(it->first);
      config.setArray(str, it->second);
      ++i;
    }
    config.setArray("a_indices_gamma", a_indices);
  }
  // log likelihood constant terms
  if(m_loglike_constterm.size() > 0)
  {
    blitz::Array<uint32_t, 1> a_indices(m_loglike_constterm.size());
    int i = 0;
    for(std::map<size_t,double>::const_iterator 
        it=m_loglike_constterm.begin(); it!=m_loglike_constterm.end(); ++it)
    {
      a_indices(i) = it->first;
      std::string str = "loglikeconstterm_" + boost::lexical_cast<std::string>(it->first);
      config.set(str, it->second);
      ++i;
    }
    config.setArray("a_indices_loglikeconstterm", a_indices);
  }

  config.setArray("Ft_beta", m_Ft_beta);
  config.setArray("Gt_isigma", m_Gt_isigma);
  config.set("logdet_alpha", m_logdet_alpha);
  config.set("logdet_sigma", m_logdet_sigma);
}

void bob::machine::PLDABaseMachine::resizeNoInit(const size_t dim_d, const size_t dim_f, 
    const size_t dim_g) 
{
  m_dim_d = dim_d;
  m_dim_f = dim_f;
  m_dim_g = dim_g;
  m_F.resize(dim_d,dim_f);
  m_G.resize(dim_d,dim_g);
  m_sigma.resize(dim_d);
  m_mu.resize(dim_d);
  m_variance_thresholds.resize(dim_d);
  m_alpha.resize(dim_g,dim_g);
  m_beta.resize(dim_d,dim_d);
  m_Ft_beta.resize(dim_f,dim_d);
  m_Gt_isigma.resize(dim_g,dim_d);
  m_gamma.clear();
  m_isigma.resize(dim_d);
  m_loglike_constterm.clear();
  m_cache_d_1.resize(dim_d);
  m_cache_d_2.resize(dim_d);
  m_cache_d_ng_1.resize(dim_d,dim_g);
  m_cache_nf_nf_1.resize(dim_f,dim_f);
  m_cache_ng_ng_1.resize(dim_g,dim_g);
}

void bob::machine::PLDABaseMachine::resize(const size_t dim_d, const size_t dim_f, 
    const size_t dim_g, const double variance_threshold) 
{
  resizeNoInit(dim_d, dim_f, dim_g);
  m_variance_thresholds = variance_threshold;
  initMuFGSigma();
}

void bob::machine::PLDABaseMachine::setF(const blitz::Array<double,2>& F) 
{
  bob::core::array::assertSameShape(F, m_F);
  m_F.reference(bob::core::array::ccopy(F));
  // Precomputes useful matrices
  precompute();
}

void bob::machine::PLDABaseMachine::setG(const blitz::Array<double,2>& G) 
{
  bob::core::array::assertSameShape(G, m_G);
  m_G.reference(bob::core::array::ccopy(G));
  // Precomputes useful matrices and values
  precompute();
  precomputeLogDetAlpha();
}

void bob::machine::PLDABaseMachine::setSigma(const blitz::Array<double,1>& sigma) 
{
  bob::core::array::assertSameShape(sigma, m_sigma);
  m_sigma.reference(bob::core::array::ccopy(sigma));
  // Apply variance flooring thresholds: This will also
  // call the precompute() and precomputeLogLike() methods!
  applyVarianceThresholds();
}

void bob::machine::PLDABaseMachine::setMu(const blitz::Array<double,1>& mu) 
{
  bob::core::array::assertSameShape(mu, m_mu);
  m_mu.reference(bob::core::array::ccopy(mu));
}

void bob::machine::PLDABaseMachine::setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds) 
{
  // Check and set
  bob::core::array::assertSameShape(m_variance_thresholds, variance_thresholds);
  m_variance_thresholds = variance_thresholds;
  // Apply variance flooring thresholds: This will also
  // call the precompute() and precomputeLogLike() methods!
  applyVarianceThresholds();
}

void bob::machine::PLDABaseMachine::setVarianceThresholds(const double value) 
{
  // Variance flooring
  m_variance_thresholds = value;
  // Apply variance flooring thresholds: This will also
  // call the precompute() and precomputeLogLike() methods!
  applyVarianceThresholds();
}

void bob::machine::PLDABaseMachine::applyVarianceThresholds() 
{
   // Apply variance flooring threshold
  m_sigma = blitz::where( m_sigma < m_variance_thresholds, m_variance_thresholds, m_sigma);
  // Re-compute constants, because m_sigma has changed
  precompute();
  precomputeLogLike();
}

const blitz::Array<double,2>& bob::machine::PLDABaseMachine::getGamma(const size_t a)
{
  if(!hasGamma(a)) 
    throw std::runtime_error("Gamma for this number of samples is not currently in cache. You could use the getAddGamma() method instead");
  return m_gamma[a];
}

const blitz::Array<double,2>& bob::machine::PLDABaseMachine::getAddGamma(const size_t a)
{
  if(!hasGamma(a)) precomputeGamma(a);
  return m_gamma[a];
}

void bob::machine::PLDABaseMachine::initMuFGSigma() 
{
  // To avoid problems related to precomputation
  m_mu = 0.;
  bob::math::eye(m_F);
  bob::math::eye(m_G);
  m_sigma = 1.;
  // Precompute variables
  precompute();
  precomputeLogLike();
}

void bob::machine::PLDABaseMachine::precompute() 
{
  precomputeISigma();
  precomputeGtISigma();
  precomputeAlpha();
  precomputeBeta();
  m_gamma.clear();
  precomputeFtBeta();
  m_loglike_constterm.clear();
}

void bob::machine::PLDABaseMachine::precomputeLogLike() 
{
  precomputeLogDetAlpha();
  precomputeLogDetSigma();
}

void bob::machine::PLDABaseMachine::precomputeISigma() 
{
  // Updates inverse of sigma
  // TODO: check division by zero
  m_isigma = 1. / m_sigma;
}

void bob::machine::PLDABaseMachine::precomputeGtISigma() 
{
  // m_Gt_isigma = G^T \Sigma^{-1}
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> Gt = m_G.transpose(1,0);
  m_Gt_isigma = Gt(i,j) * m_isigma(j);
}

void bob::machine::PLDABaseMachine::precomputeAlpha() 
{
  // alpha = (Id + G^T.sigma^-1.G)^-1

  // m_cache_ng_ng_1 = G^T.sigma^-1.G
  bob::math::prod(m_Gt_isigma, m_G, m_cache_ng_ng_1);
  // m_cache_ng_ng_1 = Id + G^T.sigma^-1.G
  for(int i=0; i<m_cache_ng_ng_1.extent(0); ++i) m_cache_ng_ng_1(i,i) += 1;
  // m_alpha = (Id + G^T.sigma^-1.G)^-1
  bob::math::inv(m_cache_ng_ng_1, m_alpha);
}

void bob::machine::PLDABaseMachine::precomputeBeta() 
{
  // beta = (sigma + G.G^T)^-1
  // BUT, there is a more efficient computation (Woodbury identity):
  // beta = sigma^-1 - sigma^-1.G.(Id + G^T.sigma^-1.G)^-1.G^T.sigma^-1
  // beta =  sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1
  
  blitz::Array<double,2> GtISigmaT = m_Gt_isigma.transpose(1,0);
  // m_cache_d_ng_1 = sigma^-1.G.alpha
  bob::math::prod(GtISigmaT, m_alpha, m_cache_d_ng_1);
  // m_beta = -sigma^-1.G.alpha.G^T.sigma^-1
  bob::math::prod(m_cache_d_ng_1, m_Gt_isigma, m_beta);
  m_beta = -m_beta;
  // m_beta = sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1
  for(int i=0; i<m_beta.extent(0); ++i) m_beta(i,i) += m_isigma(i);
}

void bob::machine::PLDABaseMachine::precomputeGamma(const size_t a)
{
  
  blitz::Array<double,2> gamma_a(getDimF(),getDimF());
  m_gamma[a].reference(gamma_a);
  computeGamma(a, gamma_a);
}

void bob::machine::PLDABaseMachine::precomputeFtBeta() 
{
  // m_Ft_beta = F^T.beta = F^T.(sigma + G.G^T)^-1 
  blitz::Array<double,2> Ft = m_F.transpose(1,0);
  bob::math::prod(Ft, m_beta, m_Ft_beta);
}

void bob::machine::PLDABaseMachine::computeGamma(const size_t a, 
  blitz::Array<double,2> res) const
{
  // gamma = (Id + a.F^T.beta.F)^-1

  // Checks destination size
  bob::core::array::assertSameShape(res, m_cache_nf_nf_1);
  // m_cache_nf_nf_1 = F^T.beta.F
  bob::math::prod(m_Ft_beta, m_F, m_cache_nf_nf_1);
   // m_cache_nf_nf_1 = a.F^T.beta.F
  m_cache_nf_nf_1 *= static_cast<double>(a);
  // m_cache_nf_nf_1 = Id + a.F^T.beta.F
  for(int i=0; i<m_cache_nf_nf_1.extent(0); ++i) m_cache_nf_nf_1(i,i) += 1;

  // res = (Id + a.F^T.beta.F)^-1
  bob::math::inv(m_cache_nf_nf_1, res);
}

void bob::machine::PLDABaseMachine::precomputeLogDetAlpha()
{
  m_logdet_alpha = log(fabs(bob::math::det(m_alpha)));
}

void bob::machine::PLDABaseMachine::precomputeLogDetSigma()
{
  m_logdet_sigma = blitz::sum(blitz::log(m_sigma));
}

double bob::machine::PLDABaseMachine::computeLogLikeConstTerm(const size_t a,
  const blitz::Array<double,2>& gamma_a) const
{
  // loglike_constterm[a] = a/2 * 
  //  ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
  double logdet_gamma_a = log(fabs(bob::math::det(gamma_a)));
  double ah = static_cast<double>(a)/2.;
  double res = ( -ah*((double)m_dim_d)*log(2*M_PI) - 
      ah*m_logdet_sigma + ah*m_logdet_alpha + logdet_gamma_a/2.);
  return res;
}

double bob::machine::PLDABaseMachine::computeLogLikeConstTerm(const size_t a)
{
  const blitz::Array<double,2>& gamma_a = getAddGamma(a);
  return computeLogLikeConstTerm(a, gamma_a);
}

void bob::machine::PLDABaseMachine::precomputeLogLikeConstTerm(const size_t a)
{
  double val = computeLogLikeConstTerm(a); 
  m_loglike_constterm[a] = val;
}

double bob::machine::PLDABaseMachine::getLogLikeConstTerm(const size_t a)
{
  if(!hasLogLikeConstTerm(a))
    throw std::runtime_error("The LogLikelihood constant term for this number of samples is not currently in cache. You could use the getAddLogLikeConstTerm() method instead");
  return m_loglike_constterm[a];
}

double bob::machine::PLDABaseMachine::getAddLogLikeConstTerm(const size_t a)
{
  if(!hasLogLikeConstTerm(a)) precomputeLogLikeConstTerm(a);
  return m_loglike_constterm[a];
}

void bob::machine::PLDABaseMachine::clearMaps()
{
  m_gamma.clear();
  m_loglike_constterm.clear();
}

double bob::machine::PLDABaseMachine::computeLogLikelihoodPointEstimate(
  const blitz::Array<double,1>& xij, const blitz::Array<double,1>& hi, 
  const blitz::Array<double,1>& wij) const
{
  // Check inputs
  bob::core::array::assertSameDimensionLength(xij.extent(0), getDimD());
  bob::core::array::assertSameDimensionLength(hi.extent(0), getDimF());
  bob::core::array::assertSameDimensionLength(wij.extent(0), getDimG());
  // Computes: -D/2 log(2pi) -1/2 log(det(\Sigma)) 
  //   -1/2 {(x_{ij}-(\mu+Fh_{i}+Gw_{ij}))^{T}\Sigma^{-1}(x_{ij}-(\mu+Fh_{i}+Gw_{ij}))}
  double res = -0.5*((double)m_dim_d)*log(2*M_PI) - 0.5*m_logdet_sigma;
  // m_cache_d_1 = (x_{ij} - (\mu+Fh_{i}+Gw_{ij}))
  m_cache_d_1 = xij - m_mu;
  bob::math::prod(m_F, hi, m_cache_d_2);
  m_cache_d_1 -= m_cache_d_2;
  bob::math::prod(m_G, wij, m_cache_d_2);
  m_cache_d_1 -= m_cache_d_2;
  // add third term to res
  res += -0.5*blitz::sum(blitz::pow2(m_cache_d_1) * m_isigma);
  return res;
}

namespace bob{
  namespace machine{
    /**
     * @brief Prints a PLDABaseMachine in the output stream. This will print
     * the values of the parameters \f$\mu\f$, \f$F\f$, \f$G\f$ and 
     * \f$\Sigma\f$ of the PLDA model.
     */
    std::ostream& operator<<(std::ostream& os, const PLDABaseMachine& m) {
      os << "mu = " << m.m_mu << std::endl;
      os << "sigma = " << m.m_sigma << std::endl;
      os << "F = " << m.m_F << std::endl;
      os << "G = " << m.m_G << std::endl;
      return os;
    }
  }
}


bob::machine::PLDAMachine::PLDAMachine():
  m_plda_base(boost::shared_ptr<bob::machine::PLDABaseMachine>()),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(0), 
  m_loglikelihood(0), m_gamma(), m_loglike_constterm(),
  m_cache_d_1(0), m_cache_d_2(0), m_cache_nf_1(0), m_cache_nf_2(0)
{
}

bob::machine::PLDAMachine::PLDAMachine(const boost::shared_ptr<bob::machine::PLDABaseMachine> plda_base): 
  m_plda_base(plda_base),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(plda_base->getDimF()),
  m_loglikelihood(0), m_gamma(), m_loglike_constterm(),
  m_cache_d_1(plda_base->getDimD()), m_cache_d_2(plda_base->getDimD()),
  m_cache_nf_1(plda_base->getDimF()), m_cache_nf_2(plda_base->getDimF())
{
}


bob::machine::PLDAMachine::PLDAMachine(const bob::machine::PLDAMachine& other):
  m_plda_base(other.m_plda_base),
  m_n_samples(other.m_n_samples), 
  m_nh_sum_xit_beta_xi(other.m_nh_sum_xit_beta_xi), 
  m_weighted_sum(bob::core::array::ccopy(other.m_weighted_sum)),
  m_loglikelihood(other.m_loglikelihood), m_gamma(), 
  m_loglike_constterm(other.m_loglike_constterm),
  m_cache_d_1(bob::core::array::ccopy(other.m_cache_d_1)),
  m_cache_d_2(bob::core::array::ccopy(other.m_cache_d_2)),
  m_cache_nf_1(bob::core::array::ccopy(other.m_cache_nf_1)),
  m_cache_nf_2(bob::core::array::ccopy(other.m_cache_nf_2))
{
  bob::core::array::ccopy(other.m_gamma, m_gamma);
}

bob::machine::PLDAMachine::PLDAMachine(bob::io::HDF5File& config):
  m_plda_base(boost::shared_ptr<bob::machine::PLDABaseMachine>())
{
  load(config);
}

bob::machine::PLDAMachine::~PLDAMachine() {
}

bob::machine::PLDAMachine& bob::machine::PLDAMachine::operator=
(const bob::machine::PLDAMachine& other) 
{
  if(this!=&other)
  {
    m_plda_base = other.m_plda_base;
    m_n_samples = other.m_n_samples;
    m_nh_sum_xit_beta_xi = other.m_nh_sum_xit_beta_xi; 
    m_weighted_sum.reference(bob::core::array::ccopy(other.m_weighted_sum));
    m_loglikelihood = other.m_loglikelihood;
    bob::core::array::ccopy(other.m_gamma, m_gamma);
    m_loglike_constterm = other.m_loglike_constterm;
    m_cache_d_1.reference(bob::core::array::ccopy(other.m_cache_d_1));
    m_cache_d_2.reference(bob::core::array::ccopy(other.m_cache_d_2));
    m_cache_nf_1.reference(bob::core::array::ccopy(other.m_cache_nf_1));
    m_cache_nf_2.reference(bob::core::array::ccopy(other.m_cache_nf_2));
  }
  return *this;
}

bool bob::machine::PLDAMachine::operator==
    (const bob::machine::PLDAMachine& b) const
{
  // Check PLDABaseMachine 
  if(*(this->m_plda_base) != *(b.m_plda_base))
    return false;

  // Check content
  if(this->m_n_samples != b.m_n_samples || this->m_nh_sum_xit_beta_xi != b.m_nh_sum_xit_beta_xi ||
     blitz::any(this->m_weighted_sum != b.m_weighted_sum) || 
     this->m_loglikelihood != b.m_loglikelihood)
    return false;

  // Check additional maps
  // 1. m_gamma
  {
    if(this->m_gamma.size() != b.m_gamma.size())
      return false;  // differing sizes, they are not the same
    std::map<size_t, blitz::Array<double,2> >::const_iterator i, j;
    for(i = this->m_gamma.begin(), j = b.m_gamma.begin(); 
      i != this->m_gamma.end(); ++i, ++j)
    {
      if(i->first != j->first || blitz::any(i->second != j->second))
        return false;
    }
  }
  // 2. m_loglike_constterm
  {
    if(this->m_loglike_constterm.size() != b.m_loglike_constterm.size())
      return false;  // differing sizes, they are not the same
    std::map<size_t, double>::const_iterator i, j;
    for(i = this->m_loglike_constterm.begin(), j = b.m_loglike_constterm.begin(); 
      i != this->m_loglike_constterm.end(); ++i, ++j)
    {
      if(i->first != j->first || i->second != j->second)
        return false;
    }
  }

  return true;
}

bool bob::machine::PLDAMachine::operator!=
    (const bob::machine::PLDAMachine& b) const 
{
  return !(this->operator==(b));
}


void bob::machine::PLDAMachine::load(bob::io::HDF5File& config) 
{
  //reads all data directly into the member variables
  m_n_samples = config.read<uint64_t>("n_samples");
  m_nh_sum_xit_beta_xi = config.read<double>("nh_sum_xit_beta_xi");
  m_weighted_sum.reference(config.readArray<double,1>("weighted_sum"));
  m_loglikelihood = config.read<double>("loglikelihood");
  // gamma and log like constant term (a-dependent terms)
  if(config.contains("a_indices"))
  {
    blitz::Array<uint32_t, 1> a_indices;
    a_indices.reference(config.readArray<uint32_t,1>("a_indices"));
    for(int i=0; i<a_indices.extent(0); ++i)
    {
      std::string str1 = "gamma_" + boost::lexical_cast<std::string>(a_indices(i));
      m_gamma[a_indices(i)].reference(config.readArray<double,2>(str1));
      std::string str2 = "loglikeconstterm_" + boost::lexical_cast<std::string>(a_indices(i));
      m_loglike_constterm[a_indices(i)] = config.read<double>(str2);
    }
  }
}

void bob::machine::PLDAMachine::save(bob::io::HDF5File& config) const 
{
  config.set("n_samples", m_n_samples);
  config.set("nh_sum_xit_beta_xi", m_nh_sum_xit_beta_xi);
  config.setArray("weighted_sum", m_weighted_sum);
  config.set("loglikelihood", m_loglikelihood);
  // Gamma
  if(m_gamma.size() > 0)
  {
    blitz::Array<uint32_t, 1> a_indices(m_gamma.size());
    int i = 0;
    for(std::map<size_t,blitz::Array<double,2> >::const_iterator 
        it=m_gamma.begin(); it!=m_gamma.end(); ++it)
    {
      a_indices(i) = it->first;
      std::string str1 = "gamma_" + boost::lexical_cast<std::string>(it->first);
      config.setArray(str1, it->second);
      std::string str2 = "loglikeconstterm_" + boost::lexical_cast<std::string>(it->first);
      double v = m_loglike_constterm.find(it->first)->second;
      config.set(str2, v);
      ++i;
    }
    config.setArray("a_indices", a_indices);
  }
}

void bob::machine::PLDAMachine::resize(const size_t dim_d, const size_t dim_f, 
  const size_t dim_g)
{
  m_weighted_sum.resizeAndPreserve(dim_f);
  m_gamma.clear();
  m_loglike_constterm.clear();
  m_cache_d_1.resize(dim_d);
  m_cache_d_2.resize(dim_d);
  m_cache_nf_1.resize(dim_f);
  m_cache_nf_2.resize(dim_f);
}

void bob::machine::PLDAMachine::setPLDABase(const boost::shared_ptr<bob::machine::PLDABaseMachine> plda_base) 
{
  m_plda_base = plda_base; 
  m_weighted_sum.resizeAndPreserve(getDimF());
  m_cache_d_1.resize(getDimD());
  m_cache_d_2.resize(getDimD());
  m_cache_nf_1.resize(getDimF());
  m_cache_nf_2.resize(getDimF());
//  resize(getDimD(), getDimF(), getDimG());
}


void bob::machine::PLDAMachine::setWeightedSum(const blitz::Array<double,1>& ws) {
  if(ws.extent(0) != m_weighted_sum.extent(0)) { //checks dimension
    throw bob::machine::NInputsMismatch(ws.extent(0), m_weighted_sum.extent(0));
  }
  m_weighted_sum.reference(bob::core::array::ccopy(ws));
}

const blitz::Array<double,2>& bob::machine::PLDAMachine::getGamma(const size_t a)
{
  // Checks in both base machine and this machine
  if(m_plda_base->hasGamma(a)) return m_plda_base->getGamma(a);
  // TODO: specialized exception
  else if(!hasGamma(a)) throw bob::machine::Exception();
  return m_gamma[a];
}

const blitz::Array<double,2>& bob::machine::PLDAMachine::getAddGamma(const size_t a)
{
  if(m_plda_base->hasGamma(a)) return m_plda_base->getGamma(a);
  else if(hasGamma(a)) return m_gamma[a];
  // else computes it and adds it to this machine
  blitz::Array<double,2> gamma_a(getDimF(),getDimF());
  m_gamma[a].reference(gamma_a);
  m_plda_base->computeGamma(a, gamma_a);
  return m_gamma[a];
}

double bob::machine::PLDAMachine::getLogLikeConstTerm(const size_t a)
{
  // Checks in both base machine and this machine
  if(m_plda_base->hasLogLikeConstTerm(a)) return m_plda_base->getLogLikeConstTerm(a);
  // TODO: specialized exception
  else if(!hasLogLikeConstTerm(a)) throw bob::machine::Exception();
  return m_loglike_constterm[a];
}

double bob::machine::PLDAMachine::getAddLogLikeConstTerm(const size_t a)
{
  if(m_plda_base->hasLogLikeConstTerm(a)) return m_plda_base->getLogLikeConstTerm(a);
  else if(hasLogLikeConstTerm(a)) return m_loglike_constterm[a];
  // else computes it and adds it to this machine
  m_loglike_constterm[a] = 
        m_plda_base->computeLogLikeConstTerm(a, getAddGamma(a));
  return m_loglike_constterm[a];
}

double bob::machine::PLDAMachine::computeLogLikelihood(const blitz::Array<double,1>& sample,
  bool enrol)
{
  // Check dimensionality
  bob::core::array::assertSameDimensionLength(sample.extent(0), getDimD());

  int n_samples = 1 + (enrol?m_n_samples:0);
  // 1/2/ Constant term of the log likelihood:
  //      1/ First term of the likelihood: -Nsamples*D/2*log(2*PI)
  //      2/ Second term of the likelihood: -1/2*log(det(SIGMA+A.A^T))
  //        Efficient way: -Nsamples/2*log(det(sigma))-Nsamples/2*log(det(I+G^T.sigma^-1.G))
  //       -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  double log_likelihood = getAddLogLikeConstTerm(static_cast<size_t>(n_samples));

  // 3/ Third term of the likelihood: -1/2*X^T*(SIGMA+A.A^T)^-1*X
  //    Efficient way: -1/2*sum_i(xi^T.sigma^-1.xi - xi^T.sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1.xi
  //      -1/2*sumWeighted^T*(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1).F)^-1*sumWeighted
  //      where sumWeighted = sum_i(F^T*(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1)*xi)
  const blitz::Array<double,2>& beta = getPLDABase()->getBeta();
  const blitz::Array<double,2>& Ft_beta = getPLDABase()->getFtBeta();
  const blitz::Array<double,1>& mu = getPLDABase()->getMu();
  double terma = (enrol?m_nh_sum_xit_beta_xi:0.);
  // sumWeighted
  if(enrol && m_n_samples > 0) m_cache_nf_1 = m_weighted_sum;
  else m_cache_nf_1 = 0;
  
  // terma += -1 / 2. * (xi^t*beta*xi)
  m_cache_d_1 = sample - mu;
  bob::math::prod(beta, m_cache_d_1, m_cache_d_2);
  terma += -1 / 2. * (blitz::sum(m_cache_d_1*m_cache_d_2));
    
  // sumWeighted
  bob::math::prod(Ft_beta, m_cache_d_1, m_cache_nf_2);
  m_cache_nf_1 += m_cache_nf_2;

  blitz::Array<double,2> gamma_a = getAddGamma(n_samples);
  bob::math::prod(gamma_a, m_cache_nf_1, m_cache_nf_2);
  double termb = 1 / 2. * (blitz::sum(m_cache_nf_1*m_cache_nf_2));
  
  log_likelihood += terma + termb;
  return log_likelihood; 
}

double bob::machine::PLDAMachine::computeLogLikelihood(const blitz::Array<double,2>& samples,
  bool enrol)
{
  // Check dimensionality
  bob::core::array::assertSameDimensionLength(samples.extent(1), getDimD());

  int n_samples = samples.extent(0) + (enrol?m_n_samples:0);
  // 1/2/ Constant term of the log likelihood:
  //      1/ First term of the likelihood: -Nsamples*D/2*log(2*PI)
  //      2/ Second term of the likelihood: -1/2*log(det(SIGMA+A.A^T))
  //        Efficient way: -Nsamples/2*log(det(sigma))-Nsamples/2*log(det(I+G^T.sigma^-1.G))
  //       -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  double log_likelihood = getAddLogLikeConstTerm(static_cast<size_t>(n_samples));

  // 3/ Third term of the likelihood: -1/2*X^T*(SIGMA+A.A^T)^-1*X
  //    Efficient way: -1/2*sum_i(xi^T.sigma^-1.xi - xi^T.sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1.xi
  //      -1/2*sumWeighted^T*(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1).F)^-1*sumWeighted
  //      where sumWeighted = sum_i(F^T*(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1)*xi)
  const blitz::Array<double,2>& beta = getPLDABase()->getBeta();
  const blitz::Array<double,2>& Ft_beta = getPLDABase()->getFtBeta();
  const blitz::Array<double,1>& mu = getPLDABase()->getMu();
  double terma = (enrol?m_nh_sum_xit_beta_xi:0.);
  // sumWeighted
  if(enrol && m_n_samples > 0) m_cache_nf_1 = m_weighted_sum;
  else m_cache_nf_1 = 0;
  for(int k=0; k<samples.extent(0); ++k) 
  {
    blitz::Array<double,1> samp = samples(k,blitz::Range::all());
    m_cache_d_1 = samp - mu;
    // terma += -1 / 2. * (xi^t*beta*xi)
    bob::math::prod(beta, m_cache_d_1, m_cache_d_2);
    terma += -1 / 2. * (blitz::sum(m_cache_d_1*m_cache_d_2));
    
    // sumWeighted
    bob::math::prod(Ft_beta, m_cache_d_1, m_cache_nf_2);
    m_cache_nf_1 += m_cache_nf_2;
  }

  blitz::Array<double,2> gamma_a = getAddGamma(n_samples);
  bob::math::prod(gamma_a, m_cache_nf_1, m_cache_nf_2);
  double termb = 1 / 2. * (blitz::sum(m_cache_nf_1*m_cache_nf_2));

  log_likelihood += terma + termb;
  return log_likelihood;
}

void bob::machine::PLDAMachine::forward(const blitz::Array<double,1>& sample, double& score)
{
  // Computes the log likelihood ratio
  score = computeLogLikelihood(sample, true) - // match
          (computeLogLikelihood(sample, false) + m_loglikelihood); // no match
}

void bob::machine::PLDAMachine::forward(const blitz::Array<double,2>& samples, double& score)
{
  // Computes the log likelihood ratio
  score = computeLogLikelihood(samples, true) - // match
          (computeLogLikelihood(samples, false) + m_loglikelihood); // no match
}

void bob::machine::PLDAMachine::clearMaps()
{
  m_gamma.clear();
  m_loglike_constterm.clear();
}

