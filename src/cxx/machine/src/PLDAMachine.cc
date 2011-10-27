/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue 11 oct 2011
 *
 * @brief Machines that implements the PLDA model
 */

#include "core/array_assert.h"
#include "core/array_copy.h"
#include "machine/Exception.h"
#include "machine/PLDAMachine.h"
#include "math/linear.h"
#include "math/lu_det.h"

#include <cmath>
#include <boost/lexical_cast.hpp>
#include <string>

#include "core/logging.h"

namespace mach = Torch::machine;
namespace tca = Torch::core::array;

mach::PLDABaseMachine::PLDABaseMachine():
  m_F(0,0), m_G(0,0), m_sigma(0), m_mu(0), 
  m_isigma(0), m_alpha(0,0), m_beta(0,0), m_gamma(),
  m_Ft_beta(0,0), m_Gt_isigma(0,0),
  m_logdet_alpha(0), m_logdet_sigma(0), m_loglike_constterm(),
  m_cache_d_ng_1(0,0), m_cache_nf_nf_1(0,0), m_cache_ng_ng_1(0,0)
{
}

mach::PLDABaseMachine::PLDABaseMachine(const size_t d, const size_t nf,
    const size_t ng):
  m_F(d,nf), m_G(d,ng), m_sigma(d), m_mu(d), 
  m_isigma(d), m_alpha(ng,ng), m_beta(d,d), m_gamma(),
  m_Ft_beta(nf,d), m_Gt_isigma(ng,d),
  m_logdet_alpha(0), m_logdet_sigma(0), m_loglike_constterm(),
  m_cache_d_ng_1(d,ng), m_cache_nf_nf_1(nf,nf), m_cache_ng_ng_1(ng,ng)
{
  initFGSigma();
}


mach::PLDABaseMachine::PLDABaseMachine(const mach::PLDABaseMachine& other):
  m_F(tca::ccopy(other.m_F)), 
  m_G(tca::ccopy(other.m_G)), 
  m_sigma(tca::ccopy(other.m_sigma)), 
  m_mu(tca::ccopy(other.m_mu)),
  m_isigma(tca::ccopy(other.m_isigma)), 
  m_alpha(tca::ccopy(other.m_alpha)),
  m_beta(tca::ccopy(other.m_beta)),
  m_gamma(),
  m_Ft_beta(tca::ccopy(other.m_Ft_beta)),
  m_Gt_isigma(tca::ccopy(other.m_Gt_isigma)), 
  m_logdet_alpha(other.m_logdet_alpha),
  m_logdet_sigma(other.m_logdet_sigma),
  m_loglike_constterm(other.m_loglike_constterm),
  m_cache_d_ng_1(tca::ccopy(other.m_cache_d_ng_1)), 
  m_cache_nf_nf_1(tca::ccopy(other.m_cache_nf_nf_1)), 
  m_cache_ng_ng_1(tca::ccopy(other.m_cache_ng_ng_1))
{
  tca::ccopy(other.m_gamma, m_gamma);
}

mach::PLDABaseMachine::PLDABaseMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::PLDABaseMachine::~PLDABaseMachine() {
}

mach::PLDABaseMachine& mach::PLDABaseMachine::operator=
    (const mach::PLDABaseMachine& other) 
{
  m_F.reference(tca::ccopy(other.m_F));
  m_G.reference(tca::ccopy(other.m_G));
  m_sigma.reference(tca::ccopy(other.m_sigma));
  m_mu.reference(tca::ccopy(other.m_mu));
  m_isigma.reference(tca::ccopy(other.m_isigma));
  m_alpha.reference(tca::ccopy(other.m_alpha));
  m_beta.reference(tca::ccopy(other.m_beta));
  tca::ccopy(other.m_gamma, m_gamma);
  m_Ft_beta.reference(tca::ccopy(other.m_Ft_beta));
  m_Gt_isigma.reference(tca::ccopy(other.m_Gt_isigma));
  m_logdet_alpha = other.m_logdet_alpha;
  m_logdet_sigma = other.m_logdet_sigma;
  m_loglike_constterm = other.m_loglike_constterm;
  m_cache_d_ng_1.reference(tca::ccopy(other.m_cache_d_ng_1));
  m_cache_nf_nf_1.reference(tca::ccopy(other.m_cache_nf_nf_1));
  m_cache_ng_ng_1.reference(tca::ccopy(other.m_cache_ng_ng_1));
  return *this;
}

void mach::PLDABaseMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_F.reference(config.readArray<double,2>("F"));
  m_G.reference(config.readArray<double,2>("G"));
  int d = m_F.extent(0);
  int nf = m_F.extent(1);
  int ng = m_G.extent(1);
  m_sigma.reference(config.readArray<double,1>("sigma"));
  m_mu.reference(config.readArray<double,1>("mu"));
  m_isigma.resize(d);
  precomputeISigma();
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
      config.read(str2, m_loglike_constterm[a_indices(i)]);
    }
  }
  m_Ft_beta.reference(config.readArray<double,2>("Ft_beta"));
  m_Gt_isigma.reference(config.readArray<double,2>("Gt_isigma"));
  config.read("logdet_alpha", m_logdet_alpha);
  config.read("logdet_sigma", m_logdet_sigma);
  m_cache_d_ng_1.resize(d,ng);
  m_cache_nf_nf_1.resize(nf,nf);
  m_cache_ng_ng_1.resize(ng,ng);
}

void mach::PLDABaseMachine::save(Torch::io::HDF5File& config) const {
  config.setArray("F", m_F);
  config.setArray("G", m_G);
  config.setArray("sigma", m_sigma);
  config.setArray("mu", m_mu);
  config.setArray("alpha", m_alpha);
  config.setArray("beta", m_beta);
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
  config.setArray("Ft_beta", m_Ft_beta);
  config.setArray("Gt_isigma", m_Gt_isigma);
  config.set("logdet_alpha", m_logdet_alpha);
  config.set("logdet_sigma", m_logdet_sigma);
}

void mach::PLDABaseMachine::resize(const size_t d, const size_t nf, 
    const size_t ng) 
{
  m_F.resize(d,nf);
  m_G.resize(d,ng);
  m_sigma.resize(d);
  m_mu.resize(d);
  m_alpha.resize(ng,ng);
  m_beta.resize(d,d);
  m_gamma.clear();
  m_isigma.resize(d);
  m_cache_d_ng_1.resize(d,ng);
  m_cache_nf_nf_1.resize(nf,nf);
  m_cache_ng_ng_1.resize(ng,ng);
  m_loglike_constterm.clear();
  initFGSigma();
}

void mach::PLDABaseMachine::setF(const blitz::Array<double,2>& F) {
  tca::assertSameShape(F, m_F);
  m_F.reference(Torch::core::array::ccopy(F));
  // Precomputes useful matrices
  precompute();
}

void mach::PLDABaseMachine::setG(const blitz::Array<double,2>& G) {
  tca::assertSameShape(G, m_G);
  m_G.reference(Torch::core::array::ccopy(G));
  // Precomputes useful matrices and values
  precompute();
  precomputeLogDetAlpha();
}

void mach::PLDABaseMachine::setSigma(const blitz::Array<double,1>& sigma) {
  tca::assertSameShape(sigma, m_sigma);
  m_sigma.reference(Torch::core::array::ccopy(sigma));
  // Precomputes useful matrices and values
  precompute();
  precomputeLogDetAlpha();
  precomputeLogDetSigma();
}

void mach::PLDABaseMachine::setMu(const blitz::Array<double,1>& mu) {
  tca::assertSameShape(mu, m_mu);
  m_mu.reference(Torch::core::array::ccopy(mu));
}

blitz::Array<double,2>& mach::PLDABaseMachine::getGamma(const size_t a)
{
  // TODO: specialized exception
  if(!hasGamma(a)) throw Torch::machine::Exception();
  return m_gamma[a];
}

blitz::Array<double,2>& mach::PLDABaseMachine::getAddGamma(const size_t a)
{
  if(!hasGamma(a)) precomputeGamma(a);
  return m_gamma[a];
}

void mach::PLDABaseMachine::initFGSigma() {
  // To avoid problems related to precomputation
  Torch::math::eye(m_F);
  Torch::math::eye(m_G);
  m_sigma = 1.;
}

void mach::PLDABaseMachine::precompute() {
  precomputeISigma();
  precomputeGtISigma();
  precomputeAlpha();
  precomputeBeta();
  m_gamma.clear();
  precomputeFtBeta();
  m_loglike_constterm.clear();
}

void mach::PLDABaseMachine::precomputeLogLike() {
  precomputeLogDetAlpha();
  precomputeLogDetSigma();
}

void mach::PLDABaseMachine::precomputeISigma() {
  // Updates inverse of sigma
  // TODO: check division by zero
  m_isigma = 1. / m_sigma;
}

void mach::PLDABaseMachine::precomputeGtISigma() {
  // m_Gt_isigma = G^T.sigma^-1
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> Gt = m_G.transpose(1,0);
  m_Gt_isigma = Gt(i,j) * m_isigma(j);
}

void mach::PLDABaseMachine::precomputeAlpha() {
  // alpha = (Id + G^T.sigma^-1.G)^-1

  // m_cache_ng_ng_1 = G^T.sigma^-1.G
  Torch::math::prod(m_Gt_isigma, m_G, m_cache_ng_ng_1);
  // m_cache_ng_ng_1 = Id + G^T.sigma^-1.G
  for(int i=0; i<m_cache_ng_ng_1.extent(0); ++i) m_cache_ng_ng_1(i,i) += 1;
  // m_alpha = (Id + G^T.sigma^-1.G)^-1
  Torch::math::inv(m_cache_ng_ng_1, m_alpha);
}

void mach::PLDABaseMachine::precomputeBeta() {
  // beta = (sigma + G.G^T)^-1
  // BUT, there is a more efficient computation (Woodbury identity):
  // beta = sigma^-1 - sigma^-1.G.(Id + G^T.sigma^-1.G)^-1.G^T.sigma^-1
  // beta =  sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1
  
  blitz::Array<double,2> GtISigmaT = m_Gt_isigma.transpose(1,0);
  // m_cache_d_ng_1 = sigma^-1.G.alpha
  Torch::math::prod(GtISigmaT, m_alpha, m_cache_d_ng_1);
  // m_beta = -sigma^-1.G.alpha.G^T.sigma^-1
  Torch::math::prod(m_cache_d_ng_1, m_Gt_isigma, m_beta);
  m_beta = -m_beta;
  // m_beta = sigma^-1 - sigma^-1.G.alpha.G^T.sigma^-1
  for(int i=0; i<m_beta.extent(0); ++i) m_beta(i,i) += m_isigma(i);
}

void mach::PLDABaseMachine::precomputeGamma(const size_t a)
{
  
  blitz::Array<double,2> gamma_a(getDimF(),getDimF());
  m_gamma[a].reference(gamma_a);
  computeGamma(a, gamma_a);
}

void mach::PLDABaseMachine::precomputeFtBeta() {
  // m_Ft_beta = F^T.beta = F^T.(sigma + G.G^T)^-1 
  blitz::Array<double,2> Ft = m_F.transpose(1,0);
  Torch::math::prod(Ft, m_beta, m_Ft_beta);
}

void mach::PLDABaseMachine::computeGamma(const size_t a, 
  blitz::Array<double,2> res)
{
  // gamma = (Id + a.F^T.beta.F)^-1

  // Checks destination size
  tca::assertSameShape(res, m_cache_nf_nf_1);
  // m_cache_nf_nf_1 = F^T.beta.F
  Torch::math::prod(m_Ft_beta, m_F, m_cache_nf_nf_1);
   // m_cache_nf_nf_1 = a.F^T.beta.F
  m_cache_nf_nf_1 *= static_cast<double>(a);
  // m_cache_nf_nf_1 = Id + a.F^T.beta.F
  for(int i=0; i<m_cache_nf_nf_1.extent(0); ++i) m_cache_nf_nf_1(i,i) += 1;

  // res = (Id + a.F^T.beta.F)^-1
  Torch::math::inv(m_cache_nf_nf_1, res);
}

void mach::PLDABaseMachine::precomputeLogDetAlpha()
{
  m_logdet_alpha = log(fabs(Torch::math::det(m_alpha)));
}

void mach::PLDABaseMachine::precomputeLogDetSigma()
{
  m_logdet_sigma = blitz::sum(blitz::log(m_sigma));
}

double mach::PLDABaseMachine::computeLogLikeConstTerm(const size_t a,
  const blitz::Array<double,2>& gamma_a)
{
  // loglike_constterm[a] = a/2 * 
  //  ( -D*log(2*pi) -log|sigma| +log|alpha| +log|gamma_a|)
  double logdet_gamma_a = log(fabs(Torch::math::det(gamma_a)));
  double ah = static_cast<double>(a)/2.;
  double res = ( -ah*static_cast<double>(getDimD())*log(2*M_PI) - 
      ah*m_logdet_sigma + ah*m_logdet_alpha + logdet_gamma_a/2.);
  return res;
}

double mach::PLDABaseMachine::computeLogLikeConstTerm(const size_t a)
{
  blitz::Array<double,2>& gamma_a = getAddGamma(a);
  return computeLogLikeConstTerm(a, gamma_a);
}

void mach::PLDABaseMachine::precomputeLogLikeConstTerm(const size_t a)
{
  double val = computeLogLikeConstTerm(a); 
  m_loglike_constterm[a] = val;
}

double mach::PLDABaseMachine::getLogLikeConstTerm(const size_t a)
{
  // TODO: specialized exception
  if(!hasLogLikeConstTerm(a)) throw Torch::machine::Exception();
  return m_loglike_constterm[a];
}

double mach::PLDABaseMachine::getAddLogLikeConstTerm(const size_t a)
{
  if(!hasLogLikeConstTerm(a)) precomputeLogLikeConstTerm(a);
  return m_loglike_constterm[a];
}




mach::PLDAMachine::PLDAMachine():
  m_plda_base(boost::shared_ptr<Torch::machine::PLDABaseMachine>()),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(0), 
  m_loglikelihood(0), m_gamma(), m_loglike_constterm(),
  m_cache_d_1(0), m_cache_d_2(0), m_cache_nf_1(0), m_cache_nf_2(0)
{
}

mach::PLDAMachine::PLDAMachine(const boost::shared_ptr<Torch::machine::PLDABaseMachine> plda_base): 
  m_plda_base(plda_base),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(plda_base->getDimF()),
  m_loglikelihood(0), m_gamma(), m_loglike_constterm(),
  m_cache_d_1(plda_base->getDimD()), m_cache_d_2(plda_base->getDimD()),
  m_cache_nf_1(plda_base->getDimF()), m_cache_nf_2(plda_base->getDimF())
{
}


mach::PLDAMachine::PLDAMachine(const mach::PLDAMachine& other):
  m_plda_base(other.m_plda_base),
  m_n_samples(other.m_n_samples), 
  m_nh_sum_xit_beta_xi(other.m_nh_sum_xit_beta_xi), 
  m_weighted_sum(tca::ccopy(other.m_weighted_sum)),
  m_loglikelihood(other.m_loglikelihood), m_gamma(), 
  m_loglike_constterm(other.m_loglike_constterm),
  m_cache_d_1(tca::ccopy(other.m_cache_d_1)),
  m_cache_d_2(tca::ccopy(other.m_cache_d_2)),
  m_cache_nf_1(tca::ccopy(other.m_cache_nf_1)),
  m_cache_nf_2(tca::ccopy(other.m_cache_nf_2))
{
  tca::ccopy(other.m_gamma, m_gamma);
}

mach::PLDAMachine::PLDAMachine(Torch::io::HDF5File& config):
  m_plda_base(boost::shared_ptr<Torch::machine::PLDABaseMachine>())
{
  load(config);
}

mach::PLDAMachine::~PLDAMachine() {
}

mach::PLDAMachine& mach::PLDAMachine::operator=
(const mach::PLDAMachine& other) {
  m_plda_base = other.m_plda_base;
  m_n_samples = other.m_n_samples;
  m_nh_sum_xit_beta_xi = other.m_nh_sum_xit_beta_xi; 
  m_weighted_sum.reference(tca::ccopy(other.m_weighted_sum));
  m_loglikelihood = other.m_loglikelihood;
  tca::ccopy(other.m_gamma, m_gamma);
  m_loglike_constterm = other.m_loglike_constterm;
  m_cache_d_1.reference(tca::ccopy(other.m_cache_d_1));
  m_cache_d_2.reference(tca::ccopy(other.m_cache_d_2));
  m_cache_nf_1.reference(tca::ccopy(other.m_cache_nf_1));
  m_cache_nf_2.reference(tca::ccopy(other.m_cache_nf_2));
  return *this;
}

void mach::PLDAMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  config.read("n_samples", m_n_samples);
  config.read("nh_sum_xit_beta_xi", m_nh_sum_xit_beta_xi);
  m_weighted_sum.reference(config.readArray<double,1>("weighted_sum"));
  config.read("loglikelihood", m_loglikelihood);
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
      config.read(str2, m_loglike_constterm[a_indices(i)]);
    }
  }
}

void mach::PLDAMachine::save(Torch::io::HDF5File& config) const {
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

void mach::PLDAMachine::resize(const size_t d, const size_t nf, 
  const size_t ng)
{
  m_weighted_sum.resizeAndPreserve(nf);
  m_gamma.clear();
  m_loglike_constterm.clear();
  m_cache_d_1.resize(d);
  m_cache_d_2.resize(d);
  m_cache_nf_1.resize(nf);
  m_cache_nf_2.resize(nf);
}

void mach::PLDAMachine::setPLDABase(const boost::shared_ptr<Torch::machine::PLDABaseMachine> plda_base) {
  m_plda_base = plda_base; 
  m_weighted_sum.resizeAndPreserve(getDimF());
  m_cache_d_1.resize(getDimD());
  m_cache_d_2.resize(getDimD());
  m_cache_nf_1.resize(getDimF());
  m_cache_nf_2.resize(getDimF());
//  resize(getDimD(), getDimF(), getDimG());
}


void mach::PLDAMachine::setWeightedSum(const blitz::Array<double,1>& ws) {
  if(ws.extent(0) != m_weighted_sum.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(ws.extent(0), m_weighted_sum.extent(0));
  }
  m_weighted_sum.reference(tca::ccopy(ws));
}

blitz::Array<double,2>& mach::PLDAMachine::getGamma(const size_t a)
{
  // Checks in both base machine and this machine
  if(m_plda_base->hasGamma(a)) return m_plda_base->getGamma(a);
  // TODO: specialized exception
  else if(!hasGamma(a)) throw Torch::machine::Exception();
  return m_gamma[a];
}

blitz::Array<double,2>& mach::PLDAMachine::getAddGamma(const size_t a)
{
  if(m_plda_base->hasGamma(a)) return m_plda_base->getGamma(a);
  else if(hasGamma(a)) return m_gamma[a];
  // else computes it and adds it to this machine
  blitz::Array<double,2> gamma_a(getDimF(),getDimF());
  m_gamma[a].reference(gamma_a);
  m_plda_base->computeGamma(a, gamma_a);
  return m_gamma[a];
}

double mach::PLDAMachine::getLogLikeConstTerm(const size_t a)
{
  // Checks in both base machine and this machine
  if(m_plda_base->hasLogLikeConstTerm(a)) return m_plda_base->getLogLikeConstTerm(a);
  // TODO: specialized exception
  else if(!hasLogLikeConstTerm(a)) throw Torch::machine::Exception();
  return m_loglike_constterm[a];
}

double mach::PLDAMachine::getAddLogLikeConstTerm(const size_t a)
{
  if(m_plda_base->hasLogLikeConstTerm(a)) return m_plda_base->getLogLikeConstTerm(a);
  else if(hasLogLikeConstTerm(a)) return m_loglike_constterm[a];
  // else computes it and adds it to this machine
  m_loglike_constterm[a] = 
        m_plda_base->computeLogLikeConstTerm(a, getAddGamma(a));
  return m_loglike_constterm[a];
}

double mach::PLDAMachine::computeLikelihood(const blitz::Array<double,1>& sample,
  bool enrol)
{
   int n_samples = 1 + (enrol?m_n_samples:0);
  // 1/2/ Constant term of the log likelihood:
  //      1/ First term of the likelihood: -Nsamples*D/2*log(2*PI)
  //      2/ Second term of the likelihood: -1/2*log(det(SIGMA+A.A^T))
  //        Efficient way: -Nsamples/2*log(det(sigma))-Nsamples/2*log(det(I+G^T.sigma^-1.G))
  //       -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  // TODO: check samples dimensionality
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
  Torch::math::prod(beta, m_cache_d_1, m_cache_d_2);
  terma += -1 / 2. * (blitz::sum(m_cache_d_1*m_cache_d_2));
    
  // sumWeighted
  Torch::math::prod(Ft_beta, m_cache_d_1, m_cache_nf_2);
  m_cache_nf_1 += m_cache_nf_2;

  blitz::Array<double,2> gamma_a = getAddGamma(n_samples);
  Torch::math::prod(gamma_a, m_cache_nf_1, m_cache_nf_2);
  double termb = 1 / 2. * (blitz::sum(m_cache_nf_1*m_cache_nf_2));
  
  log_likelihood += terma + termb;
  return log_likelihood; 
}

double mach::PLDAMachine::computeLikelihood(const blitz::Array<double,2>& samples,
  bool enrol)
{
  int n_samples = samples.extent(0) + (enrol?m_n_samples:0);
  // 1/2/ Constant term of the log likelihood:
  //      1/ First term of the likelihood: -Nsamples*D/2*log(2*PI)
  //      2/ Second term of the likelihood: -1/2*log(det(SIGMA+A.A^T))
  //        Efficient way: -Nsamples/2*log(det(sigma))-Nsamples/2*log(det(I+G^T.sigma^-1.G))
  //       -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  // TODO: check samples dimensionality
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
    Torch::math::prod(beta, m_cache_d_1, m_cache_d_2);
    terma += -1 / 2. * (blitz::sum(m_cache_d_1*m_cache_d_2));
    
    // sumWeighted
    Torch::math::prod(Ft_beta, m_cache_d_1, m_cache_nf_2);
    m_cache_nf_1 += m_cache_nf_2;
  }

  blitz::Array<double,2> gamma_a = getAddGamma(n_samples);
  Torch::math::prod(gamma_a, m_cache_nf_1, m_cache_nf_2);
  double termb = 1 / 2. * (blitz::sum(m_cache_nf_1*m_cache_nf_2));
  
  log_likelihood += terma + termb;
  return log_likelihood;
}

void mach::PLDAMachine::forward(const blitz::Array<double,1>& sample, double& score)
{
  // Computes the log likelihood ratio
  score = computeLikelihood(sample, true) - // match
          (computeLikelihood(sample, false) + m_loglikelihood); // no match
}

void mach::PLDAMachine::forward(const blitz::Array<double,2>& samples, double& score)
{
  // Computes the log likelihood ratio
  score = computeLikelihood(samples, true) - // match
          (computeLikelihood(samples, false) + m_loglikelihood); // no match
}
