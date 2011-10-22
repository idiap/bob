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
  m_cache_d_ng_1(0,0), m_cache_nf_nf_1(0,0), m_cache_ng_ng_1(0,0)
{
}

mach::PLDABaseMachine::PLDABaseMachine(const size_t d, const size_t nf,
    const size_t ng):
  m_F(d,nf), m_G(d,ng), m_sigma(d), m_mu(d), 
  m_isigma(d), m_alpha(ng,ng), m_beta(d,d), m_gamma(),
  m_Ft_beta(nf,d), m_Gt_isigma(ng,d),
  m_cache_d_ng_1(d,ng), m_cache_nf_nf_1(nf,nf), m_cache_ng_ng_1(ng,ng)
{
}


mach::PLDABaseMachine::PLDABaseMachine(const mach::PLDABaseMachine& other):
  m_F(Torch::core::array::ccopy(other.m_F)), 
  m_G(Torch::core::array::ccopy(other.m_G)), 
  m_sigma(Torch::core::array::ccopy(other.m_sigma)), 
  m_mu(Torch::core::array::ccopy(other.m_mu)),
  m_isigma(Torch::core::array::ccopy(other.m_isigma)), 
  m_alpha(Torch::core::array::ccopy(other.m_alpha)),
  m_beta(Torch::core::array::ccopy(other.m_beta)),
  m_gamma(),
  m_Ft_beta(Torch::core::array::ccopy(other.m_Ft_beta)),
  m_Gt_isigma(Torch::core::array::ccopy(other.m_Gt_isigma)), 
  m_cache_d_ng_1(Torch::core::array::ccopy(other.m_cache_d_ng_1)), 
  m_cache_nf_nf_1(Torch::core::array::ccopy(other.m_cache_nf_nf_1)), 
  m_cache_ng_ng_1(Torch::core::array::ccopy(other.m_cache_ng_ng_1))
{
  Torch::core::array::ccopy(other.m_gamma, m_gamma);
}

mach::PLDABaseMachine::PLDABaseMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::PLDABaseMachine::~PLDABaseMachine() {
}

mach::PLDABaseMachine& mach::PLDABaseMachine::operator=
    (const mach::PLDABaseMachine& other) 
{
  m_F.reference(Torch::core::array::ccopy(other.m_F));
  m_G.reference(Torch::core::array::ccopy(other.m_G));
  m_sigma.reference(Torch::core::array::ccopy(other.m_sigma));
  m_mu.reference(Torch::core::array::ccopy(other.m_mu));
  m_isigma.reference(Torch::core::array::ccopy(other.m_isigma));
  m_alpha.reference(Torch::core::array::ccopy(other.m_alpha));
  m_beta.reference(Torch::core::array::ccopy(other.m_beta));
  m_gamma.clear();
  m_Ft_beta.reference(Torch::core::array::ccopy(other.m_Ft_beta));
  m_Gt_isigma.reference(Torch::core::array::ccopy(other.m_Gt_isigma));
  Torch::core::array::ccopy(other.m_gamma, m_gamma);
  m_cache_d_ng_1.reference(Torch::core::array::ccopy(other.m_cache_d_ng_1));
  m_cache_nf_nf_1.reference(Torch::core::array::ccopy(other.m_cache_nf_nf_1));
  m_cache_ng_ng_1.reference(Torch::core::array::ccopy(other.m_cache_ng_ng_1));
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
  m_alpha.reference(config.readArray<double,2>("alpha"));
  m_beta.reference(config.readArray<double,2>("beta"));
  // gamma
  blitz::Array<uint32_t, 1> gamma_indices;
  gamma_indices.reference(config.readArray<uint32_t,1>("gamma_indices"));
  for(int i=0; i<gamma_indices.extent(0); ++i)
  {
    std::string str = "gamma_" + boost::lexical_cast<std::string>(i);
    m_gamma[i].reference(config.readArray<double,2>(str));
  }
  m_Ft_beta.reference(config.readArray<double,2>("Ft_beta"));
  m_Gt_isigma.reference(config.readArray<double,2>("Gt_isigma"));
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
  blitz::Array<uint32_t, 1> gamma_indices(m_gamma.size());
  int i = 0;
  for(std::map<size_t,blitz::Array<double,2> >::const_iterator 
      it=m_gamma.begin(); it!=m_gamma.end(); ++it)
  {
    gamma_indices(i) = it->first;
    std::string str = "gamma_" + boost::lexical_cast<std::string>(it->first);
    config.setArray(str, it->second);
    ++i;
  }
  config.setArray("gamma_indices", gamma_indices);
  config.setArray("Ft_beta", m_Ft_beta);
  config.setArray("Gt_isigma", m_Gt_isigma);
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
  // Precomputes useful matrices
  precompute();
}

void mach::PLDABaseMachine::setSigma(const blitz::Array<double,1>& sigma) {
  tca::assertSameShape(sigma, m_sigma);
  m_sigma.reference(Torch::core::array::ccopy(sigma));
  // Precomputes useful matrices
  precompute();
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

void mach::PLDABaseMachine::precompute() {
  precomputeISigma();
  precomputeGtISigma();
  precomputeAlpha();
  precomputeBeta();
  m_gamma.clear();
  precomputeFtBeta();
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



mach::PLDAMachine::PLDAMachine():
  m_plda_base(boost::shared_ptr<Torch::machine::PLDABaseMachine>()),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(0)
{
}

mach::PLDAMachine::PLDAMachine(const boost::shared_ptr<Torch::machine::PLDABaseMachine> plda_base): 
  m_plda_base(plda_base),
  m_n_samples(0), m_nh_sum_xit_beta_xi(0), m_weighted_sum(plda_base->getDimF())
{
}


mach::PLDAMachine::PLDAMachine(const mach::PLDAMachine& other):
  m_plda_base(other.m_plda_base),
  m_n_samples(other.m_n_samples), 
  m_nh_sum_xit_beta_xi(other.m_nh_sum_xit_beta_xi), 
  m_weighted_sum(Torch::core::array::ccopy(other.m_weighted_sum))
{
}

mach::PLDAMachine::PLDAMachine(Torch::io::HDF5File& config) {
  load(config);
}

mach::PLDAMachine::~PLDAMachine() {
}

mach::PLDAMachine& mach::PLDAMachine::operator=
(const mach::PLDAMachine& other) {
  m_plda_base = other.m_plda_base;
  m_n_samples = other.m_n_samples;
  m_nh_sum_xit_beta_xi = other.m_nh_sum_xit_beta_xi; 
  m_weighted_sum.reference(Torch::core::array::ccopy(other.m_weighted_sum));
  return *this;
}

void mach::PLDAMachine::load(Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  config.read("n_samples", m_n_samples);
  config.read("nh_sum_xit_beta_xi", m_nh_sum_xit_beta_xi);
  config.readArray("weighted_sum", m_weighted_sum);
}

void mach::PLDAMachine::save(Torch::io::HDF5File& config) const {
  config.set("n_samples", m_n_samples);
  config.set("nh_sum_xit_beta_xi", m_nh_sum_xit_beta_xi);
  config.setArray("weighted_sum", m_weighted_sum);
}

void mach::PLDAMachine::resize(const size_t nf, const size_t ng) 
{
  m_weighted_sum.resize(nf);
}

void mach::PLDAMachine::setPLDABase(const boost::shared_ptr<Torch::machine::PLDABaseMachine> plda_base) {
  m_plda_base = plda_base; 
  m_weighted_sum.resize(plda_base->getDimF());
}


void mach::PLDAMachine::setWeightedSum(const blitz::Array<double,1>& ws) {
  if(ws.extent(0) != m_weighted_sum.extent(0)) { //checks dimension
    throw mach::NInputsMismatch(ws.extent(0), m_weighted_sum.extent(0));
  }
  m_weighted_sum.reference(Torch::core::array::ccopy(ws));
}

double mach::PLDAMachine::computeLikelihood(const blitz::Array<double,2>& samples)
{
  int n_samples = samples.extent(0) + m_n_samples;
  // 1/ first term of the likelihood: -Nsamples*D/2*log(2*PI)
  // TODO: value in cache
  // TODO: check samples dimensionality
  double log_likelihood = 0.;
  double log_term1 = - n_samples * static_cast<double>(getDimD()) / 2. * log(2*M_PI); 
  Torch::core::info << "-nsamples*D/2.*log(2*pi)= " << log_term1 << std::endl;
  log_likelihood += log_term1;

  // 2/ Second term of the likelihood: -1/2*log(det(SIGMA+A.A^T))
  //    Efficient way: -Nsamples/2*log(det(sigma))-Nsamples/2*log(det(I+G^T.sigma^-1.G))
  //       -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  // a/ -Nsamples/2*log(det(sigma))
  size_t nf = getDimF();
  size_t ng = getDimG();
  size_t d = getDimD();
  blitz::Array<double,2> gamma_a(nf,nf);
  getPLDABase()->computeGamma(n_samples,gamma_a);
  const blitz::Array<double,1>& sigma = getPLDABase()->getSigma();
  double log_term2a = - n_samples / 2. * log(blitz::product(sigma));
  Torch::core::info << "-nsamples/2.*log(det(sigma))= " << log_term2a << std::endl;
  // b/ -Nsamples/2*log(det(I+G^T.sigma^-1.G)) = Nsamples/2*log(det(I+G^T.sigma^-1.G)^-1)
  double log_term2b = + n_samples / 2. * log(Torch::math::det(getPLDABase()->getAlpha()));
  Torch::core::info << "-nsamples/2*log(det(I+G^T.sigma^-1.G))= " << log_term2b << std::endl;
  // c/ -1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))
  double log_term2c = 1 / 2. * log(Torch::math::det(gamma_a));
  Torch::core::info << "-1/2*log(det(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)*G^T*sigma^-1).F))= " << log_term2c << std::endl;
  log_likelihood += log_term2a + log_term2b + log_term2c;

  // 3/ Third term of the likelihood: -1/2*X^T*(SIGMA+A.A^T)^-1*X
  //    Efficient way: -1/2*sum_i(xi^T.sigma^-1.xi - xi^T.sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1.xi
  //      -1/2*sumWeighted^T*(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1).F)^-1*sumWeighted
  //      where sumWeighted = sum_i(F^T*(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1)*xi)
  const blitz::Array<double,2>& beta = getPLDABase()->getBeta();
  const blitz::Array<double,2>& Ft_beta = getPLDABase()->getFtBeta();
  // TODO: cache
  blitz::Array<double,1> beta_samp(ng);
  // sumWeighted
  blitz::Array<double,1> sumWeighted(nf);
  blitz::Array<double,1> tmp_nf(nf);
  double terma = m_nh_sum_xit_beta_xi;
  sumWeighted = m_weighted_sum;
  for(int k=0; k<samples.extent(0); ++k) 
  {
    blitz::Array<double,1> samp = samples(k,blitz::Range::all());
    // terma += -1 / 2. * (xi^t*beta*xi)
    Torch::math::prod(beta, samp, beta_samp);
    terma += -1 / 2. * (blitz::sum(samp*beta_samp));
    
    // sumWeighted
    Torch::math::prod(Ft_beta, samp, tmp_nf);
    sumWeighted += tmp_nf;
  }
  Torch::core::info << "-1/2*sum_i(xi^T.sigma^-1.xi - xi^T.sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1.xi)= " << terma << std::endl;

  Torch::core::info << "sumWeighted = " << sumWeighted << std::endl;
  Torch::math::prod(gamma_a, sumWeighted, tmp_nf);
  double termb = 1 / 2. * (blitz::sum(sumWeighted*tmp_nf));
  Torch::core::info << "-1/2*sumWeighted^T*(I+aF^T.(sigma^-1-sigma^-1*G*(I+G^T.sigma^-1.G)^-1*G^T*sigma^-1).F)^-1*sumWeighted= " << termb << std::endl;
  
  log_likelihood += terma + termb;
  return log_likelihood;
}


void mach::PLDAMachine::forward(const blitz::Array<double,1>& sample, double& score)
{
/*
  // Ux and GMMStats
  estimateX(gmm_stats);
  std::vector<const Torch::machine::GMMStats*> stats;
  stats.push_back(&m_cache_gmmstats);
  m_cache_Ux.resize(getDimCD());
  Torch::math::prod(m_plda_base->getU(), m_x, m_cache_Ux);
  std::vector<blitz::Array<double,1> > channelOffset;
  channelOffset.push_back(m_cache_Ux);

  // m + Vy + Dz
  m_cache_mVyDz.resize(getDimCD());
  Torch::math::prod(m_plda_base->getV(), m_y, m_cache_mVyDz);
  m_cache_mVyDz += m_plda_base->getD()*m_z + m_plda_base->getUbm()->getMeanSupervector();
  std::vector<blitz::Array<double,1> > models;
  models.push_back(m_cache_mVyDz);

  // Linear scoring
  blitz::Array<double,2> scores(1,1);
  mach::linearScoring(models, 
    m_plda_base->getUbm()->getMeanSupervector(), m_plda_base->getUbm()->getVarianceSupervector(),
    stats, channelOffset, true, scores);
  score = scores(0,0);
*/
}
