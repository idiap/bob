/**
 * @file trainer/cxx/PLDATrainer.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Probabilistic Linear Discriminant Analysis
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

#include <algorithm>
#include <boost/random.hpp>
#include <vector>
#include <limits>

#include "bob/trainer/PLDATrainer.h"
#include "bob/core/array_copy.h"
#include "bob/math/linear.h"
#include "bob/math/inv.h"
#include "bob/math/svd.h"
#include "bob/trainer/Exception.h"


namespace tca = bob::core::array;
namespace io = bob::io;
namespace mach = bob::machine;
namespace math = bob::math;
namespace train = bob::trainer;

train::PLDABaseTrainer::PLDABaseTrainer(int nf, int ng, 
    double convergence_threshold, int max_iterations, bool compute_likelihood):
  EMTrainer<mach::PLDABaseMachine, std::vector<blitz::Array<double,2> > >
    (convergence_threshold, max_iterations, compute_likelihood), 
  m_nf(nf), m_ng(ng), m_limited_memory(false), m_S(0,0),
  m_z_first_order(0), m_sum_z_second_order(0,0),
  m_seed(-1), 
  m_initF_method(0), m_initF_ratio(1.),
  m_initG_method(0), m_initG_ratio(1.),
  m_initSigma_method(0), m_initSigma_ratio(1.),
  m_n_samples_per_id(0), m_n_samples_in_training(), m_B(0,0),
  m_Ft_isigma_G(0,0), m_eta(0,0), m_zeta(), m_iota(),
  m_cache_nf_1(0), m_cache_nf_2(0), m_cache_ng_1(0),
  m_cache_D_1(0), m_cache_D_2(0), 
  m_cache_nfng_nfng(0,0), m_cache_D_nfng_1(0,0), m_cache_D_nfng_2(0,0)
{
}

train::PLDABaseTrainer::PLDABaseTrainer(const train::PLDABaseTrainer& other):
  EMTrainer<mach::PLDABaseMachine, std::vector<blitz::Array<double,2> > >
    (other.m_convergence_threshold, other.m_max_iterations, 
     other.m_compute_likelihood),
  m_nf(other.m_nf), m_ng(other.m_ng), m_limited_memory(other.m_limited_memory),
  m_S(tca::ccopy(other.m_S)),
  m_z_first_order(),
  m_sum_z_second_order(tca::ccopy(other.m_sum_z_second_order)),
  m_seed(other.m_seed),
  m_initF_method(other.m_initF_method), m_initF_ratio(other.m_initF_ratio),
  m_initG_method(other.m_initG_method), m_initG_ratio(other.m_initG_ratio),
  m_initSigma_method(other.m_initSigma_method), m_initSigma_ratio(other.m_initSigma_ratio),
  m_n_samples_per_id(other.m_n_samples_per_id),
  m_n_samples_in_training(other.m_n_samples_in_training), 
  m_B(tca::ccopy(other.m_B)), 
  m_Ft_isigma_G(tca::ccopy(other.m_Ft_isigma_G)), 
  m_eta(tca::ccopy(other.m_eta)), 
  m_cache_nf_1(tca::ccopy(other.m_cache_nf_1)),
  m_cache_nf_2(tca::ccopy(other.m_cache_nf_2)),
  m_cache_ng_1(tca::ccopy(other.m_cache_ng_1)),
  m_cache_D_1(tca::ccopy(other.m_cache_D_1)),
  m_cache_D_2(tca::ccopy(other.m_cache_D_2)),
  m_cache_nfng_nfng(tca::ccopy(other.m_cache_nfng_nfng)),
  m_cache_D_nfng_1(tca::ccopy(other.m_cache_D_nfng_1)),
  m_cache_D_nfng_2(tca::ccopy(other.m_cache_D_nfng_2))
{
  tca::ccopy(other.m_z_first_order, m_z_first_order);
  tca::ccopy(other.m_zeta, m_zeta);
  tca::ccopy(other.m_iota, m_iota);
}

train::PLDABaseTrainer::~PLDABaseTrainer() {}

train::PLDABaseTrainer& train::PLDABaseTrainer::operator=
(const train::PLDABaseTrainer& other) 
{
  m_convergence_threshold = other.m_convergence_threshold;
  m_max_iterations = other.m_max_iterations;
  m_compute_likelihood = other.m_compute_likelihood;
  m_nf = other.m_nf;
  m_ng = other.m_ng;
  m_limited_memory = other.m_limited_memory;
  m_S = tca::ccopy(other.m_S);
  tca::ccopy(other.m_z_first_order, m_z_first_order);
  m_sum_z_second_order = tca::ccopy(other.m_sum_z_second_order);
  m_seed = other.m_seed;
  m_initF_method = other.m_initF_method;
  m_initF_ratio = other.m_initF_ratio;
  m_initG_method = other.m_initG_method;
  m_initG_ratio = other.m_initG_ratio;
  m_initSigma_method = other.m_initSigma_method;
  m_initSigma_ratio = other.m_initSigma_ratio;
  m_n_samples_per_id = other.m_n_samples_per_id;
  m_n_samples_in_training = other.m_n_samples_in_training;
  m_B = tca::ccopy(other.m_B); 
  m_Ft_isigma_G = tca::ccopy(other.m_Ft_isigma_G); 
  m_eta = tca::ccopy(other.m_eta); 
  tca::ccopy(other.m_iota, m_iota);
  m_cache_nf_1 = tca::ccopy(other.m_cache_nf_1);
  m_cache_nf_2 = tca::ccopy(other.m_cache_nf_2);
  m_cache_ng_1 = tca::ccopy(other.m_cache_ng_1);
  m_cache_D_1 = tca::ccopy(other.m_cache_D_1);
  m_cache_D_2 = tca::ccopy(other.m_cache_D_2);
  m_cache_nfng_nfng = tca::ccopy(other.m_cache_nfng_nfng);
  m_cache_D_nfng_1 = tca::ccopy(other.m_cache_D_nfng_1);
  m_cache_D_nfng_2 = tca::ccopy(other.m_cache_D_nfng_2);
  return *this;
}

void train::PLDABaseTrainer::initialization(mach::PLDABaseMachine& machine,
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Checks training data
  checkTrainingData(v_ar);

  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].extent(1);
  // Resizes the PLDABaseMachine
  machine.resize(n_features, m_nf, m_ng);

  // Reinitializes array members
  initMembers(v_ar);

  // Computes the mean and the covariance if required
  computeMeanVariance(machine, v_ar);

  // Initialization (e.g. using scatter)
  initFGSigma(machine, v_ar);
}

void train::PLDABaseTrainer::finalization(mach::PLDABaseMachine& machine,
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Precomputes constant parts of the log likelihood and (gamma_a)
  precomputeLogLike(machine, v_ar);
  // Adds the case 1 sample if not already done (always used for scoring)
  machine.getAddGamma(1);
  machine.getAddLogLikeConstTerm(1);
}

void train::PLDABaseTrainer::checkTrainingData(const std::vector<blitz::Array<double,2> >& v_ar)
{
  // Checks that the vector of Arraysets is not empty
  if(v_ar.size() == 0)
    throw bob::trainer::EmptyTrainingSet();

  // Gets dimension (first Arrayset)
  int n_features = v_ar[0].extent(1);
  // Checks dimension consistency
  for(size_t i=0; i<v_ar.size(); ++i) {
    if(v_ar[i].extent(1) != n_features)
      throw bob::trainer::WrongNumberOfFeatures(v_ar[i].extent(1), n_features, i);
  } 
}

void train::PLDABaseTrainer::initMembers(const std::vector<blitz::Array<double,2> >& v_ar)
{
  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].extent(1); // dimensionality of the data
  size_t n_identities = v_ar.size();

  m_S.resize(n_features,n_features);
  m_sum_z_second_order.resize(m_nf+m_ng, m_nf+m_ng);

  // Loops over the identities
  for(size_t i=0; i<n_identities; ++i) 
  {
    // Number of training samples for this identity
    size_t n_i = v_ar[i].extent(0); 
    // m_z_first_order
    blitz::Array<double,2> z_i(n_i, m_nf+m_ng);
    m_z_first_order.push_back(z_i);

    // m_n_samples_per_id
    m_n_samples_per_id.push_back(n_i);
    
    // Maps dependent on the number of samples per identity
    std::map<size_t,bool>::iterator it;
    it = m_n_samples_in_training.find(n_i);
    if(it == m_n_samples_in_training.end())
    {
      // Indicates if there are identities with n_i training samples and if
      // corresponding matrices are up to date.
      m_n_samples_in_training[n_i] = false;
      // Allocates arrays for identities with n_i training samples
      m_zeta[n_i].reference(blitz::Array<double,2>(m_ng, m_ng));
      m_iota[n_i].reference(blitz::Array<double,2>(m_nf, m_ng));
    }
  }

  m_B.resize(n_features, m_nf+m_ng);
  m_Ft_isigma_G.resize(m_nf,m_ng);
  m_eta.resize(m_nf,m_ng);

  // Cache
  m_cache_nf_1.resize(m_nf);
  m_cache_nf_2.resize(m_nf);
  m_cache_ng_1.resize(m_ng);
  m_cache_D_1.resize(n_features);
  m_cache_D_2.resize(n_features);
  m_cache_nfng_nfng.resize(m_nf+m_ng,m_nf+m_ng);
  m_cache_D_nfng_1.resize(n_features,m_nf+m_ng);
  m_cache_D_nfng_2.resize(n_features,m_nf+m_ng);
}

void train::PLDABaseTrainer::computeMeanVariance(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,1>& mu = machine.updateMu();
  blitz::Range all = blitz::Range::all();
  // TODO: Uncomment variance computation if required
  /*  if(m_compute_likelihood) 
  {
    // loads all the data in a single shot - required for scatter
    blitz::Array<double,2> data(n_features, n_samples);
    for (size_t i=0; i<n_samples; ++i)
      data(all,i) = ar(i,all);
    // Mean and scatter computation
    math::scatter(data, m_S, mu);
    // divides scatter by N-1
    m_S /= static_cast<double>(n_samples-1);
  }
  else */
  {
    // Computes the mean and updates mu
    mu = 0.;
    size_t n_samples = 0;
    for(size_t j=0; j<v_ar.size(); ++j) {
      n_samples += v_ar[j].extent(0);
      for (int i=0; i<v_ar[j].extent(0); ++i)
        mu += v_ar[j](i,all);
    }
    mu /= static_cast<double>(n_samples);
  }
}

void train::PLDABaseTrainer::initFGSigma(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Initializes F, G and sigma
  initF(machine, v_ar);
  initG(machine, v_ar);
  initSigma(machine, v_ar);

  // Precomputes values using new F, G and sigma
  machine.precompute();
}

void train::PLDABaseTrainer::initF(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,2>& F = machine.updateF();
  blitz::Range a = blitz::Range::all();

  // 1: between-class scatter
  if(m_initF_method==1) 
  {
    // a/ Computes between-class scatter matrix
    blitz::firstIndex bi;
    blitz::secondIndex bj;
    blitz::Array<double,2> S(machine.getDimD(), v_ar.size());
    S = 0.;
    m_cache_D_1 = 0.;
    for(size_t i=0; i<v_ar.size(); ++i)
    {
      blitz::Array<double,1> Si = S(blitz::Range::all(),i);
      Si = 0.;
      for(int j=0; j<v_ar[i].extent(0); ++j)
      {
        // Si += x_ij
        Si += v_ar[i](j,a);
      }
      // Si = mean of the samples class i
      Si /= static_cast<double>(v_ar[i].extent(0));
      m_cache_D_1 += Si;
    }
    m_cache_D_1 /= static_cast<double>(v_ar.size());

    // b/ Removes the mean
    S = S(bi,bj) - m_cache_D_1(bi);

    // c/ SVD of the between-class scatter matrix
    int n_singular = std::min(machine.getDimD(),v_ar.size());
    blitz::Array<double,2> U(machine.getDimD(), n_singular);
    blitz::Array<double,1> sigma(n_singular);
    bob::math::svd(S, U, sigma);

    // d/ Updates F
    blitz::Array<double,2> Uslice = U(a, blitz::Range(0,machine.getDimF()-1));
    blitz::Array<double,1> sigma_slice = sigma(blitz::Range(0,machine.getDimF()-1));
    sigma_slice = blitz::sqrt(sigma_slice);
    F = Uslice(bi,bj) / sigma_slice(bj);
  }
  // otherwise: random initialization
  else {
    // Initializes the random number generator
    boost::mt19937 rng;
    if(m_seed != -1)
      rng.seed((uint32_t)m_seed);
    boost::normal_distribution<> range_n;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > 
      die_n(rng, range_n);
      
    // F initialization
    for(int j=0; j<F.extent(0); ++j)
      for(int i=0; i<F.extent(1); ++i)
        F(j,i) = die_n() * m_initF_ratio;
  }
}

void train::PLDABaseTrainer::initG(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,2>& G = machine.updateG();
  blitz::Range a = blitz::Range::all();

  // 1: within-class scatter
  if(m_initG_method==1) 
  {
    // a/ Computes within-class scatter matrix
    blitz::firstIndex bi;
    blitz::secondIndex bj;
    size_t Nsamples=0;
    for(size_t i=0; i<v_ar.size(); ++i)
      Nsamples += v_ar[i].extent(0);
        
    blitz::Array<double,2> S(machine.getDimD(), Nsamples);
    S = 0.;
    m_cache_D_1 = 0.;
    int counter = 0;
    for(size_t i=0; i<v_ar.size(); ++i)
    {
      // Computes the mean of the samples class i
      m_cache_D_2 = 0.;
      for(int j=0; j<v_ar[i].extent(0); ++j)
      {
        // m_cache_D_2 += x_ij
        m_cache_D_2 += v_ar[i](j,a);
      }
      // m_cache_D_2 = mean of the samples class i
      m_cache_D_2 /= static_cast<double>(v_ar[i].extent(0));

      // Generates the scatter
      for(int j=0; j<v_ar[i].extent(0); ++j)
      {
        blitz::Array<double,1> Si = S(a, counter);
        // Si = x_ij - mean_i
        Si = v_ar[i](j,a) - m_cache_D_2;
        // mean of the within class
        m_cache_D_1 += Si;
        ++counter;
      }
    }
    m_cache_D_1 /= static_cast<double>(Nsamples);

    // b/ Removes the mean
    S = S(bi,bj) - m_cache_D_1(bi);

    // c/ SVD of the between-class scatter matrix
    blitz::Array<double,2> U(machine.getDimD(), std::min(machine.getDimD(),Nsamples));
    blitz::Array<double,1> sigma( std::min(machine.getDimD(),Nsamples) );
    bob::math::svd(S, U, sigma);

    // d/ Updates G
    blitz::Array<double,2> Uslice = U(blitz::Range::all(), blitz::Range(0,machine.getDimG()-1));
    blitz::Array<double,1> sigma_slice = sigma(blitz::Range(0,machine.getDimG()-1));
    sigma_slice = blitz::sqrt(sigma_slice);
    G = Uslice(bi,bj) / sigma_slice(bj);
  }
  // otherwise: random initialization
  else {
    // Initializes the random number generator
    boost::mt19937 rng;
    if(m_seed != -1)
      rng.seed((uint32_t)m_seed);
    boost::normal_distribution<> range_n;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > 
      die_n(rng, range_n);

    // G initialization
    for(int j=0; j<G.extent(0); ++j)
      for(int i=0; i<G.extent(1); ++i)
        G(j,i) = die_n() * m_initG_ratio;
  }
}

void train::PLDABaseTrainer::initSigma(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,1>& sigma = machine.updateSigma();
  blitz::Range a = blitz::Range::all();
  double eps = std::numeric_limits<double>::epsilon(); // Sigma should be invertible...

  // 1: percentage of the variance of G
  if(m_initSigma_method==1) {
    blitz::Array<double,2>& G = machine.updateG();
    blitz::secondIndex bj;
    m_cache_D_1 = blitz::mean(G, bj);
    // Updates sigma
    sigma = blitz::fabs(m_cache_D_1) * m_initSigma_ratio + eps;
  }
  // 2: constant value
  else if(m_initSigma_method==2) {
    sigma = m_initSigma_ratio;
  }
  // 3: percentage of the variance of the data
  else if(m_initSigma_method==3) {
    // a/ Computes the global mean
    //    m_cache_D_1 = 1/N sum_i x_i
    m_cache_D_1 = 0.;
    size_t Ns = 0;
    for(size_t i=0; i<v_ar.size(); ++i)
    {
      for(int j=0; j<v_ar[i].extent(0); ++j) 
        m_cache_D_1 += v_ar[i](j,a);
      Ns += v_ar[i].extent(0);
    }
    m_cache_D_1 /= static_cast<double>(Ns);
  
    // b/ Computes the variance:
    m_cache_D_2 = 0.;
    for(size_t i=0; i<v_ar.size(); ++i)
      for(int j=0; j<v_ar[i].extent(0); ++j) 
        m_cache_D_2 += blitz::pow2(v_ar[i](j,a) - m_cache_D_1);
    sigma = m_initSigma_ratio * m_cache_D_2 / static_cast<double>(Ns-1);
  }
  // otherwise: random initialization
  else {
    // Initializes the random number generator
    boost::mt19937 rng;
    if(m_seed != -1)
      rng.seed((uint32_t)m_seed);
    boost::normal_distribution<> range_n;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > 
      die_n(rng, range_n);

    // sigma initialization
    for(int j=0; j<sigma.extent(0); ++j)
      sigma(j) = fabs(die_n()) * m_initSigma_ratio + eps;
  }

}

void train::PLDABaseTrainer::initRandomFGSigma(mach::PLDABaseMachine& machine)
{
  // Initializes the random number generator
  boost::mt19937 rng;
  if(m_seed != -1)
    rng.seed((uint32_t)m_seed);
  boost::normal_distribution<> range_n;
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > 
    die_n(rng, range_n);
    
  double ratio = 1.; // TODO: check if a ratio is required
  // F initialization
  blitz::Array<double,2>& F = machine.updateF();
  for(int j=0; j<F.extent(0); ++j)
    for(int i=0; i<F.extent(1); ++i)
      F(j,i) = die_n() * ratio;
  // G initialization
  blitz::Array<double,2>& G = machine.updateG();
  for(int j=0; j<G.extent(0); ++j)
    for(int i=0; i<G.extent(1); ++i)
      G(j,i) = die_n() * ratio;
  // sigma2 initialization
  blitz::Array<double,1>& sigma = machine.updateSigma();
  double eps = std::numeric_limits<double>::epsilon(); // Sigma should be invertible...
  for(int j=0; j<sigma.extent(0); ++j)
    sigma(j) = fabs(die_n()) * ratio + eps;

  // Precompute values
  machine.precompute();
}


void train::PLDABaseTrainer::eStep(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar)
{  
  // Precomputes useful variables using current estimates of F,G, and sigma
  precomputeFromFGSigma(machine);
  // Gets the mean mu from the machine
  const blitz::Array<double,1>& mu = machine.getMu();
  const blitz::Array<double,2>& alpha = machine.getAlpha();
  const blitz::Array<double,2>& F = machine.getF();
  const blitz::Array<double,2>& FtBeta = machine.getFtBeta();
  const blitz::Array<double,2>& GtISigma = machine.getGtISigma();
  blitz::Range a = blitz::Range::all();

  // blitz indices
  blitz::firstIndex bi;
  blitz::secondIndex bj;
  // Initializes sum of z second order statistics to 0
  m_sum_z_second_order = 0.;
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // Computes expectation of z_ij = [h_i w_ij]
    // 1/a/ Computes expectation of h_i
    // Loop over the samples
    m_cache_nf_1 = 0.;
    for(int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_cache_D_1 = x_sj-mu
      m_cache_D_1 = v_ar[i](j,a) - mu;

      // m_cache_nf_2 = F^T.beta.(x_sj-mu)
      bob::math::prod(FtBeta, m_cache_D_1, m_cache_nf_2);
      // m_cache_nf_1 = sum_j F^T.beta.(x_sj-mu)
      m_cache_nf_1 += m_cache_nf_2;
    }
    const blitz::Array<double,2>& gamma_a = machine.getAddGamma(v_ar[i].extent(0));
    blitz::Range r_hi(0, m_nf-1);
    // m_cache_nf_2 = E(h_i) = gamma_A  sum_j F^T.beta.(x_sj-mu)
    bob::math::prod(gamma_a, m_cache_nf_1, m_cache_nf_2);

    // 1/b/ Precomputes: m_cache_D_2 = F.E{h_i}
    bob::math::prod(F, m_cache_nf_2, m_cache_D_2);

    // 2/ First and second order statistics of z
    // Precomputed values 
    blitz::Array<double,2>& zeta_a = m_zeta[v_ar[i].extent(0)];
    blitz::Array<double,2>& iota_a = m_iota[v_ar[i].extent(0)];
    blitz::Array<double,2> iotat_a = iota_a.transpose(1,0);

    // Extracts statistics of z_ij = [h_i w_ij] from y_i = [h_i w_i1 ... w_iJ]
    blitz::Range r1(0, m_nf-1);
    blitz::Range r2(m_nf, m_nf+m_ng-1);
    for(int j=0; j<v_ar[i].extent(0); ++j)
    {
      // 1/ First order statistics of z
      blitz::Array<double,1> z_first_order_ij_1 = m_z_first_order[i](j,r1);
      z_first_order_ij_1 = m_cache_nf_2; // E{h_i}
      // m_cache_D_1 = x_sj - mu - F.E{h_i}
      m_cache_D_1 = v_ar[i](j,a) - mu - m_cache_D_2;
      // m_cache_ng_1 = G^T.sigma^-1.(x_sj-mu-fhi)
      bob::math::prod(GtISigma, m_cache_D_1, m_cache_ng_1);
      // z_first_order_ij_2 = (Id+G^T.sigma^-1.G)^-1.G^T.sigma^-1.(x_sj-mu) = E{w_ij}
      blitz::Array<double,1> z_first_order_ij_2 = m_z_first_order[i](j,r2);
      bob::math::prod(alpha, m_cache_ng_1, z_first_order_ij_2); 

      // 2/ Second order statistics of z
      blitz::Array<double,2> z_so_11 = m_sum_z_second_order(r1,r1);
      z_so_11 += gamma_a + z_first_order_ij_1(bi) * z_first_order_ij_1(bj);
      blitz::Array<double,2> z_so_12 = m_sum_z_second_order(r1,r2);
      z_so_12 += iota_a + z_first_order_ij_1(bi) * z_first_order_ij_2(bj);
      blitz::Array<double,2> z_so_21 = m_sum_z_second_order(r2,r1);
      z_so_21 += iotat_a + z_first_order_ij_2(bi) * z_first_order_ij_1(bj);
      blitz::Array<double,2> z_so_22 = m_sum_z_second_order(r2,r2);
      z_so_22 += zeta_a + z_first_order_ij_2(bi) * z_first_order_ij_2(bj);
    }
  }
}

void train::PLDABaseTrainer::precomputeFromFGSigma(mach::PLDABaseMachine& machine)
{
  // non const because of transpose() (compability with old blitz versions)  
  blitz::Array<double,2>& F = machine.updateF();
  blitz::Array<double,2> Ft = F.transpose(1,0);
  blitz::Array<double,2>& Gt_isigma = machine.updateGtISigma();
  blitz::Array<double,2> Gt_isigma_t = Gt_isigma.transpose(1,0);
  const blitz::Array<double,2>& alpha = machine.getAlpha();

  // blitz indices
  blitz::firstIndex i;
  blitz::secondIndex j;

  // Precomputes F, G and sigma-based expressions
  bob::math::prod(Ft, Gt_isigma_t, m_Ft_isigma_G);
  bob::math::prod(m_Ft_isigma_G, alpha, m_eta); 
  blitz::Array<double,2> etat = m_eta.transpose(1,0);

  // Reinitializes all the zeta_a and iota_a
  std::map<size_t,bool>::iterator it;
  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
    it->second = false;

  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
  {
    size_t n_i = it->first;
    // Precomputes zeta and iota for identities with q_i training samples,
    // if not already done
    if(!it->second)
    {
      blitz::Array<double,2>& gamma_a = machine.getAddGamma(n_i);
      blitz::Array<double,2>& zeta_a = m_zeta[n_i];
      blitz::Array<double,2>& iota_a = m_iota[n_i];
      bob::math::prod(gamma_a, m_eta, iota_a);
      bob::math::prod(etat, iota_a, zeta_a);
      zeta_a += alpha;
      iota_a = - iota_a;
      // Now up to date
      it->second = true;
    }
  }
}

void train::PLDABaseTrainer::precomputeLogLike(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Precomputes the log determinant of alpha and sigma
  machine.precomputeLogLike();

  // Precomputes the log likelihood constant term
  std::map<size_t,bool>::iterator it;
  for(it=m_n_samples_in_training.begin(); it!=m_n_samples_in_training.end(); 
      ++it)
  {
    // Precomputes the log likelihood constant term for identities with q_i 
    // training samples, if not already done
    machine.getAddLogLikeConstTerm(it->first);
  }
}


void train::PLDABaseTrainer::mStep(mach::PLDABaseMachine& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // TODO: 0/ Add mean update rule as an option?

  // 1/ New estimate of B = {F G}
  updateFG(machine, v_ar);

  // 2/ New estimate of Sigma
  updateSigma(machine, v_ar);

  // 3/ Precomputes new values after updating F, G and sigma
  machine.precompute();
  // Precomputes useful variables using current estimates of F,G, and sigma
  precomputeFromFGSigma(machine);
}

void train::PLDABaseTrainer::updateFG(mach::PLDABaseMachine& machine,
  const std::vector<blitz::Array<double,2> >& v_ar)
{
  /// Computes the B matrix (B = [F G])
  /// B = (sum_ij (x_ij-mu).E{z_i}^T).(sum_ij E{z_i.z_i^T})^-1

  // 1/ Computes the numerator (sum_ij (x_ij-mu).E{z_i}^T)
  // Gets the mean mu from the machine
  const blitz::Array<double,1>& mu = machine.getMu();
  blitz::Range a = blitz::Range::all();
  m_cache_D_nfng_2 = 0.;
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for(int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_cache_D_1 = x_sj-mu
      m_cache_D_1 = v_ar[i](j,a) - mu;
      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_z_first_order[i](j, a);
      // m_cache_D_nfng_1 = (x_sj-mu).E{z_ij}^T
      bob::math::prod(m_cache_D_1, z_first_order_ij, m_cache_D_nfng_1);
      m_cache_D_nfng_2 += m_cache_D_nfng_1;
    }
  }

  // 2/ Computes the denominator inv(sum_ij E{z_i.z_i^T})
  bob::math::inv(m_sum_z_second_order, m_cache_nfng_nfng);

  // 3/ Computes numerator / denominator
  bob::math::prod(m_cache_D_nfng_2, m_cache_nfng_nfng, m_B);

  // 4/ Updates the machine 
  // TODO: Use B as cache in the trainer, and only sets F and G when calling
  //       finalization()
  blitz::Array<double, 2>& F = machine.updateF();
  blitz::Array<double, 2>& G = machine.updateG();
  F = m_B(blitz::Range::all(), blitz::Range(0,m_nf-1));
  G = m_B(blitz::Range::all(), blitz::Range(m_nf,m_nf+m_ng-1));
}

void train::PLDABaseTrainer::updateSigma(mach::PLDABaseMachine& machine,
  const std::vector<blitz::Array<double,2> >& v_ar)
{
  /// Computes the Sigma matrix
  /// Sigma = 1/IJ sum_ij Diag{(x_ij-mu).(x_ij-mu)^T - B.E{z_i}.(x_ij-mu)^T}

  // Gets the mean mu and the matrix sigma from the machine
  blitz::Array<double,1>& sigma = machine.updateSigma();
  const blitz::Array<double,1>& mu = machine.getMu();
  blitz::Range a = blitz::Range::all();

  sigma = 0.;
  size_t n_IJ=0; /// counts the number of samples
  for(size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for(int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_cache_D_1 = x_ij-mu
      m_cache_D_1 = v_ar[i](j,a) - mu;
      // sigma += Diag{(x_ij-mu).(x_ij-mu)^T}
      sigma += blitz::pow2(m_cache_D_1);

      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_z_first_order[i](j,a);
      // m_cache_D_2 = B.E{z_ij}
      bob::math::prod(m_B, z_first_order_ij, m_cache_D_2);
      // sigma -= Diag{B.E{z_ij}.(x_ij-mu)
      sigma -= (m_cache_D_1 * m_cache_D_2);
      ++n_IJ;
    }
  }
  // Normalizes by the number of samples
  sigma /= static_cast<double>(n_IJ);
}

double train::PLDABaseTrainer::computeLikelihood(mach::PLDABaseMachine& machine)
{
  double llh = 0.;
  // TODO: implement log likelihood computation
  return llh;
}


train::PLDATrainer::PLDATrainer(mach::PLDAMachine& plda_machine): 
  m_plda_machine(plda_machine),
  m_cache_D_1(plda_machine.getDimD()),
  m_cache_D_2(plda_machine.getDimD()),
  m_cache_nf_1(plda_machine.getDimF())
{
}

train::PLDATrainer::PLDATrainer(const train::PLDATrainer& other):
  m_plda_machine(other.m_plda_machine),
  m_cache_D_1(tca::ccopy(other.m_cache_D_1)),
  m_cache_D_2(tca::ccopy(other.m_cache_D_2)),
  m_cache_nf_1(tca::ccopy(other.m_cache_nf_1))
{
}

train::PLDATrainer::~PLDATrainer() {
}

train::PLDATrainer& train::PLDATrainer::operator=
(const train::PLDATrainer& other) 
{
  m_plda_machine = other.m_plda_machine;
  m_cache_D_1.reference(tca::ccopy(other.m_cache_D_1));
  m_cache_D_2.reference(tca::ccopy(other.m_cache_D_2));
  m_cache_nf_1.reference(tca::ccopy(other.m_cache_nf_1));
  return *this;
}


void train::PLDATrainer::enrol(const blitz::Array<double,2>& ar)
{
  // Checks Arrayset using the check function from the PLDABaseTrainer
  // Gets dimension (first Arrayset)
  int n_features = ar.extent(1);
  int n_samples = ar.extent(0);
    
  // TODO: Do a useful comparison against the dimensionality from the base 
  // trainer/machine
  if(ar.extent(1) != n_features)
    throw bob::trainer::WrongNumberOfFeatures(ar.extent(1), n_features, 0);

  // Useful values from the base machine
  blitz::Array<double, 1>& weighted_sum = m_plda_machine.updateWeightedSum();
  const blitz::Array<double, 1>& mu = m_plda_machine.getPLDABase()->getMu();
  const blitz::Array<double, 2>& beta = m_plda_machine.getPLDABase()->getBeta();
  const blitz::Array<double, 2>& FtBeta = m_plda_machine.getPLDABase()->getFtBeta();

  // Resizes the PLDA machine
  m_plda_machine.resize(m_plda_machine.getDimD(), m_plda_machine.getDimF(), 
    m_plda_machine.getDimG());

  // Updates the PLDA machine
  m_plda_machine.setNSamples(n_samples);
  double terma = 0.;
  weighted_sum = 0.;
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<n_samples; ++i) {
    m_cache_D_1 =  ar(i,a) - mu;
    // a/ weighted sum
    bob::math::prod(FtBeta, m_cache_D_1, m_cache_nf_1);
    weighted_sum += m_cache_nf_1;
    // b/ first xi dependent term of the log likelihood
    bob::math::prod(beta, m_cache_D_1, m_cache_D_2);
    terma += -1 / 2. * blitz::sum(m_cache_D_1 * m_cache_D_2);
  }
  m_plda_machine.setWSumXitBetaXi(terma);

  // Adds the precomputed values for the cases N and N+1 if not already 
  // in the base machine (used by the forward function, 1 already added)
  m_plda_machine.getAddGamma(n_samples);
  m_plda_machine.getAddLogLikeConstTerm(n_samples);
  m_plda_machine.getAddGamma(n_samples+1);
  m_plda_machine.getAddLogLikeConstTerm(n_samples+1);
  m_plda_machine.setLogLikelihood(m_plda_machine.computeLikelihood(
                                    blitz::Array<double,2>(0,0),true));
}
