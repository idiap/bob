/**
 * @file trainer/cxx/PLDATrainer.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Probabilistic Linear Discriminant Analysis
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

#include <bob/trainer/PLDATrainer.h>
#include <bob/core/array_copy.h>
#include <bob/core/array_random.h>
#include <bob/math/linear.h>
#include <bob/math/inv.h>
#include <bob/math/svd.h>
#include <bob/trainer/Exception.h>
#include <algorithm>
#include <boost/random.hpp>
#include <vector>
#include <limits>

bob::trainer::PLDATrainer::PLDATrainer(const size_t max_iterations, 
    const bool use_sum_second_order):
  EMTrainer<bob::machine::PLDABase, std::vector<blitz::Array<double,2> > >
    (0.001, max_iterations, false), 
  m_dim_d(0), m_dim_f(0), m_dim_g(0),
  m_use_sum_second_order(use_sum_second_order), 
  m_initF_method(bob::trainer::PLDATrainer::RANDOM_F), m_initF_ratio(1.),
  m_initG_method(bob::trainer::PLDATrainer::RANDOM_G), m_initG_ratio(1.),
  m_initSigma_method(bob::trainer::PLDATrainer::RANDOM_SIGMA), 
  m_initSigma_ratio(1.),
  m_cache_S(0,0), 
  m_cache_z_first_order(0), m_cache_sum_z_second_order(0,0), m_cache_z_second_order(0),
  m_cache_n_samples_per_id(0), m_cache_n_samples_in_training(), m_cache_B(0,0),
  m_cache_Ft_isigma_G(0,0), m_cache_eta(0,0), m_cache_zeta(), m_cache_iota(),
  m_tmp_nf_1(0), m_tmp_nf_2(0), m_tmp_ng_1(0),
  m_tmp_D_1(0), m_tmp_D_2(0), 
  m_tmp_nfng_nfng(0,0), m_tmp_D_nfng_1(0,0), m_tmp_D_nfng_2(0,0)
{
}

bob::trainer::PLDATrainer::PLDATrainer(const bob::trainer::PLDATrainer& other):
  EMTrainer<bob::machine::PLDABase, std::vector<blitz::Array<double,2> > >
    (other.m_convergence_threshold, other.m_max_iterations, 
     other.m_compute_likelihood),
  m_dim_d(other.m_dim_d), m_dim_f(other.m_dim_f), m_dim_g(other.m_dim_g), 
  m_use_sum_second_order(other.m_use_sum_second_order),
  m_initF_method(other.m_initF_method), m_initF_ratio(other.m_initF_ratio),
  m_initG_method(other.m_initG_method), m_initG_ratio(other.m_initG_ratio),
  m_initSigma_method(other.m_initSigma_method), m_initSigma_ratio(other.m_initSigma_ratio),
  m_cache_S(bob::core::array::ccopy(other.m_cache_S)),
  m_cache_z_first_order(),
  m_cache_sum_z_second_order(bob::core::array::ccopy(other.m_cache_sum_z_second_order)),
  m_cache_z_second_order(),
  m_cache_n_samples_per_id(other.m_cache_n_samples_per_id),
  m_cache_n_samples_in_training(other.m_cache_n_samples_in_training), 
  m_cache_B(bob::core::array::ccopy(other.m_cache_B)), 
  m_cache_Ft_isigma_G(bob::core::array::ccopy(other.m_cache_Ft_isigma_G)), 
  m_cache_eta(bob::core::array::ccopy(other.m_cache_eta)) 
{
  bob::core::array::ccopy(other.m_cache_z_first_order, m_cache_z_first_order);
  bob::core::array::ccopy(other.m_cache_z_second_order, m_cache_z_second_order);
  bob::core::array::ccopy(other.m_cache_zeta, m_cache_zeta);
  bob::core::array::ccopy(other.m_cache_iota, m_cache_iota);
  // Resize working arrays
  resizeTmp();
}

bob::trainer::PLDATrainer::~PLDATrainer() {}

bob::trainer::PLDATrainer& bob::trainer::PLDATrainer::operator=
(const bob::trainer::PLDATrainer& other) 
{
  if(this != &other)
  {
    bob::trainer::EMTrainer<bob::machine::PLDABase, 
      std::vector<blitz::Array<double,2> > >::operator=(other);
    m_dim_d = other.m_dim_d;
    m_dim_f = other.m_dim_f;
    m_dim_g = other.m_dim_g;
    m_use_sum_second_order = other.m_use_sum_second_order;
    m_initF_method = other.m_initF_method;
    m_initF_ratio = other.m_initF_ratio;
    m_initG_method = other.m_initG_method;
    m_initG_ratio = other.m_initG_ratio;
    m_initSigma_method = other.m_initSigma_method;
    m_initSigma_ratio = other.m_initSigma_ratio;
    m_cache_S = bob::core::array::ccopy(other.m_cache_S);
    bob::core::array::ccopy(other.m_cache_z_first_order, m_cache_z_first_order);
    m_cache_sum_z_second_order = bob::core::array::ccopy(other.m_cache_sum_z_second_order);
    bob::core::array::ccopy(other.m_cache_z_second_order, m_cache_z_second_order);
    m_cache_n_samples_per_id = other.m_cache_n_samples_per_id;
    m_cache_n_samples_in_training = other.m_cache_n_samples_in_training;
    m_cache_B = bob::core::array::ccopy(other.m_cache_B); 
    m_cache_Ft_isigma_G = bob::core::array::ccopy(other.m_cache_Ft_isigma_G); 
    m_cache_eta = bob::core::array::ccopy(other.m_cache_eta); 
    bob::core::array::ccopy(other.m_cache_iota, m_cache_iota);
    // Resize working arrays
    resizeTmp();
  }
  return *this;
}

bool bob::trainer::PLDATrainer::operator==
  (const bob::trainer::PLDATrainer& other) const
{
  return bob::trainer::EMTrainer<bob::machine::PLDABase,
           std::vector<blitz::Array<double,2> > >::operator==(other) &&
         m_dim_d == other.m_dim_d &&
         m_dim_f == other.m_dim_f &&
         m_dim_g == other.m_dim_g &&
         m_initF_method == other.m_initF_method &&
         m_initF_ratio == other.m_initF_ratio &&
         m_initG_method == other.m_initG_method &&
         m_initG_ratio == other.m_initG_ratio &&
         m_initSigma_method == other.m_initSigma_method &&
         m_initSigma_ratio == other.m_initSigma_ratio &&
         bob::core::array::isEqual(m_cache_S, m_cache_S) &&
         bob::core::array::isEqual(m_cache_z_first_order, other.m_cache_z_first_order) &&
         bob::core::array::isEqual(m_cache_sum_z_second_order, other.m_cache_sum_z_second_order) &&
         bob::core::array::isEqual(m_cache_z_second_order, other.m_cache_z_second_order) &&
         m_cache_n_samples_per_id.size() == m_cache_n_samples_per_id.size() && 
         std::equal(m_cache_n_samples_per_id.begin(), m_cache_n_samples_per_id.end(), other.m_cache_n_samples_per_id.begin()) &&
         m_cache_n_samples_in_training.size() == m_cache_n_samples_in_training.size() && 
         std::equal(m_cache_n_samples_in_training.begin(), m_cache_n_samples_in_training.end(), other.m_cache_n_samples_in_training.begin()) &&
         bob::core::array::isEqual(m_cache_B, other.m_cache_B) &&
         bob::core::array::isEqual(m_cache_Ft_isigma_G, other.m_cache_Ft_isigma_G) &&
         bob::core::array::isEqual(m_cache_eta, other.m_cache_eta) &&
         bob::core::array::isEqual(m_cache_zeta, other.m_cache_zeta) &&
         bob::core::array::isEqual(m_cache_iota, other.m_cache_iota);
}

bool bob::trainer::PLDATrainer::operator!=
  (const bob::trainer::PLDATrainer &other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::PLDATrainer::is_similar_to
  (const bob::trainer::PLDATrainer &other, const double r_epsilon, 
   const double a_epsilon) const
{
  return bob::trainer::EMTrainer<bob::machine::PLDABase,
           std::vector<blitz::Array<double,2> > >::is_similar_to(other, r_epsilon, a_epsilon) &&
         m_dim_d == other.m_dim_d &&
         m_dim_f == other.m_dim_f &&
         m_dim_g == other.m_dim_g &&
         m_use_sum_second_order == other.m_use_sum_second_order &&
         m_initF_method == other.m_initF_method &&
         bob::core::isClose(m_initF_ratio, other.m_initF_ratio, r_epsilon, a_epsilon) &&
         m_initG_method == other.m_initG_method &&
         bob::core::isClose(m_initG_ratio, other.m_initG_ratio, r_epsilon, a_epsilon) &&
         m_initSigma_method == other.m_initSigma_method &&
         bob::core::isClose(m_initSigma_ratio, other.m_initSigma_ratio, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_S, m_cache_S, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_z_first_order, other.m_cache_z_first_order, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_sum_z_second_order, other.m_cache_sum_z_second_order, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_z_second_order, other.m_cache_z_second_order, r_epsilon, a_epsilon) &&
         m_cache_n_samples_per_id.size() == m_cache_n_samples_per_id.size() && 
         std::equal(m_cache_n_samples_per_id.begin(), m_cache_n_samples_per_id.end(), other.m_cache_n_samples_per_id.begin()) &&
         m_cache_n_samples_in_training.size() == m_cache_n_samples_in_training.size() && 
         std::equal(m_cache_n_samples_in_training.begin(), m_cache_n_samples_in_training.end(), other.m_cache_n_samples_in_training.begin()) &&
         bob::core::array::isClose(m_cache_B, other.m_cache_B, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_Ft_isigma_G, other.m_cache_Ft_isigma_G, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_eta, other.m_cache_eta, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_zeta, other.m_cache_zeta, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_cache_iota, other.m_cache_iota, r_epsilon, a_epsilon);
}

void bob::trainer::PLDATrainer::initialize(bob::machine::PLDABase& machine,
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Checks training data
  checkTrainingData(v_ar);

  // Gets dimension (first Arrayset)
  size_t n_features = v_ar[0].extent(1);
  m_dim_d = machine.getDimD();
  // Get dimensionalities from the PLDABase
  bob::core::array::assertSameDimensionLength(n_features, m_dim_d);
  m_dim_f = machine.getDimF();
  m_dim_g = machine.getDimG();

  // Reinitializes array members
  initMembers(v_ar);

  // Computes the mean and the covariance if required
  computeMeanVariance(machine, v_ar);

  // Initialization (e.g. using scatter)
  initFGSigma(machine, v_ar);
}

void bob::trainer::PLDATrainer::finalize(bob::machine::PLDABase& machine,
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Precomputes constant parts of the log likelihood and (gamma_a)
  precomputeLogLike(machine, v_ar);
  // Adds the case 1 sample if not already done (always used for scoring)
  machine.getAddGamma(1);
  machine.getAddLogLikeConstTerm(1);
}

void bob::trainer::PLDATrainer::checkTrainingData(const std::vector<blitz::Array<double,2> >& v_ar)
{
  // Checks that the vector of Arraysets is not empty
  if (v_ar.size() == 0)
    throw bob::trainer::EmptyTrainingSet();

  // Gets dimension (first Arrayset)
  int n_features = v_ar[0].extent(1);
  // Checks dimension consistency
  for (size_t i=0; i<v_ar.size(); ++i) {
    if (v_ar[i].extent(1) != n_features)
      throw bob::trainer::WrongNumberOfFeatures(v_ar[i].extent(1), n_features, i);
  } 
}

void bob::trainer::PLDATrainer::initMembers(const std::vector<blitz::Array<double,2> >& v_ar)
{
  // Gets dimension (first Arrayset)
  const size_t n_features = v_ar[0].extent(1); // dimensionality of the data
  const size_t n_identities = v_ar.size();

  m_cache_S.resize(n_features, n_features);
  m_cache_sum_z_second_order.resize(m_dim_f+m_dim_g, m_dim_f+m_dim_g);

  // Loops over the identities
  for (size_t i=0; i<n_identities; ++i) 
  {
    // Number of training samples for this identity
    const size_t n_i = v_ar[i].extent(0);
    // m_cache_z_first_order
    blitz::Array<double,2> z_i(n_i, m_dim_f+m_dim_g);
    m_cache_z_first_order.push_back(z_i);
    // m_z_second_order
    if (!m_use_sum_second_order)
    {
      blitz::Array<double,3> z2_i(n_i, m_dim_f+m_dim_g, m_dim_f+m_dim_g);
      m_cache_z_second_order.push_back(z2_i);
    }

    // m_cache_n_samples_per_id
    m_cache_n_samples_per_id.push_back(n_i);
    
    // Maps dependent on the number of samples per identity
    std::map<size_t,bool>::iterator it;
    it = m_cache_n_samples_in_training.find(n_i);
    if (it == m_cache_n_samples_in_training.end())
    {
      // Indicates if there are identities with n_i training samples and if
      // corresponding matrices are up to date.
      m_cache_n_samples_in_training[n_i] = false;
      // Allocates arrays for identities with n_i training samples
      m_cache_zeta[n_i].reference(blitz::Array<double,2>(m_dim_g, m_dim_g));
      m_cache_iota[n_i].reference(blitz::Array<double,2>(m_dim_f, m_dim_g));
    }
  }

  m_cache_B.resize(n_features, m_dim_f+m_dim_g);
  m_cache_Ft_isigma_G.resize(m_dim_f, m_dim_g);
  m_cache_eta.resize(m_dim_f, m_dim_g);

  // Working arrays
  resizeTmp();
}

void bob::trainer::PLDATrainer::resizeTmp()
{
  m_tmp_nf_1.resize(m_dim_f);
  m_tmp_nf_2.resize(m_dim_f);
  m_tmp_ng_1.resize(m_dim_g);
  m_tmp_D_1.resize(m_dim_d);
  m_tmp_D_2.resize(m_dim_d);
  m_tmp_nfng_nfng.resize(m_dim_f+m_dim_g, m_dim_f+m_dim_g);
  m_tmp_D_nfng_1.resize(m_dim_d, m_dim_f+m_dim_g);
  m_tmp_D_nfng_2.resize(m_dim_d, m_dim_f+m_dim_g);
}

void bob::trainer::PLDATrainer::computeMeanVariance(bob::machine::PLDABase& machine, 
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
    bob::math::scatter(data, m_cache_S, mu);
    // divides scatter by N-1
    m_cache_S /= static_cast<double>(n_samples-1);
  }
  else */
  {
    // Computes the mean and updates mu
    mu = 0.;
    size_t n_samples = 0;
    for (size_t j=0; j<v_ar.size(); ++j) {
      n_samples += v_ar[j].extent(0);
      for (int i=0; i<v_ar[j].extent(0); ++i)
        mu += v_ar[j](i,all);
    }
    mu /= static_cast<double>(n_samples);
    m_cache_S = 0.;
  }
}

void bob::trainer::PLDATrainer::initFGSigma(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Initializes F, G and sigma
  initF(machine, v_ar);
  initG(machine, v_ar);
  initSigma(machine, v_ar);

  // Precomputes values using new F, G and sigma
  machine.precompute();
}

void bob::trainer::PLDATrainer::initF(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,2>& F = machine.updateF();
  blitz::Range a = blitz::Range::all();

  // 1: between-class scatter
  if (m_initF_method == bob::trainer::PLDATrainer::BETWEEN_SCATTER) 
  {
    // a/ Computes between-class scatter matrix
    blitz::firstIndex bi;
    blitz::secondIndex bj;
    blitz::Array<double,2> S(machine.getDimD(), v_ar.size());
    S = 0.;
    m_tmp_D_1 = 0.;
    for (size_t i=0; i<v_ar.size(); ++i)
    {
      blitz::Array<double,1> Si = S(blitz::Range::all(),i);
      Si = 0.;
      for (int j=0; j<v_ar[i].extent(0); ++j)
      {
        // Si += x_ij
        Si += v_ar[i](j,a);
      }
      // Si = mean of the samples class i
      Si /= static_cast<double>(v_ar[i].extent(0));
      m_tmp_D_1 += Si;
    }
    m_tmp_D_1 /= static_cast<double>(v_ar.size());

    // b/ Removes the mean
    S = S(bi,bj) - m_tmp_D_1(bi);

    // c/ SVD of the between-class scatter matrix
    const size_t n_singular = std::min(machine.getDimD(),v_ar.size());
    blitz::Array<double,2> U(machine.getDimD(), n_singular);
    blitz::Array<double,1> sigma(n_singular);
    bob::math::svd(S, U, sigma);

    // d/ Updates F
    blitz::Array<double,2> Uslice = U(a, blitz::Range(0,m_dim_f-1));
    blitz::Array<double,1> sigma_slice = sigma(blitz::Range(0,m_dim_f-1));
    sigma_slice = blitz::sqrt(sigma_slice);
    F = Uslice(bi,bj) / sigma_slice(bj);
  }
  // otherwise: random initialization
  else {
    // F initialization
    bob::core::array::randn(*m_rng, F);
    F *= m_initF_ratio;
  }
}

void bob::trainer::PLDATrainer::initG(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,2>& G = machine.updateG();
  blitz::Range a = blitz::Range::all();

  // 1: within-class scatter
  if (m_initG_method == bob::trainer::PLDATrainer::WITHIN_SCATTER) 
  {
    // a/ Computes within-class scatter matrix
    blitz::firstIndex bi;
    blitz::secondIndex bj;
    size_t Nsamples=0;
    for (size_t i=0; i<v_ar.size(); ++i)
      Nsamples += v_ar[i].extent(0);
        
    blitz::Array<double,2> S(machine.getDimD(), Nsamples);
    S = 0.;
    m_tmp_D_1 = 0.;
    int counter = 0;
    for (size_t i=0; i<v_ar.size(); ++i)
    {
      // Computes the mean of the samples class i
      m_tmp_D_2 = 0.;
      for (int j=0; j<v_ar[i].extent(0); ++j)
      {
        // m_tmp_D_2 += x_ij
        m_tmp_D_2 += v_ar[i](j,a);
      }
      // m_tmp_D_2 = mean of the samples class i
      m_tmp_D_2 /= static_cast<double>(v_ar[i].extent(0));

      // Generates the scatter
      for (int j=0; j<v_ar[i].extent(0); ++j)
      {
        blitz::Array<double,1> Si = S(a, counter);
        // Si = x_ij - mean_i
        Si = v_ar[i](j,a) - m_tmp_D_2;
        // mean of the within class
        m_tmp_D_1 += Si;
        ++counter;
      }
    }
    m_tmp_D_1 /= static_cast<double>(Nsamples);

    // b/ Removes the mean
    S = S(bi,bj) - m_tmp_D_1(bi);

    // c/ SVD of the between-class scatter matrix
    blitz::Array<double,2> U(m_dim_d, std::min(m_dim_d, Nsamples));
    blitz::Array<double,1> sigma(std::min(m_dim_d, Nsamples));
    bob::math::svd(S, U, sigma);

    // d/ Updates G
    blitz::Array<double,2> Uslice = U(blitz::Range::all(), blitz::Range(0,m_dim_g-1));
    blitz::Array<double,1> sigma_slice = sigma(blitz::Range(0,m_dim_g-1));
    sigma_slice = blitz::sqrt(sigma_slice);
    G = Uslice(bi,bj) / sigma_slice(bj);
  }
  // otherwise: random initialization
  else {
    // G initialization
    bob::core::array::randn(*m_rng, G);
    G *= m_initG_ratio;
  }
}

void bob::trainer::PLDATrainer::initSigma(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  blitz::Array<double,1>& sigma = machine.updateSigma();
  blitz::Range a = blitz::Range::all();

  // 1: percentage of the variance of G
  if (m_initSigma_method == bob::trainer::PLDATrainer::VARIANCE_G) {
    const blitz::Array<double,2>& G = machine.getG();
    blitz::secondIndex bj;
    m_tmp_D_1 = blitz::mean(G, bj);
    // Updates sigma
    sigma = blitz::fabs(m_tmp_D_1) * m_initSigma_ratio;
  }
  // 2: constant value
  else if (m_initSigma_method == bob::trainer::PLDATrainer::CONSTANT) {
    sigma = m_initSigma_ratio;
  }
  // 3: percentage of the variance of the data
  else if (m_initSigma_method == bob::trainer::PLDATrainer::VARIANCE_DATA) {
    // a/ Computes the global mean
    //    m_tmp_D_1 = 1/N sum_i x_i
    m_tmp_D_1 = 0.;
    size_t Ns = 0;
    for (size_t i=0; i<v_ar.size(); ++i)
    {
      for (int j=0; j<v_ar[i].extent(0); ++j) 
        m_tmp_D_1 += v_ar[i](j,a);
      Ns += v_ar[i].extent(0);
    }
    m_tmp_D_1 /= static_cast<double>(Ns);
  
    // b/ Computes the variance:
    m_tmp_D_2 = 0.;
    for (size_t i=0; i<v_ar.size(); ++i)
      for (int j=0; j<v_ar[i].extent(0); ++j) 
        m_tmp_D_2 += blitz::pow2(v_ar[i](j,a) - m_tmp_D_1);
    sigma = m_initSigma_ratio * m_tmp_D_2 / static_cast<double>(Ns-1);
  }
  // otherwise: random initialization
  else {
    // sigma initialization
    bob::core::array::randn(*m_rng, sigma);
    sigma = blitz::fabs(sigma) * m_initSigma_ratio;
  }
  // Apply variance threshold
  machine.applyVarianceThreshold();
}

void bob::trainer::PLDATrainer::eStep(bob::machine::PLDABase& machine, 
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
  m_cache_sum_z_second_order = 0.;
  for (size_t i=0; i<v_ar.size(); ++i)
  {
    // Computes expectation of z_ij = [h_i w_ij]
    // 1/a/ Computes expectation of h_i
    // Loop over the samples
    m_tmp_nf_1 = 0.;
    for (int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_tmp_D_1 = x_sj-mu
      m_tmp_D_1 = v_ar[i](j,a) - mu;

      // m_tmp_nf_2 = F^T.beta.(x_sj-mu)
      bob::math::prod(FtBeta, m_tmp_D_1, m_tmp_nf_2);
      // m_tmp_nf_1 = sum_j F^T.beta.(x_sj-mu)
      m_tmp_nf_1 += m_tmp_nf_2;
    }
    const blitz::Array<double,2>& gamma_a = machine.getAddGamma(v_ar[i].extent(0));
    blitz::Range r_hi(0, m_dim_f-1);
    // m_tmp_nf_2 = E(h_i) = gamma_A  sum_j F^T.beta.(x_sj-mu)
    bob::math::prod(gamma_a, m_tmp_nf_1, m_tmp_nf_2);

    // 1/b/ Precomputes: m_tmp_D_2 = F.E{h_i}
    bob::math::prod(F, m_tmp_nf_2, m_tmp_D_2);

    // 2/ First and second order statistics of z
    // Precomputed values 
    blitz::Array<double,2>& zeta_a = m_cache_zeta[v_ar[i].extent(0)];
    blitz::Array<double,2>& iota_a = m_cache_iota[v_ar[i].extent(0)];
    blitz::Array<double,2> iotat_a = iota_a.transpose(1,0);

    // Extracts statistics of z_ij = [h_i w_ij] from y_i = [h_i w_i1 ... w_iJ]
    blitz::Range r1(0, m_dim_f-1);
    blitz::Range r2(m_dim_f, m_dim_f+m_dim_g-1);
    for (int j=0; j<v_ar[i].extent(0); ++j)
    {
      // 1/ First order statistics of z
      blitz::Array<double,1> z_first_order_ij_1 = m_cache_z_first_order[i](j,r1);
      z_first_order_ij_1 = m_tmp_nf_2; // E{h_i}
      // m_tmp_D_1 = x_sj - mu - F.E{h_i}
      m_tmp_D_1 = v_ar[i](j,a) - mu - m_tmp_D_2;
      // m_tmp_ng_1 = G^T.sigma^-1.(x_sj-mu-fhi)
      bob::math::prod(GtISigma, m_tmp_D_1, m_tmp_ng_1);
      // z_first_order_ij_2 = (Id+G^T.sigma^-1.G)^-1.G^T.sigma^-1.(x_sj-mu) = E{w_ij}
      blitz::Array<double,1> z_first_order_ij_2 = m_cache_z_first_order[i](j,r2);
      bob::math::prod(alpha, m_tmp_ng_1, z_first_order_ij_2); 

      // 2/ Second order statistics of z
      blitz::Array<double,2> z_sum_so_11 = m_cache_sum_z_second_order(r1,r1);
      blitz::Array<double,2> z_sum_so_12 = m_cache_sum_z_second_order(r1,r2);
      blitz::Array<double,2> z_sum_so_21 = m_cache_sum_z_second_order(r2,r1);
      blitz::Array<double,2> z_sum_so_22 = m_cache_sum_z_second_order(r2,r2);
      if (m_use_sum_second_order)
      {
        z_sum_so_11 += gamma_a + z_first_order_ij_1(bi) * z_first_order_ij_1(bj);
        z_sum_so_12 += iota_a + z_first_order_ij_1(bi) * z_first_order_ij_2(bj);
        z_sum_so_21 += iotat_a + z_first_order_ij_2(bi) * z_first_order_ij_1(bj);
        z_sum_so_22 += zeta_a + z_first_order_ij_2(bi) * z_first_order_ij_2(bj);
      }
      else
      {
        blitz::Array<double,2> z_so_11 = m_cache_z_second_order[i](j,r1,r1);
        z_so_11 = gamma_a + z_first_order_ij_1(bi) * z_first_order_ij_1(bj);
        z_sum_so_11 += z_so_11;
        blitz::Array<double,2> z_so_12 = m_cache_z_second_order[i](j,r1,r2);
        z_so_12 = iota_a + z_first_order_ij_1(bi) * z_first_order_ij_2(bj);
        z_sum_so_12 += z_so_12;
        blitz::Array<double,2> z_so_21 = m_cache_z_second_order[i](j,r2,r1);
        z_so_21 = iotat_a + z_first_order_ij_2(bi) * z_first_order_ij_1(bj);
        z_sum_so_21 += z_so_21;
        blitz::Array<double,2> z_so_22 = m_cache_z_second_order[i](j,r2,r2);
        z_so_22 = zeta_a + z_first_order_ij_2(bi) * z_first_order_ij_2(bj);
        z_sum_so_22 += z_so_22;
      }
    }
  }
}

void bob::trainer::PLDATrainer::precomputeFromFGSigma(bob::machine::PLDABase& machine)
{
  // Blitz compatibility: ugly fix (const_cast, as old blitz version does not  
  // provide a non-const version of transpose()) 
  const blitz::Array<double,2>& F = machine.getF();
  const blitz::Array<double,2> Ft = const_cast<blitz::Array<double,2>&>(F).transpose(1,0);
  const blitz::Array<double,2>& Gt_isigma = machine.getGtISigma();
  const blitz::Array<double,2> Gt_isigma_t = const_cast<blitz::Array<double,2>&>(Gt_isigma).transpose(1,0);
  const blitz::Array<double,2>& alpha = machine.getAlpha();

  // Precomputes F, G and sigma-based expressions
  bob::math::prod(Ft, Gt_isigma_t, m_cache_Ft_isigma_G);
  bob::math::prod(m_cache_Ft_isigma_G, alpha, m_cache_eta); 
  blitz::Array<double,2> etat = m_cache_eta.transpose(1,0);

  // Reinitializes all the zeta_a and iota_a
  std::map<size_t,bool>::iterator it;
  for (it=m_cache_n_samples_in_training.begin(); it!=m_cache_n_samples_in_training.end(); 
      ++it)
    it->second = false;

  for (it=m_cache_n_samples_in_training.begin(); it!=m_cache_n_samples_in_training.end(); 
      ++it)
  {
    size_t n_i = it->first;
    // Precomputes zeta and iota for identities with q_i training samples,
    // if not already done
    if (!it->second)
    {
      const blitz::Array<double,2>& gamma_a = machine.getAddGamma(n_i);
      blitz::Array<double,2>& zeta_a = m_cache_zeta[n_i];
      blitz::Array<double,2>& iota_a = m_cache_iota[n_i];
      bob::math::prod(gamma_a, m_cache_eta, iota_a);
      bob::math::prod(etat, iota_a, zeta_a);
      zeta_a += alpha;
      iota_a = - iota_a;
      // Now up to date
      it->second = true;
    }
  }
}

void bob::trainer::PLDATrainer::precomputeLogLike(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // Precomputes the log determinant of alpha and sigma
  machine.precomputeLogLike();

  // Precomputes the log likelihood constant term
  std::map<size_t,bool>::iterator it;
  for (it=m_cache_n_samples_in_training.begin(); 
       it!=m_cache_n_samples_in_training.end(); ++it)
  {
    // Precomputes the log likelihood constant term for identities with q_i 
    // training samples, if not already done
    machine.getAddLogLikeConstTerm(it->first);
  }
}


void bob::trainer::PLDATrainer::mStep(bob::machine::PLDABase& machine, 
  const std::vector<blitz::Array<double,2> >& v_ar) 
{
  // 1/ New estimate of B = {F G}
  updateFG(machine, v_ar);

  // 2/ New estimate of Sigma
  updateSigma(machine, v_ar);

  // 3/ Precomputes new values after updating F, G and sigma
  machine.precompute();
  // Precomputes useful variables using current estimates of F,G, and sigma
  precomputeFromFGSigma(machine);
}

void bob::trainer::PLDATrainer::updateFG(bob::machine::PLDABase& machine,
  const std::vector<blitz::Array<double,2> >& v_ar)
{
  /// Computes the B matrix (B = [F G])
  /// B = (sum_ij (x_ij-mu).E{z_i}^T).(sum_ij E{z_i.z_i^T})^-1

  // 1/ Computes the numerator (sum_ij (x_ij-mu).E{z_i}^T)
  // Gets the mean mu from the machine
  const blitz::Array<double,1>& mu = machine.getMu();
  blitz::Range a = blitz::Range::all();
  m_tmp_D_nfng_2 = 0.;
  for (size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for (int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_tmp_D_1 = x_sj-mu
      m_tmp_D_1 = v_ar[i](j,a) - mu;
      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_cache_z_first_order[i](j, a);
      // m_tmp_D_nfng_1 = (x_sj-mu).E{z_ij}^T
      bob::math::prod(m_tmp_D_1, z_first_order_ij, m_tmp_D_nfng_1);
      m_tmp_D_nfng_2 += m_tmp_D_nfng_1;
    }
  }

  // 2/ Computes the denominator inv(sum_ij E{z_i.z_i^T})
  bob::math::inv(m_cache_sum_z_second_order, m_tmp_nfng_nfng);

  // 3/ Computes numerator / denominator
  bob::math::prod(m_tmp_D_nfng_2, m_tmp_nfng_nfng, m_cache_B);

  // 4/ Updates the machine 
  blitz::Array<double, 2>& F = machine.updateF();
  blitz::Array<double, 2>& G = machine.updateG();
  F = m_cache_B(a, blitz::Range(0, m_dim_f-1));
  G = m_cache_B(a, blitz::Range(m_dim_f, m_dim_f+m_dim_g-1));
}

void bob::trainer::PLDATrainer::updateSigma(bob::machine::PLDABase& machine,
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
  for (size_t i=0; i<v_ar.size(); ++i)
  {
    // Loop over the samples
    for (int j=0; j<v_ar[i].extent(0); ++j)
    {
      // m_tmp_D_1 = x_ij-mu
      m_tmp_D_1 = v_ar[i](j,a) - mu;
      // sigma += Diag{(x_ij-mu).(x_ij-mu)^T}
      sigma += blitz::pow2(m_tmp_D_1);

      // z_first_order_ij = E{z_ij}
      blitz::Array<double,1> z_first_order_ij = m_cache_z_first_order[i](j,a);
      // m_tmp_D_2 = B.E{z_ij}
      bob::math::prod(m_cache_B, z_first_order_ij, m_tmp_D_2);
      // sigma -= Diag{B.E{z_ij}.(x_ij-mu)
      sigma -= (m_tmp_D_1 * m_tmp_D_2);
      ++n_IJ;
    }
  }
  // Normalizes by the number of samples
  sigma /= static_cast<double>(n_IJ);
  // Apply variance threshold
  machine.applyVarianceThreshold();
}

double bob::trainer::PLDATrainer::computeLikelihood(bob::machine::PLDABase& machine)
{
  double llh = 0.;
  // TODO: implement log likelihood computation
  return llh;
}

void bob::trainer::PLDATrainer::enrol(bob::machine::PLDAMachine& plda_machine,
  const blitz::Array<double,2>& ar) const
{
  // Gets dimension
  const size_t dim_d = ar.extent(1);
  const int n_samples = ar.extent(0);  
  // Compare the dimensionality from the base trainer/machine with the one
  // of the enrollment samples
  if (plda_machine.getDimD() != dim_d)
    throw bob::trainer::WrongNumberOfFeatures(plda_machine.getDimD(), dim_d, 0);
  const size_t dim_f = plda_machine.getDimF();
 
  // Resize working arrays
  m_tmp_D_1.resize(dim_d);
  m_tmp_D_2.resize(dim_d);
  m_tmp_nf_1.resize(dim_f);

  // Useful values from the base machine
  blitz::Array<double,1>& weighted_sum = plda_machine.updateWeightedSum();
  const blitz::Array<double,1>& mu = plda_machine.getPLDABase()->getMu();
  const blitz::Array<double,2>& beta = plda_machine.getPLDABase()->getBeta();
  const blitz::Array<double,2>& FtBeta = plda_machine.getPLDABase()->getFtBeta();

  // Updates the PLDA machine
  plda_machine.setNSamples(n_samples);
  double terma = 0.;
  weighted_sum = 0.;
  blitz::Range a = blitz::Range::all();
  for (int i=0; i<n_samples; ++i) {
    m_tmp_D_1 =  ar(i,a) - mu;
    // a/ weighted sum
    bob::math::prod(FtBeta, m_tmp_D_1, m_tmp_nf_1);
    weighted_sum += m_tmp_nf_1;
    // b/ first xi dependent term of the log likelihood
    bob::math::prod(beta, m_tmp_D_1, m_tmp_D_2);
    terma += -1 / 2. * blitz::sum(m_tmp_D_1 * m_tmp_D_2);
  }
  plda_machine.setWSumXitBetaXi(terma);

  // Adds the precomputed values for the cases N and N+1 if not already 
  // in the base machine (used by the forward function, 1 already added)
  plda_machine.getAddGamma(n_samples);
  plda_machine.getAddLogLikeConstTerm(n_samples);
  plda_machine.getAddGamma(n_samples+1);
  plda_machine.getAddLogLikeConstTerm(n_samples+1);
  plda_machine.setLogLikelihood(plda_machine.computeLogLikelihood(
                                  blitz::Array<double,2>(0,dim_d),true));
}
