/**
 * @file machine/cxx/GMMMachine.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
#include "bob/machine/GMMMachine.h"
#include "bob/core/array_assert.h"
#include "bob/machine/Exception.h"
#include "bob/math/log.h"

bob::machine::GMMMachine::GMMMachine(): m_gaussians(0) {
  resize(0,0);
}

bob::machine::GMMMachine::GMMMachine(const size_t n_gaussians, const size_t n_inputs): 
  m_gaussians(0)
{
  resize(n_gaussians,n_inputs);
}

bob::machine::GMMMachine::GMMMachine(bob::io::HDF5File& config): 
  m_gaussians(0)
{
  load(config);
}

bob::machine::GMMMachine::GMMMachine(const GMMMachine& other): 
  Machine<blitz::Array<double,1>, double>(other), m_gaussians(0)
{
  copy(other);
}

bob::machine::GMMMachine& bob::machine::GMMMachine::operator=(const bob::machine::GMMMachine &other) {
  // protect against invalid self-assignment
  if (this != &other) 
    copy(other);
  
  // by convention, always return *this
  return *this;
}

bool bob::machine::GMMMachine::operator==(const bob::machine::GMMMachine& b) const {
  if(m_n_gaussians != b.m_n_gaussians || m_n_inputs != b.m_n_inputs) 
    return false;

  for(size_t i=0; i<m_n_gaussians; ++i) {
    if(!(*(m_gaussians[i]) == *(b.m_gaussians[i])))
      return false;
  }

  if(blitz::any(m_weights != b.m_weights))
    return false;

  return true;
}

bool bob::machine::GMMMachine::operator!=(const bob::machine::GMMMachine& b) const {
  return !(this->operator==(b));
}
 
void bob::machine::GMMMachine::copy(const GMMMachine& other) {
  m_n_gaussians = other.m_n_gaussians;
  m_n_inputs = other.m_n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = other.m_weights;

  // Initialise Gaussians
  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i) {
    boost::shared_ptr<bob::machine::Gaussian> g(new bob::machine::Gaussian(*(other.m_gaussians[i])));
    m_gaussians.push_back(g);
  }

  // Initialise cache
  initCache();
}

bob::machine::GMMMachine::~GMMMachine() { }

void bob::machine::GMMMachine::setNInputs(const size_t n_inputs) {
  resize(m_n_gaussians,n_inputs);
}

void bob::machine::GMMMachine::resize(const size_t n_gaussians, const size_t n_inputs) {
  m_n_gaussians = n_gaussians;
  m_n_inputs = n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = 1.0 / m_n_gaussians;

  // Initialise Gaussians
  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i) 
    m_gaussians.push_back(boost::shared_ptr<bob::machine::Gaussian>(new bob::machine::Gaussian(n_inputs)));

  // Initialise cache arrays
  initCache();
}


void bob::machine::GMMMachine::setWeights(const blitz::Array<double,1> &weights) {
  bob::core::array::assertSameShape(weights, m_weights);
  m_weights = weights;
  recomputeLogWeights();
}

void bob::machine::GMMMachine::recomputeLogWeights() const
{
  m_cache_log_weights = blitz::log(m_weights);
}

void bob::machine::GMMMachine::setMeans(const blitz::Array<double,2> &means) {
  bob::core::array::assertSameDimensionLength(means.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(means.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->updateMean() = means(i,blitz::Range::all());
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::getMeans(blitz::Array<double,2> &means) const {
  bob::core::array::assertSameDimensionLength(means.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(means.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) 
    means(i,blitz::Range::all()) = m_gaussians[i]->getMean(); 
}

void bob::machine::GMMMachine::setMeanSupervector(const blitz::Array<double,1> &mean_supervector) {
  bob::core::array::assertSameDimensionLength(mean_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) 
    m_gaussians[i]->updateMean() = mean_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::getMeanSupervector(blitz::Array<double,1> &mean_supervector) const {
  bob::core::array::assertSameDimensionLength(mean_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    mean_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1)) = m_gaussians[i]->getMean(); 
}

void bob::machine::GMMMachine::setVariances(const blitz::Array<double, 2 >& variances) {
  bob::core::array::assertSameDimensionLength(variances.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variances.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i]->updateVariance() = variances(i,blitz::Range::all());
    m_gaussians[i]->applyVarianceThresholds();
  }
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::getVariances(blitz::Array<double, 2 >& variances) const {
  bob::core::array::assertSameDimensionLength(variances.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variances.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) 
    variances(i,blitz::Range::all()) = m_gaussians[i]->getVariance();
}

void bob::machine::GMMMachine::setVarianceSupervector(const blitz::Array<double,1> &variance_supervector) {
  bob::core::array::assertSameDimensionLength(variance_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i]->updateVariance() = variance_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i]->applyVarianceThresholds();
  }
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::getVarianceSupervector(blitz::Array<double,1> &variance_supervector) const {
  bob::core::array::assertSameDimensionLength(variance_supervector.extent(0), m_n_gaussians*m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) {
    variance_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1)) = m_gaussians[i]->getVariance(); 
  }
}

void bob::machine::GMMMachine::setVarianceThresholds(const double value) {
  for(size_t i=0; i<m_n_gaussians; ++i) 
    m_gaussians[i]->setVarianceThresholds(value);
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::setVarianceThresholds(blitz::Array<double, 1> variance_thresholds) {
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) 
    m_gaussians[i]->setVarianceThresholds(variance_thresholds);
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::setVarianceThresholds(const blitz::Array<double, 2>& variance_thresholds) {
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i)
    m_gaussians[i]->setVarianceThresholds(variance_thresholds(i,blitz::Range::all())); 
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::getVarianceThresholds(blitz::Array<double, 2>& variance_thresholds) const {
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(variance_thresholds.extent(1), m_n_inputs);
  for(size_t i=0; i<m_n_gaussians; ++i) 
    variance_thresholds(i,blitz::Range::all()) = m_gaussians[i]->getVarianceThresholds();
}

double bob::machine::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x, 
  blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const 
{
  // Check dimension
  bob::core::array::assertSameDimensionLength(log_weighted_gaussian_likelihoods.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_n_inputs);
  return logLikelihood_(x,log_weighted_gaussian_likelihoods);
}

double bob::machine::GMMMachine::logLikelihood_(const blitz::Array<double, 1> &x, 
  blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const 
{
  // Initialise variables
  double log_likelihood = bob::math::Log::LogZero;

  // Accumulate the weighted log likelihoods from each Gaussian
  for(size_t i=0; i<m_n_gaussians; ++i) {
    double l = m_cache_log_weights(i) + m_gaussians[i]->logLikelihood_(x);
    log_weighted_gaussian_likelihoods(i) = l;
    log_likelihood = bob::math::Log::logAdd(log_likelihood, l);
  }

  // Return log(p(x|GMMMachine))
  return log_likelihood;
}

double bob::machine::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x) const {
  // Check dimension
  bob::core::array::assertSameDimensionLength(x.extent(0), m_n_inputs);
  // Call the other logLikelihood_ (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  return logLikelihood_(x,m_cache_log_weighted_gaussian_likelihoods);
}

double bob::machine::GMMMachine::logLikelihood_(const blitz::Array<double, 1> &x) const {
  // Call the other logLikelihood (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  return logLikelihood_(x,m_cache_log_weighted_gaussian_likelihoods);
}

void bob::machine::GMMMachine::forward(const blitz::Array<double,1>& input, double& output) const {
  if(static_cast<size_t>(input.extent(0)) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }

  forward_(input,output);
}

void bob::machine::GMMMachine::forward_(const blitz::Array<double,1>& input,
    double& output) const {
  output = logLikelihood(input);
}

void bob::machine::GMMMachine::accStatistics(const blitz::Array<double,2>& input,
    bob::machine::GMMStats& stats) const {
  // iterate over data
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<input.extent(0); ++i) {
    // Get example
    blitz::Array<double,1> x(input(i,a));
    // Accumulate statistics
    accStatistics(x,stats);
  }
}

void bob::machine::GMMMachine::accStatistics_(const blitz::Array<double,2>& input, bob::machine::GMMStats& stats) const {
  // iterate over data
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<input.extent(0); ++i) {
    // Get example
    blitz::Array<double,1> x(input(i, a));
    // Accumulate statistics
    accStatistics_(x,stats);
  }
}

void bob::machine::GMMMachine::accStatistics(const blitz::Array<double, 1>& x, bob::machine::GMMStats& stats) const {
  // check GMMStats size
  bob::core::array::assertSameDimensionLength(stats.sumPx.extent(0), m_n_gaussians);
  bob::core::array::assertSameDimensionLength(stats.sumPx.extent(1), m_n_inputs);

  // Calculate Gaussian and GMM likelihoods
  // - m_cache_log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  double log_likelihood = logLikelihood(x, m_cache_log_weighted_gaussian_likelihoods);

  accStatisticsInternal(x, stats, log_likelihood);
}

void bob::machine::GMMMachine::accStatistics_(const blitz::Array<double, 1>& x, bob::machine::GMMStats& stats) const {
  // Calculate Gaussian and GMM likelihoods
  // - m_cache_log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  double log_likelihood = logLikelihood_(x, m_cache_log_weighted_gaussian_likelihoods);

  accStatisticsInternal(x, stats, log_likelihood);
}

void bob::machine::GMMMachine::accStatisticsInternal(const blitz::Array<double, 1>& x,
  bob::machine::GMMStats& stats, const double log_likelihood) const 
{
  // Calculate responsibilities
  m_cache_P = blitz::exp(m_cache_log_weighted_gaussian_likelihoods - log_likelihood);

  // Accumulate statistics
  // - total likelihood
  stats.log_likelihood += log_likelihood;

  // - number of samples
  stats.T++;

  // - responsibilities
  stats.n += m_cache_P;

  // - first order stats
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  m_cache_Px = m_cache_P(i) * x(j);
  
  stats.sumPx += m_cache_Px;

  // - second order stats
  stats.sumPxx += (m_cache_Px(i,j) * x(j));
}


boost::shared_ptr<bob::machine::Gaussian> bob::machine::GMMMachine::getGaussian(const size_t i) {
  if(i>=m_n_gaussians) 
    throw bob::machine::Exception();
  return m_gaussians[i];
}

void bob::machine::GMMMachine::save(bob::io::HDF5File& config) const {
  int64_t v = static_cast<int64_t>(m_n_gaussians);
  config.set("m_n_gaussians", v);
  v = static_cast<int64_t>(m_n_inputs);
  config.set("m_n_inputs", v);

  for(size_t i=0; i<m_n_gaussians; ++i) {
    std::ostringstream oss;
    oss << "m_gaussians" << i;
   
    if (!config.hasGroup(oss.str())) config.createGroup(oss.str());
    config.cd(oss.str());
    m_gaussians[i]->save(config);
    config.cd("..");
  }

  config.setArray("m_weights", m_weights);
}

void bob::machine::GMMMachine::load(bob::io::HDF5File& config) {
  int64_t v;
  v = config.read<int64_t>("m_n_gaussians");
  m_n_gaussians = static_cast<size_t>(v);
  v = config.read<int64_t>("m_n_inputs");
  m_n_inputs = static_cast<size_t>(v);
  
  m_gaussians.clear();
  for(size_t i=0; i<m_n_gaussians; ++i) {
    m_gaussians.push_back(boost::shared_ptr<bob::machine::Gaussian>(new bob::machine::Gaussian(m_n_inputs)));
    std::ostringstream oss;
    oss << "m_gaussians" << i;
    config.cd(oss.str());
    m_gaussians[i]->load(config);
    config.cd("..");
  }

  m_weights.resize(m_n_gaussians);
  config.readArray("m_weights", m_weights);

  // Initialise cache
  initCache();
}

void bob::machine::GMMMachine::updateCacheSupervectors() const
{
  m_cache_mean_supervector.resize(m_n_gaussians*m_n_inputs);
  m_cache_variance_supervector.resize(m_n_gaussians*m_n_inputs);
  
  for(size_t i=0; i<m_n_gaussians; ++i) {
    blitz::Range range(i*m_n_inputs, (i+1)*m_n_inputs-1);
    m_cache_mean_supervector(range) = m_gaussians[i]->getMean();
    m_cache_variance_supervector(range) = m_gaussians[i]->getVariance();
  }
  m_cache_supervector = true;
}

void bob::machine::GMMMachine::initCache() const {
  // Initialise cache arrays
  m_cache_log_weights.resize(m_n_gaussians);
  recomputeLogWeights();
  m_cache_log_weighted_gaussian_likelihoods.resize(m_n_gaussians);
  m_cache_P.resize(m_n_gaussians);
  m_cache_Px.resize(m_n_gaussians,m_n_inputs);
  m_cache_supervector = false;
}

void bob::machine::GMMMachine::reloadCacheSupervectors() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
}

const blitz::Array<double,1>& bob::machine::GMMMachine::getMeanSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_mean_supervector;
} 

const blitz::Array<double,1>& bob::machine::GMMMachine::getVarianceSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_variance_supervector;
} 

namespace bob {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const GMMMachine& machine) {
      os << "Weights = " << machine.m_weights << std::endl;
      for(size_t i=0; i < machine.m_n_gaussians; ++i) {
        os << "Gaussian " << i << ": " << std::endl << *(machine.m_gaussians[i]);
      }

      return os;
    }
  }
}
