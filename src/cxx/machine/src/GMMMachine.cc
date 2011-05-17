#include "machine/GMMMachine.h"
#include <core/logging.h>

using namespace Torch::machine::Log;

Torch::machine::GMMMachine::GMMMachine() : m_gaussians(NULL) {
  resize(0,0);
}

Torch::machine::GMMMachine::GMMMachine(int n_gaussians, int n_inputs) : m_gaussians(NULL) {
  resize(n_gaussians,n_inputs);
}

Torch::machine::GMMMachine::GMMMachine(const GMMMachine& other) : Machine<FrameSample, double>(other), m_gaussians(NULL) {
  copy(other);
}

Torch::machine::GMMMachine & Torch::machine::GMMMachine::operator= (const GMMMachine &other) {
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // by convention, always return *this
  return *this;
}

void Torch::machine::GMMMachine::copy(const GMMMachine& other) {
  m_n_gaussians = other.m_n_gaussians;
  m_n_inputs = other.m_n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = other.m_weights;

  // Initialise Gaussians
  if (m_gaussians != NULL) {
    delete [] m_gaussians;
  }
  
  m_gaussians = new Gaussian[m_n_gaussians];
  
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i] = other.m_gaussians[i];
  }
}

Torch::machine::GMMMachine:: ~GMMMachine() {
  if (m_gaussians != NULL) delete[] m_gaussians;
}

void Torch::machine::GMMMachine::setNInputs(int n_inputs) {
  resize(m_n_gaussians,n_inputs);
}

int Torch::machine::GMMMachine::getNInputs() const {
  return m_n_inputs;
}

void Torch::machine::GMMMachine::resize(int n_gaussians, int n_inputs) {
  m_n_gaussians = n_gaussians;
  m_n_inputs = n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = 1.0 / m_n_gaussians;

  // Initialise Gaussians
  if (m_gaussians != NULL) {
    delete [] m_gaussians;
  }
  
  m_gaussians = new Gaussian [m_n_gaussians];
  
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].resize(m_n_inputs);
  }
}


void Torch::machine::GMMMachine::getWeights(blitz::Array<double,1> &weights) const {
  weights.resize(m_n_gaussians);
  weights = m_weights;
}

blitz::Array<double, 1> Torch::machine::GMMMachine::getWeights() const {
  blitz::Array<double, 1> weights(m_n_gaussians);
  getWeights(weights);
  return weights;
}

void Torch::machine::GMMMachine::setWeights(const blitz::Array< double, 1 >& weights) {
  m_weights = weights;
}

void Torch::machine::GMMMachine::setMeans(const blitz::Array< double, 2 >& means) {
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].setMean(means(i,blitz::Range::all()));
  }
}

void Torch::machine::GMMMachine::getMeans(blitz::Array<double,2> &means) const {
  means.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> mean(m_n_inputs);
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].getMean(mean);
    means(i,blitz::Range::all()) = mean;
  }
}

blitz::Array<double,2> Torch::machine::GMMMachine::getMeans() const {
  blitz::Array<double, 2> means(m_n_gaussians, m_n_inputs);
  getMeans(means);
  return means;
}

void Torch::machine::GMMMachine::setVariances(const blitz::Array< double, 2 >& variances) {
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].setVariance(variances(i,blitz::Range::all()));
  }
}

void Torch::machine::GMMMachine::getVariances( blitz::Array< double, 2 >& variances) const {
  variances.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> variance(m_n_inputs);
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].getVariance(variance);
    variances(i,blitz::Range::all()) = variance;
  }
}

blitz::Array<double,2> Torch::machine::GMMMachine::getVariances() const {
  blitz::Array<double, 2> variances(m_n_gaussians, m_n_inputs);
  getVariances(variances);
  return variances;
}

void Torch::machine::GMMMachine::setVarianceThresholds(double factor) {
  for(int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(factor);
  } 
}

void Torch::machine::GMMMachine::setVarianceThresholds(blitz::Array<double, 1> variance_thresholds) {
  for(int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(variance_thresholds);
  }
}

void Torch::machine::GMMMachine::setVarianceThresholds(const blitz::Array<double, 2>& variance_thresholds) {
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(variance_thresholds(i,blitz::Range::all()));
  }
}

void Torch::machine::GMMMachine::getVarianceThresholds(blitz::Array<double, 2>& variance_thresholds) const {
  variance_thresholds.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> this_variance_thresholds(m_n_inputs);
  for (int i=0; i < m_n_gaussians; ++i) {
    m_gaussians[i].getVarianceThresholds(this_variance_thresholds);
    variance_thresholds(i,blitz::Range::all()) = this_variance_thresholds;
  }
}


blitz::Array<double,2> Torch::machine::GMMMachine::getVarianceThresholds() const {
  blitz::Array<double, 2> varianceThresholds(m_n_gaussians, m_n_inputs);
  getVarianceThresholds(varianceThresholds);
  return varianceThresholds;
}

double Torch::machine::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x, blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const {
  // Initialise variables
  log_weighted_gaussian_likelihoods.resize(m_n_gaussians);
  double log_likelihood = LogZero;

  // Accumulate the weighted log likelihoods from each Gaussian
  for (int i=0; i < m_n_gaussians; ++i) {
    double l = log(m_weights(i)) + m_gaussians[i].logLikelihood(x);
    log_weighted_gaussian_likelihoods(i) = l;
    log_likelihood = LogAdd(log_likelihood, l);
  }

  // Return log(p(x|GMMMachine))
  return log_likelihood;
}

double Torch::machine::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x) const {
  // Call the other logLikelihood (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  blitz::Array<double,1> log_weighted_gaussian_likelihoods;
  return logLikelihood(x,log_weighted_gaussian_likelihoods);
}

void Torch::machine::GMMMachine::forward (const FrameSample& input, double& output) const {
  output = logLikelihood(input.getFrame());
}

void Torch::machine::GMMMachine::accStatistics(const Torch::trainer::Sampler<Torch::machine::FrameSample>& sampler, Torch::machine::GMMStats& stats) const {
  // iterate over data
  for (int64_t i=0; i < sampler.getNSamples(); ++i) {

    // Get example
    blitz::Array<double,1> x(sampler.getSample(i).getFrame());

    // Accumulate statistics
    accStatistics(x,stats);
  }
}

void Torch::machine::GMMMachine::accStatistics(const blitz::Array<double, 1>& x, Torch::machine::GMMStats& stats) const {

  // Calculate Gaussian and GMM likelihoods
  // - log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  blitz::Array<double,1> log_weighted_gaussian_likelihoods;
  double log_likelihood = logLikelihood(x, log_weighted_gaussian_likelihoods);

  // Calculate responsibilities
  blitz::Array<double,1> P(m_n_gaussians);
  P = blitz::exp(log_weighted_gaussian_likelihoods - log_likelihood);

  // Accumulate statistics
  // - total likelihood
  stats.log_likelihood += log_likelihood;

  // - number of samples
  stats.T++;

  // - responsibilities
  stats.n += P;

  // - first order stats
  blitz::Array<double,2> Px(m_n_gaussians,m_n_inputs);
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  Px = P(i) * x(j);
  
  /*
  std::cout << "P:" << P << std::endl;
  std::cout << "x:" << x << std::endl;
  std::cout << "Px:" << Px << std::endl;
  std::cout << "sumPx:" << stats.sumPx << std::endl;
  */
  stats.sumPx += Px;
  //std::cout << "sumPx:" << stats.sumPx << std::endl;

  // - second order stats
  blitz::Array<double,2> Pxx(m_n_gaussians,m_n_inputs);
  Pxx = Px(i,j) * x(j);
  stats.sumPxx += Pxx;
}


Torch::machine::Gaussian* Torch::machine::GMMMachine::getGaussian(int i) const {
  if (i >= 0 && i < m_n_gaussians) {
    return &(m_gaussians[i]);
  }
  else {
    return NULL;
  }
}

int Torch::machine::GMMMachine::getNGaussians() const {
  return m_n_gaussians;
}


void Torch::machine::GMMMachine::print() const {
  Torch::core::info << "Weights = " << m_weights << std::endl;
  for (int i=0; i < m_n_gaussians; ++i) {
    Torch::core::info << "Gaussian " << i << ": " << std::endl;
    m_gaussians[i].print();
  }
}

