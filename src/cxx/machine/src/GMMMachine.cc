#include "machine/GMMMachine.h"
#include "core/logging.h"
#include "machine/Exception.h"

namespace TLog = Torch::machine::Log;
namespace mach = Torch::machine;

mach::GMMMachine::GMMMachine(): m_gaussians(0), m_cache_supervector(false) {
  resize(0,0);
}

mach::GMMMachine::GMMMachine(int n_gaussians, int n_inputs): m_gaussians(0), m_cache_supervector(false) {
  resize(n_gaussians,n_inputs);
}

mach::GMMMachine::GMMMachine(Torch::io::HDF5File& config): m_gaussians(0), m_cache_supervector(false) {
  load(config);
}

mach::GMMMachine::GMMMachine(const GMMMachine& other): Machine<blitz::Array<double,1>, double>(other), m_gaussians(0), m_cache_supervector(false) {
  copy(other);
}

mach::GMMMachine& mach::GMMMachine::operator=(const mach::GMMMachine &other) {
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // Always ignore the cache
  m_cache_supervector = false;
  // by convention, always return *this
  return *this;
}

bool mach::GMMMachine::operator==(const mach::GMMMachine& b) const {
  if (m_n_gaussians != b.m_n_gaussians || m_n_inputs != b.m_n_inputs) {
    return false;
  }

  for(int i = 0; i < m_n_gaussians; i++) {
    if(!(m_gaussians[i] == b.m_gaussians[i])) {
      return false;
    }
  }

  if(blitz::any(m_weights != b.m_weights)) {
    return false;
  }

  return true;
}

void mach::GMMMachine::copy(const GMMMachine& other) {
  m_n_gaussians = other.m_n_gaussians;
  m_n_inputs = other.m_n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = other.m_weights;

  // Initialise Gaussians
  if(m_gaussians != 0) {
    delete [] m_gaussians;
  }
  
  m_gaussians = new Gaussian[m_n_gaussians];
  
  for (int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i] = other.m_gaussians[i];
  }
}

mach::GMMMachine::~GMMMachine() {
  if(m_gaussians != 0) delete [] m_gaussians;
}

void mach::GMMMachine::setNInputs(int n_inputs) {
  resize(m_n_gaussians,n_inputs);
  m_cache_supervector = false;
}

int mach::GMMMachine::getNInputs() const {
  return m_n_inputs;
}

void mach::GMMMachine::resize(int n_gaussians, int n_inputs) {
  m_n_gaussians = n_gaussians;
  m_n_inputs = n_inputs;

  // Initialise weights
  m_weights.resize(m_n_gaussians);
  m_weights = 1.0 / m_n_gaussians;

  // Initialise Gaussians
  if(m_gaussians != 0) {
    delete [] m_gaussians;
  }
  
  m_gaussians = new Gaussian [m_n_gaussians];
  
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].resize(m_n_inputs);
  }

  // Initialise cache arrays
  m_cache_log_weighted_gaussian_likelihoods.resize(m_n_gaussians);
  m_cache_P.resize(m_n_gaussians);
  m_cache_Px.resize(m_n_gaussians,m_n_inputs);
  m_cache_Pxx.resize(m_n_gaussians,m_n_inputs);
  m_cache_supervector = false;
}


void mach::GMMMachine::setWeights(const blitz::Array<double,1> &weights) {
  m_weights = weights;
}

void mach::GMMMachine::getWeights(blitz::Array<double,1> &weights) const {
  weights.resize(m_n_gaussians);
  weights = m_weights;
}

void mach::GMMMachine::setMeans(const blitz::Array<double,2> &means) {
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].setMean(means(i,blitz::Range::all()));
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::getMeans(blitz::Array<double,2> &means) const {
  means.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> mean(m_n_inputs);
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].getMean(mean);
    means(i,blitz::Range::all()) = mean;
  }
}

void mach::GMMMachine::setMeanSupervector(const blitz::Array<double,1> &mean_supervector) {
  for(int i=0; i<m_n_gaussians; ++i) {
    const blitz::Array<double,1> mean = mean_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i].setMean(mean);
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::getMeanSupervector(blitz::Array<double,1> &mean_supervector) const {
  mean_supervector.resize(m_n_gaussians*m_n_inputs);
  for(int i=0; i<m_n_gaussians; ++i) {
    blitz::Array<double,1> mean = mean_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i].getMean(mean);
  }
}

void mach::GMMMachine::setVariances(const blitz::Array<double, 2 >& variances) {
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].setVariance(variances(i,blitz::Range::all()));
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::getVariances(blitz::Array<double, 2 >& variances) const {
  variances.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> variance(m_n_inputs);
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].getVariance(variance);
    variances(i,blitz::Range::all()) = variance;
  }
}

void mach::GMMMachine::setVarianceSupervector(const blitz::Array<double,1> &variance_supervector) {
  for(int i=0; i<m_n_gaussians; ++i) {
    const blitz::Array<double,1> variance = variance_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i].setVariance(variance);
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::getVarianceSupervector(blitz::Array<double,1> &variance_supervector) const {
  variance_supervector.resize(m_n_gaussians*m_n_inputs);
  for(int i=0; i<m_n_gaussians; ++i) {
    blitz::Array<double,1> variance = variance_supervector(blitz::Range(i*m_n_inputs, (i+1)*m_n_inputs-1));
    m_gaussians[i].getVariance(variance);
  }
}

void mach::GMMMachine::setVarianceThresholds(double factor) {
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(factor);
  } 
}

void mach::GMMMachine::setVarianceThresholds(blitz::Array<double, 1> variance_thresholds) {
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(variance_thresholds);
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::setVarianceThresholds(const blitz::Array<double, 2>& variance_thresholds) {
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].setVarianceThresholds(variance_thresholds(i,blitz::Range::all()));
  }
  m_cache_supervector = false;
}

void mach::GMMMachine::getVarianceThresholds(blitz::Array<double, 2>& variance_thresholds) const {
  variance_thresholds.resize(m_n_gaussians,m_n_inputs);
  blitz::Array<double,1> this_variance_thresholds(m_n_inputs);
  for(int i=0; i<m_n_gaussians; ++i) {
    m_gaussians[i].getVarianceThresholds(this_variance_thresholds);
    variance_thresholds(i,blitz::Range::all()) = this_variance_thresholds;
  }
}

double mach::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x, blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const {
  // Initialise variables
  log_weighted_gaussian_likelihoods.resize(m_n_gaussians);
  double log_likelihood = TLog::LogZero;

  // Accumulate the weighted log likelihoods from each Gaussian
  for(int i=0; i<m_n_gaussians; ++i) {
    double l = log(m_weights(i)) + m_gaussians[i].logLikelihood(x);
    log_weighted_gaussian_likelihoods(i) = l;
    log_likelihood = TLog::LogAdd(log_likelihood, l);
  }

  // Return log(p(x|GMMMachine))
  return log_likelihood;
}

double mach::GMMMachine::logLikelihood(const blitz::Array<double, 1> &x) const {
  // Call the other logLikelihood (overloaded) function
  // (log_weighted_gaussian_likelihoods will be discarded)
  return logLikelihood(x,m_cache_log_weighted_gaussian_likelihoods);
}

void mach::GMMMachine::forward(const blitz::Array<double,1>& input, double& output) const {
  if(input.extent(0) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }

  forward_(input,output);
}

void mach::GMMMachine::forward_(const blitz::Array<double,1>& input, double& output) const {
  output = logLikelihood(input);
}

void mach::GMMMachine::accStatistics(const Torch::io::Arrayset& ar, mach::GMMStats& stats) const {
  // iterate over data
  for(size_t i=0; i<ar.size(); ++i) {

    // Get example
    blitz::Array<double,1> x(ar.get<double,1>(i));

    // Accumulate statistics
    accStatistics(x,stats);
  }
}

void mach::GMMMachine::accStatistics(const blitz::Array<double, 1>& x, mach::GMMStats& stats) const {

  // Calculate Gaussian and GMM likelihoods
  // - m_cache_log_weighted_gaussian_likelihoods(i) = log(weight_i*p(x|gaussian_i))
  // - log_likelihood = log(sum_i(weight_i*p(x|gaussian_i)))
  double log_likelihood = logLikelihood(x, m_cache_log_weighted_gaussian_likelihoods);

  // Calculate responsibilities
  m_cache_P.resize(m_n_gaussians);
  m_cache_P = blitz::exp(m_cache_log_weighted_gaussian_likelihoods - log_likelihood);

  // Accumulate statistics
  // - total likelihood
  stats.log_likelihood += log_likelihood;

  // - number of samples
  stats.T++;

  // - responsibilities
  stats.n += m_cache_P;

  // - first order stats
  m_cache_Px.resize(m_n_gaussians,m_n_inputs);
  blitz::firstIndex i;
  blitz::secondIndex j;
  
  m_cache_Px = m_cache_P(i) * x(j);
  
  /*
  std::cout << "P:" << m_cache_P << std::endl;
  std::cout << "x:" << x << std::endl;
  std::cout << "Px:" << m_cache_Px << std::endl;
  std::cout << "sumPx:" << stats.sumPx << std::endl;
  */
  stats.sumPx += m_cache_Px;
  //std::cout << "sumPx:" << stats.sumPx << std::endl;

  // - second order stats
  m_cache_Pxx.resize(m_n_gaussians,m_n_inputs);
  m_cache_Pxx = m_cache_Px(i,j) * x(j);
  stats.sumPxx += m_cache_Pxx;
}


mach::Gaussian* mach::GMMMachine::getGaussian(int i) const {
  if(i >= 0 && i < m_n_gaussians) {
    return &(m_gaussians[i]);
  }
  else {
    return 0;
  }
}

int mach::GMMMachine::getNGaussians() const {
  return m_n_gaussians;
}

void mach::GMMMachine::save(Torch::io::HDF5File& config) const {
  config.set("m_n_gaussians", m_n_gaussians);
  config.set("m_n_inputs", m_n_inputs);

  for(int i=0; i<m_n_gaussians; i++) {
    std::ostringstream oss;
    oss << "m_gaussians" << i;
    
    config.cd(oss.str());
    m_gaussians[i].save(config);
    config.cd("..");
  }

  config.setArray("m_weights", m_weights);
}

void mach::GMMMachine::load(Torch::io::HDF5File& config) {
  m_n_gaussians = config.read<int64_t>("m_n_gaussians");
  m_n_inputs = config.read<int64_t>("m_n_inputs");
  
  if(m_gaussians != 0) {
    delete [] m_gaussians;
  }

  m_gaussians = new Gaussian[m_n_gaussians];
  
  for(int i=0; i<m_n_gaussians; i++) {
    std::ostringstream oss;
    oss << "m_gaussians" << i;
    config.cd(oss.str());
    m_gaussians[i].load(config);
    config.cd("..");
  }

  m_weights.resize(m_n_gaussians);
  config.readArray("m_weights", m_weights);
  m_cache_supervector = false;
}

void mach::GMMMachine::updateCacheSupervectors() const
{
  m_cache_mean_supervector.resize(m_n_gaussians*m_n_inputs);
  m_cache_variance_supervector.resize(m_n_gaussians*m_n_inputs);
  
  for(int i=0; i<m_n_gaussians; ++i) {
    blitz::Range range(i*m_n_inputs, (i+1)*m_n_inputs-1);
    blitz::Array<double,1> mean = m_cache_mean_supervector(range);
    m_gaussians[i].getMean(mean);
    blitz::Array<double,1> variance = m_cache_variance_supervector(range);
    m_gaussians[i].getVariance(variance);
  }
}

void mach::GMMMachine::reloadCacheSupervectors() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
}

const blitz::Array<double,1>& mach::GMMMachine::getMeanSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_mean_supervector;
} 

const blitz::Array<double,1>& mach::GMMMachine::getVarianceSupervector() const {
  if(!m_cache_supervector)
    updateCacheSupervectors();
  return m_cache_variance_supervector;
} 

namespace Torch {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const GMMMachine& machine) {
      os << "Weights = " << machine.m_weights << std::endl;
      for (int i=0; i < machine.m_n_gaussians; ++i) {
        os << "Gaussian " << i << ": " << std::endl << machine.m_gaussians[i];
      }

      return os;
    }
  }
}
