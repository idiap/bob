#include "machine/KMeansMachine.h"

using namespace std;

Torch::machine::KMeansMachine::KMeansMachine(int n_means, int n_inputs) : m_n_means(n_means), m_n_inputs(n_inputs), m_means(n_means, n_inputs) {
  m_means = 0;
}

Torch::machine::KMeansMachine::~KMeansMachine() { 
  
}

void Torch::machine::KMeansMachine::setMeans(const blitz::Array<double,2> &means) {
  m_means = means;
}

void Torch::machine::KMeansMachine::setMean(int i, const blitz::Array<double,1> &mean) {
  m_means(i,blitz::Range::all()) = mean;
}

void Torch::machine::KMeansMachine::getMean(int i, blitz::Array<double,1> &mean) const {
  mean.resize(m_n_inputs);
  mean = m_means(i,blitz::Range::all());
}

void Torch::machine::KMeansMachine::getMeans(blitz::Array<double,2> &means) const {
  means.resize(m_n_means,m_n_inputs);
  means = m_means;
}


blitz::Array<double,2> Torch::machine::KMeansMachine::getMeans() const {
  blitz::Array<double,2> means(m_n_means, m_n_inputs);
  getMeans(means);
  return means;
}

double Torch::machine::KMeansMachine::getDistanceFromMean(const blitz::Array<float,1> &x, int i) const {
  return blitz::sum(blitz::pow2(m_means(i,blitz::Range::all()) - x));
}

void Torch::machine::KMeansMachine::getClosestMean(const blitz::Array<float,1> &x, int &closest_mean, double &min_distance) const {
  
  min_distance = DBL_MAX;
  
  for(int i = 0; i < m_n_means; i++) {
    double this_distance = getDistanceFromMean(x,i);
    if(this_distance < min_distance) {
      min_distance = this_distance;
      closest_mean = i;
    }
  }
  
}

double Torch::machine::KMeansMachine::getMinDistance(const blitz::Array<float,1> &input) const {
  int closest_mean = -1;
  double min_distance = -1;
  getClosestMean(input,closest_mean,min_distance);
  return min_distance;
}

void Torch::machine::KMeansMachine::getVariancesAndWeightsForEachCluster(const Torch::trainer::Sampler<FrameSample> &sampler, blitz::Array<double,2> &variances, blitz::Array<double,1> &weights) const {
  // initialise output arrays
  variances.resize(m_n_means, m_n_inputs);
  weights.resize(m_n_means);
  variances = 0;
  weights = 0;
  
  // initialise (temporary) mean array
  blitz::Array<double,2> means(m_n_means, m_n_inputs);
  means = 0;
  
  // iterate over data
  for (int64_t i=0; i < sampler.getNSamples(); ++i) {
    // - get example
    blitz::Array<float,1> x = sampler.getSample(i).getFrame();
    
    // - find closest mean
    int closest_mean = -1;
    double min_distance = -1;
    getClosestMean(x,closest_mean,min_distance);
    
    // - accumulate stats
    means(closest_mean, blitz::Range::all()) += x;
    variances(closest_mean, blitz::Range::all()) += blitz::pow2(x);
    weights(closest_mean)++;
  }
  
  // calculate final variances and weights
  blitz::firstIndex idx1;
  blitz::secondIndex idx2;
  
  // find means
  means = means(idx1,idx2) / weights(idx1);
  
  // find variances
  variances = variances(idx1,idx2) / weights(idx1);
  variances -= blitz::pow2(means);
  
  // find weights
  weights = weights / blitz::sum(weights);
}

void Torch::machine::KMeansMachine::forward(const FrameSample& input, double& output) const {
  output = getMinDistance(input.getFrame());
}

int Torch::machine::KMeansMachine::getNMeans() const {
  return m_n_means;
}

int Torch::machine::KMeansMachine::getNInputs() const {
  return m_n_inputs;
}
