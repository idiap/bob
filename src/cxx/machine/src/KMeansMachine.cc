/**
 * @file cxx/machine/src/KMeansMachine.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "machine/KMeansMachine.h"
#include "machine/Exception.h"

using namespace std;

Torch::machine::KMeansMachine::KMeansMachine(int n_means, int n_inputs): m_n_means(n_means), 
    m_n_inputs(n_inputs), m_means(n_means, n_inputs), m_cache_means(n_means, n_inputs) {
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


double Torch::machine::KMeansMachine::getDistanceFromMean(const blitz::Array<double,1> &x, int i) const {
  return blitz::sum(blitz::pow2(m_means(i,blitz::Range::all()) - x));
}

void Torch::machine::KMeansMachine::getClosestMean(const blitz::Array<double,1> &x, int &closest_mean, double &min_distance) const {
  
  min_distance = DBL_MAX;
  
  for(int i = 0; i < m_n_means; i++) {
    double this_distance = getDistanceFromMean(x,i);
    if(this_distance < min_distance) {
      min_distance = this_distance;
      closest_mean = i;
    }
  }
  
}

double Torch::machine::KMeansMachine::getMinDistance(const blitz::Array<double,1> &input) const {
  int closest_mean = -1;
  double min_distance = -1;
  getClosestMean(input,closest_mean,min_distance);
  return min_distance;
}

void Torch::machine::KMeansMachine::getVariancesAndWeightsForEachCluster(const Torch::io::Arrayset &ar, blitz::Array<double,2> &variances, blitz::Array<double,1> &weights) const {
  // initialise output arrays
  variances.resize(m_n_means, m_n_inputs);
  weights.resize(m_n_means);
  variances = 0;
  weights = 0;
  
  // initialise (temporary) mean array
  m_cache_means.resize(m_n_means, m_n_inputs);
  m_cache_means = 0;
  
  // iterate over data
  for (size_t i=0; i < ar.size(); ++i) {
    // - get example
    blitz::Array<double,1> x = ar.get<double,1>(i);
    
    // - find closest mean
    int closest_mean = -1;
    double min_distance = -1;
    getClosestMean(x,closest_mean,min_distance);
    
    // - accumulate stats
    m_cache_means(closest_mean, blitz::Range::all()) += x;
    variances(closest_mean, blitz::Range::all()) += blitz::pow2(x);
    weights(closest_mean)++;
  }
  
  // calculate final variances and weights
  blitz::firstIndex idx1;
  blitz::secondIndex idx2;
  
  // find means
  m_cache_means = m_cache_means(idx1,idx2) / weights(idx1);
  
  // find variances
  variances = variances(idx1,idx2) / weights(idx1);
  variances -= blitz::pow2(m_cache_means);
  
  // find weights
  weights = weights / blitz::sum(weights);
}

void Torch::machine::KMeansMachine::forward(const blitz::Array<double,1>& input, double& output) const {
  if (input.extent(0) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }
  forward_(input,output); 
}

void Torch::machine::KMeansMachine::forward_(const blitz::Array<double,1>& input, double& output) const {
  output = getMinDistance(input);
}

int Torch::machine::KMeansMachine::getNMeans() const {
  return m_n_means;
}

int Torch::machine::KMeansMachine::getNInputs() const {
  return m_n_inputs;
}
