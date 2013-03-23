/**
 * @file machine/cxx/KMeansMachine.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/machine/KMeansMachine.h>

#include <bob/core/assert.h>
#include <bob/core/array_copy.h>
#include <bob/machine/Exception.h>
#include <limits>

bob::machine::KMeansMachine::KMeansMachine(): 
  m_n_means(0), m_n_inputs(0), m_means(0,0),
  m_cache_means(0,0) 
{
  m_means = 0;
}

bob::machine::KMeansMachine::KMeansMachine(const size_t n_means, const size_t n_inputs): 
  m_n_means(n_means), m_n_inputs(n_inputs), m_means(n_means, n_inputs),
  m_cache_means(n_means, n_inputs) 
{
  m_means = 0;
}

bob::machine::KMeansMachine::KMeansMachine(const blitz::Array<double,2>& means): 
  m_n_means(means.extent(0)), m_n_inputs(means.extent(1)), 
  m_means(bob::core::array::ccopy(means)),
  m_cache_means(means.shape()) 
{
}

bob::machine::KMeansMachine::KMeansMachine(const bob::machine::KMeansMachine& other): 
  m_n_means(other.m_n_means), m_n_inputs(other.m_n_inputs), 
  m_means(bob::core::array::ccopy(other.m_means)),
  m_cache_means(other.m_cache_means.shape()) 
{
}

bob::machine::KMeansMachine::KMeansMachine(bob::io::HDF5File& config) 
{
  load(config);
}

bob::machine::KMeansMachine::~KMeansMachine() { }

bob::machine::KMeansMachine& bob::machine::KMeansMachine::operator=
(const bob::machine::KMeansMachine& other) 
{
  if(this != &other)
  {
    m_n_means = other.m_n_means;
    m_n_inputs = other.m_n_inputs;
    m_means.reference(bob::core::array::ccopy(other.m_means));
    m_cache_means.resize(other.m_means.shape());
  }
  return *this;
}

bool bob::machine::KMeansMachine::operator==(const bob::machine::KMeansMachine& b) const {
  return m_n_inputs == b.m_n_inputs && m_n_means == b.m_n_means &&
         blitz::all(m_means == b.m_means);
}

bool bob::machine::KMeansMachine::operator!=(const bob::machine::KMeansMachine& b) const {
  return !(this->operator==(b));
}

void bob::machine::KMeansMachine::load(bob::io::HDF5File& config) 
{
  //reads all data directly into the member variables
  m_means.reference(config.readArray<double,2>("means"));
  m_n_means = m_means.extent(0);
  m_n_inputs = m_means.extent(1);
  m_cache_means.resize(m_n_means, m_n_inputs);
}

void bob::machine::KMeansMachine::save(bob::io::HDF5File& config) const 
{
  config.setArray("means", m_means);
} 

void bob::machine::KMeansMachine::setMeans(const blitz::Array<double,2> &means) 
{
  bob::core::array::assertSameShape(means, m_means);
  m_means = means;
}

void bob::machine::KMeansMachine::setMean(const size_t i, const blitz::Array<double,1> &mean) 
{
  if(i>=m_n_means) throw bob::core::InvalidArgumentException("mean index", i);
  bob::core::array::assertSameDimensionLength(mean.extent(0), m_means.extent(1));
  m_means(i,blitz::Range::all()) = mean;
}

void bob::machine::KMeansMachine::getMean(const size_t i, blitz::Array<double,1> &mean) const 
{
  if(i>=m_n_means) throw bob::core::InvalidArgumentException("mean index", i);
  bob::core::array::assertSameDimensionLength(mean.extent(0), m_means.extent(1));
  mean = m_means(i,blitz::Range::all());
}

double bob::machine::KMeansMachine::getDistanceFromMean(const blitz::Array<double,1> &x, 
  const size_t i) const 
{
  return blitz::sum(blitz::pow2(m_means(i,blitz::Range::all()) - x));
}

void bob::machine::KMeansMachine::getClosestMean(const blitz::Array<double,1> &x, 
  size_t &closest_mean, double &min_distance) const 
{
  min_distance = std::numeric_limits<double>::max();
  
  for(size_t i=0; i<m_n_means; ++i) {
    double this_distance = getDistanceFromMean(x,i);
    if(this_distance < min_distance) {
      min_distance = this_distance;
      closest_mean = i;
    }
  } 
}

double bob::machine::KMeansMachine::getMinDistance(const blitz::Array<double,1>& input) const 
{
  size_t closest_mean = 0;
  double min_distance = 0;
  getClosestMean(input,closest_mean,min_distance);
  return min_distance;
}

void bob::machine::KMeansMachine::getVariancesAndWeightsForEachClusterInit(blitz::Array<double,2>& variances, blitz::Array<double,1>& weights) const 
{
  // check arguments
  bob::core::array::assertSameShape(variances, m_means);
  bob::core::array::assertSameDimensionLength(weights.extent(0), m_n_means);

  // initialise output arrays
  bob::core::array::assertSameShape(variances, m_means);
  bob::core::array::assertSameDimensionLength(weights.extent(0), m_n_means);
  variances = 0;
  weights = 0;
  
  // initialise (temporary) mean array
  m_cache_means = 0;
}
  
void bob::machine::KMeansMachine::getVariancesAndWeightsForEachClusterAcc(const blitz::Array<double,2>& data, blitz::Array<double,2>& variances, blitz::Array<double,1>& weights) const 
{
  // check arguments
  bob::core::array::assertSameShape(variances, m_means);
  bob::core::array::assertSameDimensionLength(weights.extent(0), m_n_means);

  // iterate over data
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<data.extent(0); ++i) {
    // - get example
    blitz::Array<double,1> x(data(i,a));
    
    // - find closest mean
    size_t closest_mean = 0;
    double min_distance = 0;
    getClosestMean(x,closest_mean,min_distance);
    
    // - accumulate stats
    m_cache_means(closest_mean, blitz::Range::all()) += x;
    variances(closest_mean, blitz::Range::all()) += blitz::pow2(x);
    ++weights(closest_mean);
  }
}
  
void bob::machine::KMeansMachine::getVariancesAndWeightsForEachClusterFin(blitz::Array<double,2>& variances, blitz::Array<double,1>& weights) const 
{
  // check arguments
  bob::core::array::assertSameShape(variances, m_means);
  bob::core::array::assertSameDimensionLength(weights.extent(0), m_n_means);

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

void bob::machine::KMeansMachine::setCacheMeans(const blitz::Array<double,2> &cache_means) 
{
  bob::core::array::assertSameShape(cache_means, m_cache_means);
  m_cache_means = cache_means;
}

void bob::machine::KMeansMachine::getVariancesAndWeightsForEachCluster(const blitz::Array<double,2>& data, blitz::Array<double,2>& variances, blitz::Array<double,1>& weights) const 
{
  // initialise
  getVariancesAndWeightsForEachClusterInit(variances, weights);
  // accumulate
  getVariancesAndWeightsForEachClusterAcc(data, variances, weights);
  // merge/finalize
  getVariancesAndWeightsForEachClusterFin(variances, weights);
}

void bob::machine::KMeansMachine::forward(const blitz::Array<double,1>& input, double& output) const 
{
  if(static_cast<size_t>(input.extent(0)) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }
  forward_(input,output); 
}

void bob::machine::KMeansMachine::forward_(const blitz::Array<double,1>& input, double& output) const 
{
  output = getMinDistance(input);
}

void bob::machine::KMeansMachine::resize(const size_t n_means, const size_t n_inputs) 
{
  m_n_means = n_means;
  m_n_inputs = n_inputs;
  m_means.resizeAndPreserve(n_means, n_inputs);
  m_cache_means.resizeAndPreserve(n_means, n_inputs);
}

namespace bob{
  namespace machine{
    std::ostream& operator<<(std::ostream& os, const KMeansMachine& km) {
      os << "Means = " << km.m_means << std::endl;
      return os;
    }
  }
}
