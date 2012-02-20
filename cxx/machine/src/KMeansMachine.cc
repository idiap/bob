/**
 * @file cxx/machine/src/KMeansMachine.cc
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

#include "machine/KMeansMachine.h"

#include "core/array_assert.h"
#include "core/array_copy.h"
#include "machine/Exception.h"
#include <limits>

namespace ca = bob::core::array;
namespace mach = bob::machine;
namespace io = bob::io;

mach::KMeansMachine::KMeansMachine(): 
  m_n_means(0), m_n_inputs(0), m_means(0,0),
  m_cache_means(0,0) 
{
  m_means = 0;
}

mach::KMeansMachine::KMeansMachine(const size_t n_means, const size_t n_inputs): 
  m_n_means(n_means), m_n_inputs(n_inputs), m_means(n_means, n_inputs),
  m_cache_means(n_means, n_inputs) 
{
  m_means = 0;
}

mach::KMeansMachine::KMeansMachine(const blitz::Array<double,2>& means): 
  m_n_means(means.extent(0)), m_n_inputs(means.extent(1)), 
  m_means(ca::ccopy(means)),
  m_cache_means(means.shape()) 
{
}

mach::KMeansMachine::KMeansMachine(const mach::KMeansMachine& other): 
  m_n_means(other.m_n_means), m_n_inputs(other.m_n_inputs), 
  m_means(ca::ccopy(other.m_means)),
  m_cache_means(other.m_cache_means.shape()) 
{
}

mach::KMeansMachine::KMeansMachine(io::HDF5File& config) 
{
  load(config);
}

mach::KMeansMachine::~KMeansMachine() { }

mach::KMeansMachine& mach::KMeansMachine::operator=
(const mach::KMeansMachine& other) 
{
  m_n_means = other.m_n_means;
  m_n_inputs = other.m_n_inputs;
  m_means.reference(ca::ccopy(other.m_means));
  m_cache_means.resize(other.m_means.shape());
  return *this;
}

bool mach::KMeansMachine::operator==(const mach::KMeansMachine& b) const {
  return m_n_inputs == b.m_n_inputs && m_n_means == b.m_n_means &&
         blitz::all(m_means == b.m_means);
}

void mach::KMeansMachine::load(io::HDF5File& config) 
{
  //reads all data directly into the member variables
  m_means.reference(config.readArray<double,2>("means"));
  m_n_means = m_means.extent(0);
  m_n_inputs = m_means.extent(1);
  m_cache_means.resize(m_n_means, m_n_inputs);
}

void mach::KMeansMachine::save(io::HDF5File& config) const 
{
  config.setArray("means", m_means);
} 

void mach::KMeansMachine::setMeans(const blitz::Array<double,2> &means) 
{
  ca::assertSameShape(means, m_means);
  m_means = means;
}

void mach::KMeansMachine::setMean(const size_t i, const blitz::Array<double,1> &mean) 
{
  // TODO: check i is in range
  ca::assertSameDimensionLength(mean.extent(0), m_means.extent(1));
  m_means(i,blitz::Range::all()) = mean;
}

void mach::KMeansMachine::getMean(const size_t i, blitz::Array<double,1> &mean) const 
{
  // TODO: check i is in range
  ca::assertSameDimensionLength(mean.extent(0), m_means.extent(1));
  mean = m_means(i,blitz::Range::all());
}

double mach::KMeansMachine::getDistanceFromMean(const blitz::Array<double,1> &x, 
  const size_t i) const 
{
  return blitz::sum(blitz::pow2(m_means(i,blitz::Range::all()) - x));
}

void mach::KMeansMachine::getClosestMean(const blitz::Array<double,1> &x, 
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

double mach::KMeansMachine::getMinDistance(const blitz::Array<double,1>& input) const 
{
  size_t closest_mean = 0;
  double min_distance = 0;
  getClosestMean(input,closest_mean,min_distance);
  return min_distance;
}

void mach::KMeansMachine::getVariancesAndWeightsForEachCluster(const io::Arrayset &ar, 
  blitz::Array<double,2>& variances, blitz::Array<double,1>& weights) const 
{
  // check and initialise output arrays
  ca::assertSameShape(variances, m_means);
  ca::assertSameDimensionLength(weights.extent(0), m_n_means);
  variances = 0;
  weights = 0;
  
  // initialise (temporary) mean array
  m_cache_means = 0;
  
  // iterate over data
  for(size_t i=0; i<ar.size(); ++i) {
    // - get example
    blitz::Array<double,1> x = ar.get<double,1>(i);
    
    // - find closest mean
    size_t closest_mean = 0;
    double min_distance = 0;
    getClosestMean(x,closest_mean,min_distance);
    
    // - accumulate stats
    m_cache_means(closest_mean, blitz::Range::all()) += x;
    variances(closest_mean, blitz::Range::all()) += blitz::pow2(x);
    ++weights(closest_mean);
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

void mach::KMeansMachine::forward(const blitz::Array<double,1>& input, double& output) const 
{
  if(static_cast<size_t>(input.extent(0)) != m_n_inputs) {
    throw NInputsMismatch(m_n_inputs, input.extent(0));
  }
  forward_(input,output); 
}

void mach::KMeansMachine::forward_(const blitz::Array<double,1>& input, double& output) const 
{
  output = getMinDistance(input);
}

void mach::KMeansMachine::resize(const size_t n_means, const size_t n_inputs) 
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
