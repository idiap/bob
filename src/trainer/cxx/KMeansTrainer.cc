/**
 * @file trainer/cxx/KMeansTrainer.cc
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

#include "bob/trainer/KMeansTrainer.h"
#include "bob/core/array_copy.h"
#include <boost/random.hpp>

bob::trainer::KMeansTrainer::KMeansTrainer(double convergence_threshold,
    size_t max_iterations, bool compute_likelihood):
  bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >(
    convergence_threshold, max_iterations, compute_likelihood), 
  m_seed(-1), m_average_min_distance(0),
  m_zeroethOrderStats(0), m_firstOrderStats(0,0)
{
}

bob::trainer::KMeansTrainer::KMeansTrainer(const bob::trainer::KMeansTrainer& other):
  bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >(
    other.m_convergence_threshold, other.m_max_iterations, other.m_compute_likelihood), 
  m_seed(other.m_seed), m_average_min_distance(other.m_average_min_distance),
  m_zeroethOrderStats(bob::core::array::ccopy(other.m_zeroethOrderStats)), 
  m_firstOrderStats(bob::core::array::ccopy(other.m_firstOrderStats))
{
}
 
bob::trainer::KMeansTrainer& bob::trainer::KMeansTrainer::operator=
(const bob::trainer::KMeansTrainer& other) 
{
  if(this != &other)
  {
    EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >::operator=(other);
    m_seed = other.m_seed;
    m_average_min_distance = other.m_average_min_distance;
    m_zeroethOrderStats.reference(bob::core::array::ccopy(other.m_zeroethOrderStats));
    m_firstOrderStats.reference(bob::core::array::ccopy(other.m_firstOrderStats));
  }
  return *this;
}

bool bob::trainer::KMeansTrainer::operator==(const bob::trainer::KMeansTrainer& b) const {
  return EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >::operator==(b) && 
         m_seed == b.m_seed && m_average_min_distance == b.m_average_min_distance &&
         bob::core::array::hasSameShape(m_zeroethOrderStats, b.m_zeroethOrderStats) &&
         bob::core::array::hasSameShape(m_firstOrderStats, b.m_firstOrderStats) &&
         blitz::all(m_zeroethOrderStats == b.m_zeroethOrderStats) &&
         blitz::all(m_firstOrderStats == b.m_firstOrderStats);
}

bool bob::trainer::KMeansTrainer::operator!=(const bob::trainer::KMeansTrainer& b) const {
  return !(this->operator==(b));
}
 
void bob::trainer::KMeansTrainer::initialization(bob::machine::KMeansMachine& kmeans,
  const blitz::Array<double,2>& ar) 
{
  // split data into as many chunks as there are means
  size_t n_data = ar.extent(0);
  unsigned int n_chunk = n_data / kmeans.getNMeans();
  
  boost::mt19937 rng;
  if(m_seed != -1) rng.seed((uint32_t)m_seed);
  
  // assign the i'th mean to a random example within the i'th chunk
  blitz::Range a = blitz::Range::all();
  for(size_t i=0; i<kmeans.getNMeans(); ++i) 
  {
    // TODO: Check that samples are not equal?
    boost::uniform_int<> range(i*n_chunk, (i+1)*n_chunk-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(rng, range);
    
    // get random index within chunk
    unsigned int index = die();

    // get the example at that index
    blitz::Array<double, 1> mean = ar(index,a);
    
    // set the mean
    kmeans.setMean(i, mean);
  }

  // Resize the accumulator
  m_zeroethOrderStats.resize(kmeans.getNMeans());
  m_firstOrderStats.resize(kmeans.getNMeans(), kmeans.getNInputs());
}

void bob::trainer::KMeansTrainer::eStep(bob::machine::KMeansMachine& kmeans, 
  const blitz::Array<double,2>& ar)
{
  // initialise the accumulators
  resetAccumulators(kmeans);

  // iterate over data samples
  blitz::Range a = blitz::Range::all();
  for(int i=0; i<ar.extent(0); ++i) {
    // get example
    blitz::Array<double, 1> x(ar(i,a));

    // find closest mean, and distance from that mean
    size_t closest_mean = 0;
    double min_distance = 0;
    kmeans.getClosestMean(x,closest_mean,min_distance);

    // accumulate the stats
    m_average_min_distance += min_distance;
    ++m_zeroethOrderStats(closest_mean);
    m_firstOrderStats(closest_mean,blitz::Range::all()) += x;
  }
  m_average_min_distance /= static_cast<double>(ar.extent(0));
}

void bob::trainer::KMeansTrainer::mStep(bob::machine::KMeansMachine& kmeans, 
  const blitz::Array<double,2>&) 
{
  blitz::Array<double,2>& means = kmeans.updateMeans();
  for(size_t i=0; i<kmeans.getNMeans(); ++i)
  {
    means(i,blitz::Range::all()) = 
      m_firstOrderStats(i,blitz::Range::all()) / m_zeroethOrderStats(i);
  }
}

double bob::trainer::KMeansTrainer::computeLikelihood(bob::machine::KMeansMachine& kmeans)
{
  return m_average_min_distance;
}

void bob::trainer::KMeansTrainer::finalization(bob::machine::KMeansMachine& kmeans,
  const blitz::Array<double,2>& ar) 
{
}

bool bob::trainer::KMeansTrainer::resetAccumulators(bob::machine::KMeansMachine& kmeans)
{
  m_average_min_distance = 0;
  m_zeroethOrderStats = 0;
  m_firstOrderStats = 0;
  return true;
}

void bob::trainer::KMeansTrainer::setSeed(int seed) {
  m_seed = seed;
}

void bob::trainer::KMeansTrainer::setZeroethOrderStats(const blitz::Array<double,1>& zeroethOrderStats)
{
  bob::core::array::assertSameShape(m_zeroethOrderStats, zeroethOrderStats);
  m_zeroethOrderStats = zeroethOrderStats;
}

void bob::trainer::KMeansTrainer::setFirstOrderStats(const blitz::Array<double,2>& firstOrderStats)
{
  bob::core::array::assertSameShape(m_firstOrderStats, firstOrderStats);
  m_firstOrderStats = firstOrderStats;
}

