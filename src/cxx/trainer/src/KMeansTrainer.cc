/**
 * @file cxx/trainer/src/KMeansTrainer.cc
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
#include "trainer/KMeansTrainer.h"

#include <io/Arrayset.h>
#include <cfloat>
#include <core/logging.h>
#include <boost/random.hpp>

using namespace Torch::machine;

Torch::trainer::KMeansTrainer::KMeansTrainer(double convergence_threshold, int max_iterations) :
  EMTrainer<KMeansMachine, Torch::io::Arrayset>(convergence_threshold, max_iterations) {
  seed = -1;
}
  
void Torch::trainer::KMeansTrainer::initialization(KMeansMachine& kMeansMachine, const Torch::io::Arrayset& ar) {
  // split data into as many chunks as there are means
  size_t n_data = ar.size();
  unsigned int n_chunk = n_data / kMeansMachine.getNMeans();
  
  boost::mt19937 rng;
  if (seed != -1) {
    rng.seed((uint32_t)seed);
  }
  
  // assign the i'th mean to a random example within the i'th chunk
  for(int i = 0; i < kMeansMachine.getNMeans(); i++) {
    boost::uniform_int<> range(i*n_chunk, (i+1)*n_chunk-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(rng, range);
    
    // get random index within chunk
    unsigned int index = die();

    // get the example at that index
    const blitz::Array<double, 1>& mean = ar.get<double,1>(index);
    
    // set the mean
    kMeansMachine.setMean(i, mean);
  } 
}

double Torch::trainer::KMeansTrainer::eStep(KMeansMachine& kmeans, const Torch::io::Arrayset& ar) {
    // initialise the accumulators
    double average_min_distance = 0;
    resetAccumulators(kmeans);

    // iterate over data samples
    for (size_t i=0; i < ar.size(); ++i) {
      // get example
      blitz::Array<double, 1> x(ar.get<double,1>(i));

      // find closest mean, and distance from that mean
      int closest_mean = -1;
      double min_distance = -1;
      kmeans.getClosestMean(x,closest_mean,min_distance);

      // accumulate the stats
      average_min_distance += min_distance;
      m_zeroethOrderStats(closest_mean)++;
      m_firstOrderStats(closest_mean,blitz::Range::all()) += x;
    }
    average_min_distance /= ar.size();
    
    return average_min_distance;
}

void Torch::trainer::KMeansTrainer::mStep(KMeansMachine& kmeans, const Torch::io::Arrayset&) {
    m_cache_newMeans.resize(kmeans.getNMeans(),kmeans.getNInputs());
    blitz::firstIndex i;
    blitz::secondIndex j;
    m_cache_newMeans = m_firstOrderStats(i,j) / m_zeroethOrderStats(i);
    kmeans.setMeans(m_cache_newMeans);
}

bool Torch::trainer::KMeansTrainer::resetAccumulators(KMeansMachine& kMeansMachine) {
  m_zeroethOrderStats.resize(kMeansMachine.getNMeans());
  m_zeroethOrderStats = 0;
  m_firstOrderStats.resize(kMeansMachine.getNMeans(), kMeansMachine.getNInputs());
  m_firstOrderStats = 0;
  return true;
}

void Torch::trainer::KMeansTrainer::setSeed(int seed) {
  this->seed = seed;
}

int Torch::trainer::KMeansTrainer::getSeed() {
  return seed;
}



