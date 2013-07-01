/**
 * @file trainer/cxx/KMeansTrainer.cc
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

#include <bob/trainer/KMeansTrainer.h>
#include <bob/core/array_copy.h>
#include <bob/trainer/Exception.h>
#include <boost/random.hpp>

#if BOOST_VERSION >= 104700
#include <boost/random/discrete_distribution.hpp>
#endif

bob::trainer::KMeansTrainer::KMeansTrainer(double convergence_threshold,
    size_t max_iterations, bool compute_likelihood, InitializationMethod i_m):
  bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >(
    convergence_threshold, max_iterations, compute_likelihood), 
  m_initialization_method(i_m),
  m_rng(new boost::mt19937()), m_average_min_distance(0),
  m_zeroethOrderStats(0), m_firstOrderStats(0,0)
{
}

bob::trainer::KMeansTrainer::KMeansTrainer(const bob::trainer::KMeansTrainer& other):
  bob::trainer::EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >(
    other.m_convergence_threshold, other.m_max_iterations, other.m_compute_likelihood), 
  m_initialization_method(other.m_initialization_method),
  m_rng(other.m_rng), m_average_min_distance(other.m_average_min_distance),
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
    m_initialization_method = other.m_initialization_method;
    m_rng = other.m_rng;
    m_average_min_distance = other.m_average_min_distance;
    m_zeroethOrderStats.reference(bob::core::array::ccopy(other.m_zeroethOrderStats));
    m_firstOrderStats.reference(bob::core::array::ccopy(other.m_firstOrderStats));
  }
  return *this;
}

bool bob::trainer::KMeansTrainer::operator==(const bob::trainer::KMeansTrainer& b) const {
  return EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >::operator==(b) &&
         m_initialization_method == b.m_initialization_method &&
         *m_rng == *(b.m_rng) && m_average_min_distance == b.m_average_min_distance &&
         bob::core::array::hasSameShape(m_zeroethOrderStats, b.m_zeroethOrderStats) &&
         bob::core::array::hasSameShape(m_firstOrderStats, b.m_firstOrderStats) &&
         blitz::all(m_zeroethOrderStats == b.m_zeroethOrderStats) &&
         blitz::all(m_firstOrderStats == b.m_firstOrderStats);
}

bool bob::trainer::KMeansTrainer::operator!=(const bob::trainer::KMeansTrainer& b) const {
  return !(this->operator==(b));
}
 
void bob::trainer::KMeansTrainer::initialize(bob::machine::KMeansMachine& kmeans,
  const blitz::Array<double,2>& ar) 
{
  // split data into as many chunks as there are means
  size_t n_data = ar.extent(0);
 
  // assign the i'th mean to a random example within the i'th chunk
  blitz::Range a = blitz::Range::all();
#if BOOST_VERSION >= 104700
  if(m_initialization_method == RANDOM || m_initialization_method == RANDOM_NO_DUPLICATE) // Random initialization
#endif
  {
    unsigned int n_chunk = n_data / kmeans.getNMeans();
    size_t n_max_trials = (size_t)n_chunk * 5;
    blitz::Array<double,1> cur_mean;
    if(m_initialization_method == RANDOM_NO_DUPLICATE)
      cur_mean.resize(kmeans.getNInputs());
 
    for(size_t i=0; i<kmeans.getNMeans(); ++i) 
    {
      boost::uniform_int<> range(i*n_chunk, (i+1)*n_chunk-1);
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(*m_rng, range);
      
      // get random index within chunk
      unsigned int index = die();

      // get the example at that index
      blitz::Array<double, 1> mean = ar(index,a);

      if(m_initialization_method == RANDOM_NO_DUPLICATE)
      {
        size_t count = 0;
        while(count < n_max_trials)
        {
          // check that the selected sampled is different than all the previously 
          // selected ones
          bool valid = true;
          for(size_t j=0; j<i && valid; ++j)
          {
            kmeans.getMean(j, cur_mean);
            valid = blitz::any(mean != cur_mean);
          }
          // if different, stop otherwise, try with another one
          if(valid) 
            break;
          else
          {
            index = die();
            mean = ar(index,a);
            ++count;
          }
        }
        // Initialization fails
        if(count >= n_max_trials)
          throw bob::trainer::KMeansInitializationFailure();
      }
      
      // set the mean
      kmeans.setMean(i, mean);
    }
  }
#if BOOST_VERSION >= 104700
  else // K-Means++
  {
    // 1.a. Selects one sample randomly
    boost::uniform_int<> range(0, n_data-1);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(*m_rng, range);
    //   Gets the example at a random index
    blitz::Array<double,1> mean = ar(die(),a);
    kmeans.setMean(0, mean);

    // 1.b. Loops, computes probability distribution and select samples accordingly
    blitz::Array<double,1> weights(n_data);
    for(size_t m=1; m<kmeans.getNMeans(); ++m) 
    {
      // For each sample, puts the distance to the closest mean in the weight vector
      for(size_t s=0; s<n_data; ++s)
      {
        blitz::Array<double,1> s_cur = ar(s,a);
        double& w_cur = weights(s);
        // Initializes with the distance to first mean
        w_cur = kmeans.getDistanceFromMean(s_cur, 0);
        // Loops over the remaining mean and update the mean distance if required
        for(size_t i=1; i<m; ++i)
          w_cur = std::min(w_cur, kmeans.getDistanceFromMean(s_cur, i));
      }
      // Square and normalize the weights vectors such that
      // \f$weights[x] = D(x)^{2} \sum_{y} D(y)^{2}\f$
      weights = blitz::pow2(weights);
      weights /= blitz::sum(weights);

      // Takes a sample according to the weights distribution
      // Blitz iterators is fine as the weights array should be C-style contiguous
      bob::core::array::assertCContiguous(weights);
      boost::random::discrete_distribution<> die2(weights.begin(), weights.end());
      blitz::Array<double,1> new_mean = ar(die2(*m_rng),a); 
      kmeans.setMean(m, new_mean);
    }
  }
#endif
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

void bob::trainer::KMeansTrainer::finalize(bob::machine::KMeansMachine& kmeans,
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

