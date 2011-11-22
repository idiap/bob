/**
 * @file cxx/trainer/trainer/KMeansTrainer.h
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
#ifndef KMEANSTRAINER_H
#define KMEANSTRAINER_H

#include "io/Arrayset.h"
#include "machine/KMeansMachine.h"
#include "trainer/EMTrainer.h"

namespace Torch {
namespace trainer {

/// Trains a KMeans machine.
/// @brief This class implements the expectation-maximisation algorithm for a k-means machine.
/// @details See Section 9.1 of Bishop, "Pattern recognition and machine learning", 2006
///          It uses a random initialisation of the means followed by the expectation-maximization algorithm
class KMeansTrainer : public EMTrainer<Torch::machine::KMeansMachine, Torch::io::Arrayset>
{
public:
  
  /// Constructor
  KMeansTrainer(double convergence_threshold = 0.001, int max_iterations = 10);
  
  virtual ~KMeansTrainer() {}


  /// Initialise the means randomly. 
  /// Data is split into as many chunks as there are means, 
  /// then each mean is set to a random example within each chunk.
  virtual void initialization(Torch::machine::KMeansMachine& kMeansMachine, const Torch::io::Arrayset& sampler);
  
  /// Accumulate across the dataset:
  /// - zeroeth and first order statistics
  /// - average distance from the closest mean 
  /// Implements EMTrainer::eStep(double &)
  virtual double eStep(Torch::machine::KMeansMachine& kmeans, const Torch::io::Arrayset& data);
  
  virtual void mStep(Torch::machine::KMeansMachine& kmeans, const Torch::io::Arrayset&);
  
  /// Reset the statistics accumulators
  /// to the correct size and a value of zero.
  bool resetAccumulators(Torch::machine::KMeansMachine& kMeansMachine);

  /// Set the seed used to genrated pseudo-random numbers
  void setSeed(int seed);

  /// Get the seed
  int getSeed();
  
protected:

  /// Seed used to genrated pseudo-random numbers
  int seed;
  
  /// Zeroeth order statistics accumulator.
  /// The k'th value in m_zeroethOrderStats is the denominator of 
  /// equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
  blitz::Array<double,1> m_zeroethOrderStats;

  /// First order statistics accumulator.
  /// The k'th row of m_firstOrderStats is the numerator of 
  /// equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
  blitz::Array<double,2> m_firstOrderStats;

private:

  /// Cache to avoid re-allocation
  mutable blitz::Array<double,2> m_cache_newMeans;
};

}
}

#endif // KMEANSTRAINER_H
