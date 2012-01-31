/**
 * @file cxx/trainer/src/ML_GMMTrainer.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "trainer/ML_GMMTrainer.h"

#include <algorithm>

namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;

train::ML_GMMTrainer::ML_GMMTrainer(bool update_means, bool update_variances, 
    bool update_weights, double mean_var_update_responsibilities_threshold): 
  train::GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {

}

train::ML_GMMTrainer::~ML_GMMTrainer() {
  
}

void train::ML_GMMTrainer::initialization(mach::GMMMachine& gmm, const io::Arrayset& data) {
  train::GMMTrainer::initialization(gmm, data);
  // Allocate cache
  size_t n_gaussians = gmm.getNGaussians();
  m_cache_ss_n_thresholded.resize(n_gaussians);
}


void train::ML_GMMTrainer::mStep(mach::GMMMachine& gmm, const io::Arrayset& data) {
  // Read options and variables
  size_t n_gaussians = gmm.getNGaussians();

  // TODO: to keep or not?
  m_cache_ss_n_thresholded.resize(n_gaussians);

  // - Update weights if requested
  //   Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006
  if (update_weights) {
    blitz::Array<double,1>& weights = gmm.updateWeights();
    weights = m_ss.n / static_cast<double>(m_ss.T); //cast req. for linux/32-bits & osx
  }

  // Generate a thresholded version of m_ss.n
  for(size_t i=0; i<n_gaussians; ++i)
    m_cache_ss_n_thresholded(i) = std::max(m_ss.n(i), m_mean_var_update_responsibilities_threshold);  

  // Update GMM parameters using the sufficient statistics (m_ss)
  // - Update means if requested
  //   Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006
  if (update_means) {
    for(size_t i=0; i<n_gaussians; ++i) {
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      means = m_ss.sumPx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i);
    }   
  }

  // - Update variance if requested
  //   See Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006
  //   ...but we use the "computational formula for the variance", i.e.
  //   var = 1/n * sum (P(x-mean)(x-mean))
  //       = 1/n * sum (Pxx) - mean^2
  if (update_variances) {
    for(size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& means = gmm.getGaussian(i)->getMean();
      blitz::Array<double,1>& variances = gmm.getGaussian(i)->updateVariance();
      variances = m_ss.sumPxx(i, blitz::Range::all()) / m_cache_ss_n_thresholded(i) - blitz::pow2(means);
      gmm.getGaussian(i)->applyVarianceThresholds();
    }
  }
}

