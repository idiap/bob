/**
 * @file trainer/cxx/MAP_GMMTrainer.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
#include "bob/trainer/MAP_GMMTrainer.h"
#include "bob/trainer/Exception.h"

namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;

train::MAP_GMMTrainer::MAP_GMMTrainer(double relevance_factor, bool update_means, bool update_variances, 
    bool update_weights, double mean_var_update_responsibilities_threshold): 
  GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold), 
  relevance_factor(relevance_factor), m_prior_gmm(boost::shared_ptr<mach::GMMMachine>()), m_T3_alpha(0.), m_T3_adaptation(false) {
  
}

train::MAP_GMMTrainer::~MAP_GMMTrainer() {
  
}

void train::MAP_GMMTrainer::initialization(mach::GMMMachine& gmm, const blitz::Array<double,2>& data) {
  // Allocate memory for the sufficient statistics and initialise
  train::GMMTrainer::initialization(gmm, data);

  size_t n_gaussians = gmm.getNGaussians();
  // TODO: check size?
  gmm.setWeights(m_prior_gmm->getWeights());
  for(size_t i=0; i<n_gaussians; ++i)
  {
    gmm.getGaussian(i)->updateMean() = m_prior_gmm->getGaussian(i)->getMean();
    gmm.getGaussian(i)->updateVariance() = m_prior_gmm->getGaussian(i)->getVariance();
    gmm.getGaussian(i)->applyVarianceThresholds();
  }
  // Initializes cache
  m_cache_alpha.resize(n_gaussians);
  m_cache_ml_weights.resize(n_gaussians);
}

bool train::MAP_GMMTrainer::setPriorGMM(boost::shared_ptr<bob::machine::GMMMachine> prior_gmm) {
  if(!prior_gmm) return false;
  m_prior_gmm = prior_gmm;
  return true;
}

void train::MAP_GMMTrainer::mStep(mach::GMMMachine& gmm, const blitz::Array<double,2>& data) {
  // Read options and variables
  double n_gaussians = gmm.getNGaussians();
  
  // Check that the prior GMM has been specified
  if (!m_prior_gmm) {
    throw NoPriorGMM();
  }

  blitz::firstIndex i;
  blitz::secondIndex j;

  // Calculate the "data-dependent adaptation coefficient", alpha_i
  // TODO: check if required // m_cache_alpha.resize(n_gaussians);
  if( m_T3_adaptation )
    m_cache_alpha = m_T3_alpha;
  else
    m_cache_alpha = m_ss.n(i) / (m_ss.n(i) + relevance_factor);

  // - Update weights if requested
  //   Equation 11 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_weights) {
    // Calculate the maximum likelihood weights
    m_cache_ml_weights = m_ss.n / static_cast<double>(m_ss.T); //cast req. for linux/32-bits & osx

    // Get the prior weights
    const blitz::Array<double,1>& prior_weights = m_prior_gmm->getWeights();
    blitz::Array<double,1>& new_weights = gmm.updateWeights();

    // Calculate the new weights
    new_weights = m_cache_alpha * m_cache_ml_weights + (1-m_cache_alpha) * prior_weights;

    // Apply the scale factor, gamma, to ensure the new weights sum to unity 
    double gamma = blitz::sum(new_weights);
    new_weights /= gamma;

    // Recompute the log weights in the cache of the GMMMachine
    gmm.recomputeLogWeights();
  }

  // Update GMM parameters
  // - Update means if requested
  //   Equation 12 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_means) {
    // Calculate new means
    for(size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& prior_means = m_prior_gmm->getGaussian(i)->getMean();
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      if(m_ss.n(i) < m_mean_var_update_responsibilities_threshold) {
        means = prior_means;
      }
      else {
        // Use the maximum likelihood means
        means = m_cache_alpha(i) * (m_ss.sumPx(i,blitz::Range::all()) / m_ss.n(i)) + (1-m_cache_alpha(i)) * prior_means;
      }
    }
  }

  // - Update variance if requested
  //   Equation 13 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_variances) {
    // Calculate new variances (equation 13)
    for(size_t i=0; i<n_gaussians; ++i) {
      const blitz::Array<double,1>& prior_means = m_prior_gmm->getGaussian(i)->getMean();
      blitz::Array<double,1>& means = gmm.getGaussian(i)->updateMean();
      const blitz::Array<double,1>& prior_variances = m_prior_gmm->getGaussian(i)->getVariance();
      blitz::Array<double,1>& variances = gmm.getGaussian(i)->updateVariance();
      if(m_ss.n(i) < m_mean_var_update_responsibilities_threshold) {
        variances = (prior_variances + prior_means) - blitz::pow2(means);
      }
      else {
        variances = m_cache_alpha(i) * m_ss.sumPxx(i,blitz::Range::all()) / m_ss.n(i) + (1-m_cache_alpha(i)) * (prior_variances + prior_means) - blitz::pow2(means);
      }
      gmm.getGaussian(i)->applyVarianceThresholds();
    }
  }
}
