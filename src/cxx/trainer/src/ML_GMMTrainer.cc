#include "trainer/ML_GMMTrainer.h"

Torch::trainer::ML_GMMTrainer::ML_GMMTrainer(bool update_means, bool update_variances, 
    bool update_weights, double mean_var_update_responsibilities_threshold): 
  GMMTrainer(update_means, update_variances, update_weights, mean_var_update_responsibilities_threshold) {

}

Torch::trainer::ML_GMMTrainer::~ML_GMMTrainer() {
  
}

void Torch::trainer::ML_GMMTrainer::mStep(Torch::machine::GMMMachine& gmm, const Torch::io::Arrayset& data) {
  // Read options and variables
  int n_gaussians = gmm.getNGaussians();
  int n_inputs = gmm.getNInputs();

  blitz::firstIndex i;
  blitz::secondIndex j;

  // - Update weights if requested
  //   Equation 9.26 of Bishop, "Pattern recognition and machine learning", 2006
  if (update_weights) {
    m_cache_weights.resize(n_gaussians);
    m_cache_weights = m_ss.n / (int32_t)m_ss.T; //cast req. for linux/32-bits & osx
    gmm.setWeights(m_cache_weights);
  }

  // Generate a thresholded version of m_ss.n
  m_cache_ss_n_thresholded.resize(n_gaussians);
  for(int i=0; i<n_gaussians; ++i) {
    m_cache_ss_n_thresholded(i) = (m_ss.n(i) < m_mean_var_update_responsibilities_threshold? m_mean_var_update_responsibilities_threshold : m_ss.n(i) );
  }

  // Update GMM parameters using the sufficient statistics (m_ss)
  // - Update means if requested
  //   Equation 9.24 of Bishop, "Pattern recognition and machine learning", 2006
  if (update_means) {
    m_cache_means.resize(n_gaussians, n_inputs);
    m_cache_means = m_ss.sumPx(i, j) / m_cache_ss_n_thresholded(i);
    gmm.setMeans(m_cache_means);
  }

  // - Update variance if requested
  //   See Equation 9.25 of Bishop, "Pattern recognition and machine learning", 2006
  //   ...but we use the "computational formula for the variance", i.e.
  //   var = 1/n * sum (P(x-mean)(x-mean))
  //       = 1/n * sum (Pxx) - mean^2
  if (update_variances) {
    m_cache_means.resize(n_gaussians, n_inputs);
    gmm.getMeans(m_cache_means);
    m_cache_variances.resize(n_gaussians, n_inputs);
    m_cache_variances = m_ss.sumPxx(i, j) / m_cache_ss_n_thresholded(i) - blitz::pow2(m_cache_means(i, j));
    gmm.setVariances(m_cache_variances);
  }
}

