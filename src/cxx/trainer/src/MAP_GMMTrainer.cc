#include "trainer/MAP_GMMTrainer.h"

Torch::trainer::MAP_GMMTrainer::MAP_GMMTrainer(double relevance_factor) : GMMTrainer(), relevance_factor(relevance_factor), m_prior_gmm(NULL) {
  
}

Torch::trainer::MAP_GMMTrainer::~MAP_GMMTrainer() {
  
}

bool Torch::trainer::MAP_GMMTrainer::setPriorGMM(Torch::machine::GMMMachine *prior_gmm) {
  if (prior_gmm == NULL) return false;
  m_prior_gmm = prior_gmm;
  return true;
}

void Torch::trainer::MAP_GMMTrainer::mStep(Torch::machine::GMMMachine& gmm, const Sampler<Torch::machine::FrameSample>& data) {
  // Read options and variables
  double n_gaussians = gmm.getNGaussians();
  double n_inputs = gmm.getNInputs();
  
  // Check that the prior GMM has been specified
  if (m_prior_gmm == NULL) {
    Torch::core::error << "Cannot do MAP mStep() because prior GMM has not been set." << std::endl;
    throw Torch::core::Exception();
  }

  blitz::firstIndex i;
  blitz::secondIndex j;

  // Calculate the "data-dependent adaptation coefficient", alpha_i
  blitz::Array<double,1> alpha(n_gaussians);
  alpha = m_ss.n(i) / (m_ss.n(i) + relevance_factor);

  // - Update weights if requested
  //   Equation 11 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_weights) {

    // Calculate the maximum likelihood weights
    blitz::Array<double,1> ml_weights(n_gaussians);
    ml_weights = m_ss.n / m_ss.T;

    // Get the prior weights
    blitz::Array<double,1> prior_weights(n_gaussians);
    m_prior_gmm->getWeights(prior_weights);

    // Calculate the new weights
    blitz::Array<double,1> new_weights(n_gaussians);
    new_weights = alpha * ml_weights + (1-alpha) * prior_weights;

    // Apply the scale factor, gamma, to ensure the new weights sum to unity 
    double gamma = blitz::sum(new_weights);
    new_weights /= gamma;

    // Set the new weights
    gmm.setWeights(new_weights);
  }

  // Update GMM parameters
  // - Update means if requested
  //   Equation 12 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_means) {

    // Calculate the maximum likelihood means
    blitz::Array<double,2> ml_means(n_gaussians,n_inputs);
    ml_means = m_ss.sumPx(i,j) / m_ss.n(i);

    // Get the prior means
    blitz::Array<double,2> prior_means(n_gaussians,n_inputs);
    m_prior_gmm->getMeans(prior_means);

    // Calculate new means
    blitz::Array<double,2> new_means(n_gaussians,n_inputs);
    new_means = alpha(i) * ml_means(i,j) + (1-alpha(i)) * prior_means(i,j);

    // Set the new means
    gmm.setMeans(new_means);
  }

  // - Update variance if requested
  //   Equation 13 of Reynolds et al., "Speaker Verification Using Adapted Gaussian Mixture Models", Digital Signal Processing, 2000
  if (update_variances) {

    // Get the prior variances
    blitz::Array<double,2> prior_variances(n_gaussians,n_inputs);
    m_prior_gmm->getVariances(prior_variances);

    // Calculate E_i(x^2) (equation 10)
    blitz::Array<double,2> Exx(n_gaussians,n_inputs);
    Exx = m_ss.sumPxx(i,j) / m_ss.n(i);

    // Get means and prior means
    blitz::Array<double,2> means; 
    gmm.getMeans(means);
    blitz::Array<double,2> prior_means; 
    m_prior_gmm->getMeans(prior_means);

    // Calculate new variances (equation 13)
    blitz::Array<double,2> new_variances(n_gaussians,n_inputs);
    new_variances = alpha(i) * Exx(i,j) + (1-alpha(i)) * (prior_variances(i,j) + prior_means(i,j)) - blitz::pow2(means(i,j));

    // Set the new variances
    gmm.setVariances(new_variances);
  }
}