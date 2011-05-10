#include "trainer/GMMTrainer.h"

using namespace Torch::machine;

Torch::trainer::GMMTrainer::GMMTrainer(bool update_means, bool update_variances, bool update_weights) :
  EMTrainer<GMMMachine, FrameSample>(), update_means(update_means), update_variances(update_variances), update_weights(update_weights) {
}

Torch::trainer::GMMTrainer::~GMMTrainer() {
  
}

void Torch::trainer::GMMTrainer::initialization(Torch::machine::GMMMachine& gmm, const Sampler<FrameSample>& data) {
  // Allocate memory for the sufficient statistics and initialise
  m_ss.resize(gmm.getNGaussians(),gmm.getNInputs());
}

double Torch::trainer::GMMTrainer::eStep(Torch::machine::GMMMachine& gmm, const Sampler<FrameSample>& data) {
  // Calculate the sufficient statistics and save in m_ss
  gmm.accStatistics(data, m_ss);
  return m_ss.log_likelihood / m_ss.T;
}
