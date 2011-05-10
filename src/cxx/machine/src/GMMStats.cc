#include "machine/GMMStats.h"

#include "core/logging.h"

Torch::machine::GMMStats::GMMStats() {
  resize(0,0);
}

Torch::machine::GMMStats::GMMStats(int n_gaussians, int n_inputs) {
  resize(n_gaussians,n_inputs);
}

Torch::machine::GMMStats::~GMMStats() {
  
}

void Torch::machine::GMMStats::resize(int n_gaussians, int n_inputs) {
  n.resize(n_gaussians);
  sumPx.resize(n_gaussians,n_inputs);
  sumPxx.resize(n_gaussians,n_inputs);
  init();
}

bool Torch::machine::GMMStats::init() {
  log_likelihood = 0;
  T = 0;
  n = 0.0;
  sumPx = 0.0;
  sumPxx = 0.0;
  return true;
}

bool Torch::machine::GMMStats::print() {
  Torch::core::info << "log_likelihood = " << log_likelihood << std::endl;
  Torch::core::info << "T = " << T << std::endl;
  Torch::core::info << "n = " << n;
  Torch::core::info << "sumPx = " << sumPx;
  Torch::core::info << "sumPxx = " << sumPxx;
  return true;
}
