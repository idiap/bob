#include "machine/GMMStats.h"

#include <core/logging.h>
#include <database/Array.h>
#include <database/Arrayset.h>

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

void Torch::machine::GMMStats::save(Torch::config::Configuration& config) {
  Torch::database::Array arrayn(n);
  Torch::database::Array arraySumPx(sumPx);
  Torch::database::Array arraySumPxx(sumPxx);
  
  config.set("log_liklihood", log_likelihood);
  config.set("T", T);
  config.set("n", arrayn);
  config.set("sumPx", arraySumPx);
  config.set("sumPxx", arraySumPxx);
}

void Torch::machine::GMMStats::load(const Torch::config::Configuration& config) {
  log_likelihood = config.get<std::vector<double> >("log_liklihood").at(0);

  T = (int)config.get<std::vector<long> >("T").at(0);
  n = config.get<Torch::database::Arrayset>("n").get<double, 1>(1);
  sumPx = config.get<Torch::database::Arrayset>("sumPx").get<double, 2>(1);
  sumPxx = config.get<Torch::database::Arrayset>("sumPxx").get<double, 2>(1);
}