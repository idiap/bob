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

Torch::machine::GMMStats::GMMStats(Torch::config::Configuration& config) {
  load(config);
}

Torch::machine::GMMStats::~GMMStats() {
  
}

void Torch::machine::GMMStats::resize(int n_gaussians, int n_inputs) {
  n.resize(n_gaussians);
  sumPx.resize(n_gaussians, n_inputs);
  sumPxx.resize(n_gaussians, n_inputs);
  init();
}

void Torch::machine::GMMStats::init() {
  log_likelihood = 0;
  T = 0;
  n = 0.0;
  sumPx = 0.0;
  sumPxx = 0.0;
}

void Torch::machine::GMMStats::save(Torch::config::Configuration& config) {
  Torch::database::Array arrayn(n);
  Torch::database::Array arraySumPx(sumPx);
  Torch::database::Array arraySumPxx(sumPxx);

  config.set("n_gaussians", sumPx.shape()[0]);
  config.set("n_inputs", sumPx.shape()[1]);
  config.set("log_liklihood", log_likelihood);
  config.set("T", T);
  config.set("n", arrayn);
  config.set("sumPx", arraySumPx);
  config.set("sumPxx", arraySumPxx);
}

void Torch::machine::GMMStats::load(const Torch::config::Configuration& config) {
  log_likelihood = config.get<std::vector<double> >("log_liklihood").at(0);
  int n_gaussians = (int)config.get<std::vector<int64_t> >("n_gaussians").at(0);
  int n_inputs = (int)config.get<std::vector<int64_t> >("n_inputs").at(0);
  T = (int)config.get<std::vector<int64_t> >("T").at(0);

  n.resize(n_gaussians);
  sumPx.resize(n_gaussians, n_inputs);
  sumPxx.resize(n_gaussians, n_inputs);
  
  n = config.get<Torch::database::Arrayset>("n").get<double, 1>(1);
  sumPx = config.get<Torch::database::Arrayset>("sumPx").get<double, 2>(1);
  sumPxx = config.get<Torch::database::Arrayset>("sumPxx").get<double, 2>(1);
}

namespace Torch {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const GMMStats& g) {
      os << "log_likelihood = " << g.log_likelihood << std::endl;
      os << "T = " << g.T << std::endl;
      os << "n = " << g.n;
      os << "sumPx = " << g.sumPx;
      os << "sumPxx = " << g.sumPxx;
      
      return os;
    }
  }
}