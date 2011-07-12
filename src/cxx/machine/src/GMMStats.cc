#include "machine/GMMStats.h"

#include <core/logging.h>

Torch::machine::GMMStats::GMMStats() {
  resize(0,0);
}

Torch::machine::GMMStats::GMMStats(int n_gaussians, int n_inputs) {
  resize(n_gaussians,n_inputs);
}

Torch::machine::GMMStats::GMMStats(Torch::io::HDF5File& config) {
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

void Torch::machine::GMMStats::save(Torch::io::HDF5File& config) const {
  //please note we fix the output values to be of a precise type so they can be
  //retrieved at any platform with the exact same precision.
  int64_t sumpx_shape_0 = sumPx.shape()[0];
  int64_t sumpx_shape_1 = sumPx.shape()[1];
  config.set("n_gaussians", sumpx_shape_0);
  config.set("n_inputs", sumpx_shape_1);
  config.set("log_liklihood", log_likelihood); //double
  config.set("T", T);
  config.setArray("n", n); //Array1d
  config.setArray("sumPx", sumPx); //Array2d
  config.setArray("sumPxx", sumPxx); //Array2d
}

void Torch::machine::GMMStats::load(Torch::io::HDF5File& config) {
  config.read("log_liklihood", log_likelihood);
  int64_t n_gaussians;
  config.read("n_gaussians", n_gaussians);
  int64_t n_inputs;
  config.read("n_inputs", n_inputs);
  config.read("T", T);
  
  //resize arrays to prepare for HDF5 readout
  n.resize(n_gaussians);
  sumPx.resize(n_gaussians, n_inputs);
  sumPxx.resize(n_gaussians, n_inputs);
  
  //load data
  config.readArray("n", n);
  config.readArray("sumPx", sumPx);
  config.readArray("sumPxx", sumPxx);
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
