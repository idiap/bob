/**
 * @file cxx/machine/src/GMMStats.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include "machine/GMMStats.h"
#include "machine/Exception.h"
#include "core/logging.h"

namespace mach = bob::machine;
namespace io = bob::io;

mach::GMMStats::GMMStats() {
  resize(0,0);
}

mach::GMMStats::GMMStats(const size_t n_gaussians, const size_t n_inputs) {
  resize(n_gaussians,n_inputs);
}

mach::GMMStats::GMMStats(io::HDF5File& config) {
  load(config);
}

mach::GMMStats::GMMStats(const mach::GMMStats& other) {
  copy(other);
}

mach::GMMStats::~GMMStats() { 
}

mach::GMMStats& mach::GMMStats::operator=(const mach::GMMStats& other) {
  // protect against invalid self-assignment
  if(this != &other) 
    copy(other);
  
  // by convention, always return *this
  return *this;
}

bool mach::GMMStats::operator==(const mach::GMMStats& b) const {
  // Check dimensions
  if(n.extent(0) != b.n.extent(0) || 
      sumPx.extent(0) != b.sumPx.extent(0) || sumPx.extent(1) != b.sumPx.extent(1) ||
      sumPxx.extent(0) != b.sumPxx.extent(0) || sumPxx.extent(1) != b.sumPxx.extent(1))
    return false;
  
  // Check content
  if(T != b.T || log_likelihood != b.log_likelihood || blitz::any(n != b.n) || 
      blitz::any(sumPx != b.sumPx) || blitz::any(sumPxx != b.sumPxx))
    return false;
  
  return true;
} 

void mach::GMMStats::operator+=(const mach::GMMStats& b) {
  // Check dimensions
  if(n.extent(0) != b.n.extent(0) || 
      sumPx.extent(0) != b.sumPx.extent(0) || sumPx.extent(1) != b.sumPx.extent(1) ||
      sumPxx.extent(0) != b.sumPxx.extent(0) || sumPxx.extent(1) != b.sumPxx.extent(1))
    // TODO: add a specialized exception
    throw mach::Exception();

  // Update GMMStats object with the content of the other one
  T += b.T;
  log_likelihood += b.log_likelihood;
  n += b.n;
  sumPx += b.sumPx;
  sumPxx += b.sumPxx;
}

void mach::GMMStats::copy(const GMMStats& other) {
  // Resize arrays
  resize(other.sumPx.extent(0),other.sumPx.extent(1));
  // Copy content
  T = other.T;
  log_likelihood = other.log_likelihood;
  n = other.n;
  sumPx = other.sumPx;
  sumPxx = other.sumPxx;
}

void mach::GMMStats::resize(const size_t n_gaussians, const size_t n_inputs) {
  n.resize(n_gaussians);
  sumPx.resize(n_gaussians, n_inputs);
  sumPxx.resize(n_gaussians, n_inputs);
  init();
}

void mach::GMMStats::init() {
  log_likelihood = 0;
  T = 0;
  n = 0.0;
  sumPx = 0.0;
  sumPxx = 0.0;
}

void mach::GMMStats::save(io::HDF5File& config) const {
  //please note we fix the output values to be of a precise type so they can be
  //retrieved at any platform with the exact same precision.
  // TODO: add versioning, replace int64_t by uint64_t and log_liklihood by log_likelihood
  int64_t sumpx_shape_0 = sumPx.shape()[0];
  int64_t sumpx_shape_1 = sumPx.shape()[1];
  config.set("n_gaussians", sumpx_shape_0);
  config.set("n_inputs", sumpx_shape_1);
  config.set("log_liklihood", log_likelihood); //double
  config.set("T", static_cast<int64_t>(T));
  config.setArray("n", n); //Array1d
  config.setArray("sumPx", sumPx); //Array2d
  config.setArray("sumPxx", sumPxx); //Array2d
}

void mach::GMMStats::load(io::HDF5File& config) {
  log_likelihood = config.read<double>("log_liklihood");
  int64_t n_gaussians = config.read<int64_t>("n_gaussians");
  int64_t n_inputs = config.read<int64_t>("n_inputs");
  T = static_cast<size_t>(config.read<int64_t>("T"));
  
  //resize arrays to prepare for HDF5 readout
  n.resize(n_gaussians);
  sumPx.resize(n_gaussians, n_inputs);
  sumPxx.resize(n_gaussians, n_inputs);
  
  //load data
  config.readArray("n", n);
  config.readArray("sumPx", sumPx);
  config.readArray("sumPxx", sumPxx);
}

namespace bob {
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
