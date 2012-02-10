/**
 * @file cxx/machine/src/Gaussian.cc
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

#include "machine/Gaussian.h"

#include "core/array_assert.h"
#include "machine/Exception.h"

namespace ca = bob::core::array;
namespace mach = bob::machine;
namespace io = bob::io;

double mach::Log::LogAdd(double log_a, double log_b) {
  double minusdif;

  if (log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }

  minusdif = log_b - log_a;
  //#ifdef DEBUG
  if (std::isnan(minusdif)) {
    printf("LogAdd: minusdif (%f) log_b (%f) or log_a (%f) is nan\n", minusdif, log_b, log_a);
    throw mach::Exception();
  }
  //#endif
  if (minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(exp(minusdif));
}

double mach::Log::LogSub(double log_a, double log_b) {
  double minusdif;

  if (log_a < log_b) {
    printf("LogSub: log_a (%f) should be greater than log_b (%f)", log_a, log_b);
    throw mach::Exception();
  }

  minusdif = log_b - log_a;
  //#ifdef DEBUG
  if (std::isnan(minusdif)) {
    printf("LogSub: minusdif (%f) log_b (%f) or log_a (%f) is nan", minusdif, log_b, log_a);
    throw mach::Exception();
  }
  //#endif
  if (log_a == log_b) return LogZero;
  else if (minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(-exp(minusdif));
}


mach::Gaussian::Gaussian() {
  resize(0);
}

mach::Gaussian::Gaussian(size_t n_inputs) {
  resize(n_inputs);
}

mach::Gaussian::Gaussian(const mach::Gaussian& other) {
  copy(other);
}

mach::Gaussian::Gaussian(io::HDF5File& config) {
  load(config);
}

mach::Gaussian::~Gaussian() {
}

mach::Gaussian& mach::Gaussian::operator=(const mach::Gaussian &other) {
  if(this != &other) 
    copy(other);

  return *this;
}

bool mach::Gaussian::operator==(const mach::Gaussian& b) const {
  return m_n_inputs == b.m_n_inputs &&
         blitz::all(m_mean == b.m_mean) &&
         blitz::all(m_variance == b.m_variance) &&
         blitz::all(m_variance_thresholds == b.m_variance_thresholds);
}


void mach::Gaussian::copy(const mach::Gaussian& other) {
  m_n_inputs = other.m_n_inputs;

  m_mean.resize(m_n_inputs);
  m_mean = other.m_mean;

  m_variance.resize(m_n_inputs);
  m_variance = other.m_variance;

  m_variance_thresholds.resize(m_n_inputs);
  m_variance_thresholds = other.m_variance_thresholds;

  m_n_log2pi = other.m_n_log2pi;
  m_g_norm = other.m_g_norm;
}


void mach::Gaussian::setNInputs(size_t n_inputs) {
  resize(n_inputs);
}

void mach::Gaussian::resize(size_t n_inputs) {
  m_n_inputs = n_inputs;
  m_mean.resize(m_n_inputs);
  m_mean = 0;
  m_variance.resize(m_n_inputs);
  m_variance = 1;
  m_variance_thresholds.resize(m_n_inputs);
  m_variance_thresholds = 0;

  // Re-compute g_norm, because m_n_inputs and m_variance
  // have changed
  preComputeNLog2Pi();
  preComputeConstants();
}

void mach::Gaussian::setMean(const blitz::Array<double,1> &mean) {
  // Check and set
  ca::assertSameShape(m_mean, mean);
  m_mean = mean;
}

void mach::Gaussian::setVariance(const blitz::Array<double,1> &variance) {
  // Check and set
  ca::assertSameShape(m_variance, variance);
  m_variance = variance;

  // Variance flooring
  applyVarianceThresholds();
}

void mach::Gaussian::setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds) {
  // Check and set
  ca::assertSameShape(m_variance_thresholds, variance_thresholds);
  m_variance_thresholds = variance_thresholds;

  // Variance flooring
  applyVarianceThresholds();
}

void mach::Gaussian::setVarianceThresholds(double value) {
  blitz::Array<double,1> variance_thresholds(m_n_inputs);
  variance_thresholds = value;
  setVarianceThresholds(variance_thresholds);
}

void mach::Gaussian::applyVarianceThresholds() {
   // Apply variance flooring threshold
  m_variance = blitz::where( m_variance < m_variance_thresholds, m_variance_thresholds, m_variance);

  // Re-compute g_norm, because m_variance has changed
  preComputeConstants(); 
}

double mach::Gaussian::logLikelihood(const blitz::Array<double,1> &x) const {
  // Check 
  ca::assertSameShape(x, m_mean);
  return logLikelihood_(x);
}

double mach::Gaussian::logLikelihood_(const blitz::Array<double,1> &x) const {
  double z = blitz::sum(blitz::pow2(x - m_mean) / m_variance);
  // Log Likelihood
  return (-0.5 * (m_g_norm + z));
}

void mach::Gaussian::preComputeNLog2Pi() {
  m_n_log2pi = m_n_inputs * mach::Log::Log2Pi;
}

void mach::Gaussian::preComputeConstants() {
  m_g_norm = m_n_log2pi + blitz::sum(blitz::log(m_variance));
}

void mach::Gaussian::save(bob::io::HDF5File& config) const {
  config.setArray("m_mean", m_mean);
  config.setArray("m_variance", m_variance);
  config.setArray("m_variance_thresholds", m_variance_thresholds);
  config.set("g_norm", m_g_norm);
  int64_t v = static_cast<int64_t>(m_n_inputs);
  config.set("m_n_inputs", v);
}

void mach::Gaussian::load(bob::io::HDF5File& config) {
  int64_t v = config.read<int64_t>("m_n_inputs");
  m_n_inputs = static_cast<size_t>(v);
  
  m_mean.resize(m_n_inputs);
  m_variance.resize(m_n_inputs);
  m_variance_thresholds.resize(m_n_inputs);
 
  config.readArray("m_mean", m_mean);
  config.readArray("m_variance", m_variance);
  config.readArray("m_variance_thresholds", m_variance_thresholds);

  preComputeNLog2Pi();
  m_g_norm = config.read<double>("g_norm");
}

namespace bob{
  namespace machine{
    std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
      os << "Mean = " << g.m_mean << std::endl;
      os << "Variance = " << g.m_variance << std::endl;
      return os;
    }
  }
}
