/**
 * @file machine/cxx/Gaussian.cc
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/machine/Gaussian.h"

#include "bob/core/assert.h"
#include "bob/math/log.h"

bob::machine::Gaussian::Gaussian() {
  resize(0);
}

bob::machine::Gaussian::Gaussian(const size_t n_inputs) {
  resize(n_inputs);
}

bob::machine::Gaussian::Gaussian(const bob::machine::Gaussian& other) {
  copy(other);
}

bob::machine::Gaussian::Gaussian(bob::io::HDF5File& config) {
  load(config);
}

bob::machine::Gaussian::~Gaussian() {
}

bob::machine::Gaussian& bob::machine::Gaussian::operator=(const bob::machine::Gaussian &other) {
  if(this != &other)
    copy(other);

  return *this;
}

bool bob::machine::Gaussian::operator==(const bob::machine::Gaussian& b) const {
  // Check dimensions
  if(this->m_mean.extent(0) != b.m_mean.extent(0) ||
     this->m_variance.extent(0) != b.m_variance.extent(0) ||
     this->m_variance_thresholds.extent(0) != b.m_variance_thresholds.extent(0))
    return false;

  // Check content
  if(this->m_n_inputs != b.m_n_inputs || blitz::any(this->m_mean != b.m_mean) ||
     blitz::any(this->m_variance != b.m_variance) ||
     blitz::any(this->m_variance_thresholds != b.m_variance_thresholds))
    return false;

  return true;
}

bool bob::machine::Gaussian::operator!=(const bob::machine::Gaussian& b) const {
  return !(this->operator==(b));
}

bool bob::machine::Gaussian::is_similar_to(const bob::machine::Gaussian& b, const double epsilon) const {
  // Check dimensions
  if (m_mean.extent(0) != b.m_mean.extent(0) ||
      m_variance.extent(0) != b.m_variance.extent(0) ||
      m_variance_thresholds.extent(0) != b.m_variance_thresholds.extent(0))
    return false;

  // Check content
  if (m_n_inputs != b.m_n_inputs || blitz::any(blitz::abs(m_mean - b.m_mean) > epsilon) ||
      blitz::any(blitz::abs(m_variance - b.m_variance) > epsilon) ||
      blitz::any(blitz::abs(m_variance_thresholds - b.m_variance_thresholds) > epsilon))
    return false;

  return true;
}

void bob::machine::Gaussian::copy(const bob::machine::Gaussian& other) {
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


void bob::machine::Gaussian::setNInputs(const size_t n_inputs) {
  resize(n_inputs);
}

void bob::machine::Gaussian::resize(const size_t n_inputs) {
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

void bob::machine::Gaussian::setMean(const blitz::Array<double,1> &mean) {
  // Check and set
  bob::core::array::assertSameShape(m_mean, mean);
  m_mean = mean;
}

void bob::machine::Gaussian::setVariance(const blitz::Array<double,1> &variance) {
  // Check and set
  bob::core::array::assertSameShape(m_variance, variance);
  m_variance = variance;

  // Variance flooring
  applyVarianceThresholds();
}

void bob::machine::Gaussian::setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds) {
  // Check and set
  bob::core::array::assertSameShape(m_variance_thresholds, variance_thresholds);
  m_variance_thresholds = variance_thresholds;

  // Variance flooring
  applyVarianceThresholds();
}

void bob::machine::Gaussian::setVarianceThresholds(const double value) {
  blitz::Array<double,1> variance_thresholds(m_n_inputs);
  variance_thresholds = value;
  setVarianceThresholds(variance_thresholds);
}

void bob::machine::Gaussian::applyVarianceThresholds() {
   // Apply variance flooring threshold
  m_variance = blitz::where( m_variance < m_variance_thresholds, m_variance_thresholds, m_variance);

  // Re-compute g_norm, because m_variance has changed
  preComputeConstants();
}

double bob::machine::Gaussian::logLikelihood(const blitz::Array<double,1> &x) const {
  // Check
  bob::core::array::assertSameShape(x, m_mean);
  return logLikelihood_(x);
}

double bob::machine::Gaussian::logLikelihood_(const blitz::Array<double,1> &x) const {
  double z = blitz::sum(blitz::pow2(x - m_mean) / m_variance);
  // Log Likelihood
  return (-0.5 * (m_g_norm + z));
}

void bob::machine::Gaussian::preComputeNLog2Pi() {
  m_n_log2pi = m_n_inputs * bob::math::Log::Log2Pi;
}

void bob::machine::Gaussian::preComputeConstants() {
  m_g_norm = m_n_log2pi + blitz::sum(blitz::log(m_variance));
}

void bob::machine::Gaussian::save(bob::io::HDF5File& config) const {
  config.setArray("m_mean", m_mean);
  config.setArray("m_variance", m_variance);
  config.setArray("m_variance_thresholds", m_variance_thresholds);
  config.set("g_norm", m_g_norm);
  int64_t v = static_cast<int64_t>(m_n_inputs);
  config.set("m_n_inputs", v);
}

void bob::machine::Gaussian::load(bob::io::HDF5File& config) {
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
