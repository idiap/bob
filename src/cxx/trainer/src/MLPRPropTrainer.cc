/**
 * @file cxx/trainer/src/MLPRPropTrainer.cc
 * @date Mon Jul 11 16:19:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of the RProp algorithm for MLP training.
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

#include <algorithm>
#include "core/array_check.h"
#include "core/array_copy.h"
#include "math/linear.h"
#include "machine/MLPException.h"
#include "trainer/Exception.h"
#include "trainer/MLPRPropTrainer.h"

namespace array = bob::core::array;
namespace mach = bob::machine;
namespace math = bob::math;
namespace train = bob::trainer;

train::MLPRPropTrainer::MLPRPropTrainer(const mach::MLP& machine,
    size_t batch_size):
  m_train_bias(true),
  m_H(machine.numOfHiddenLayers()), ///< handy!
  m_weight_ref(m_H + 1),
  m_bias_ref(m_H + 1),
  m_delta(m_H + 1),
  m_delta_bias(m_H + 1),
  m_deriv(m_H + 1),
  m_deriv_bias(m_H + 1),
  m_prev_deriv(m_H + 1),
  m_prev_deriv_bias(m_H + 1),
  m_actfun(machine.getActivationFunction()),
  m_bwdfun(),
  m_target(),
  m_error(m_H + 1),
  m_output(m_H + 2)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_delta[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_delta_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_prev_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_prev_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();

  switch (machine.getActivation()) {
    case mach::LINEAR:
      m_bwdfun = mach::linear_derivative;
      break;
    case mach::TANH:
      m_bwdfun = mach::tanh_derivative;
      break;
    case mach::LOG:
      m_bwdfun = mach::logistic_derivative;
      break;
    default:
      throw mach::UnsupportedActivation(machine.getActivation());
  }

  setBatchSize(batch_size);
}

train::MLPRPropTrainer::~MLPRPropTrainer() { }

train::MLPRPropTrainer::MLPRPropTrainer(const MLPRPropTrainer& other):
  m_train_bias(other.m_train_bias),
  m_H(other.m_H),
  m_weight_ref(m_H + 1),
  m_bias_ref(m_H + 1),
  m_delta(m_H + 1),
  m_delta_bias(m_H + 1),
  m_deriv(m_H + 1),
  m_deriv_bias(m_H + 1),
  m_prev_deriv(m_H + 1),
  m_prev_deriv_bias(m_H + 1),
  m_actfun(other.m_actfun),
  m_bwdfun(other.m_bwdfun),
  m_target(bob::core::array::ccopy(other.m_target)),
  m_error(m_H + 1),
  m_output(m_H + 2)
{
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k].reference(bob::core::array::ccopy(other.m_delta[k]));
    m_delta_bias[k].reference(bob::core::array::ccopy(other.m_delta_bias[k]));
    m_deriv[k].reference(bob::core::array::ccopy(other.m_deriv[k]));
    m_deriv_bias[k].reference(bob::core::array::ccopy(other.m_deriv_bias[k]));
    m_prev_deriv[k].reference(bob::core::array::ccopy(other.m_prev_deriv[k]));
    m_prev_deriv_bias[k].reference(bob::core::array::ccopy(other.m_prev_deriv_bias[k]));
    m_error[k].reference(bob::core::array::ccopy(other.m_error[k]));
    m_output[k].reference(bob::core::array::ccopy(other.m_output[k]));
  }
  m_output[m_H + 1].reference(bob::core::array::ccopy(other.m_output[m_H + 1]));
}

train::MLPRPropTrainer& train::MLPRPropTrainer::operator=
(const train::MLPRPropTrainer::MLPRPropTrainer& other) {
  m_train_bias = other.m_train_bias;
  m_H = other.m_H;
  m_weight_ref.resize(m_H + 1);
  m_bias_ref.resize(m_H + 1);
  m_delta.resize(m_H + 1);
  m_delta_bias.resize(m_H + 1);
  m_deriv.resize(m_H + 1);
  m_deriv_bias.resize(m_H + 1);
  m_prev_deriv.resize(m_H + 1);
  m_prev_deriv_bias.resize(m_H + 1);
  m_actfun = other.m_actfun;
  m_bwdfun = other.m_bwdfun;
  m_target.reference(bob::core::array::ccopy(other.m_target));
  m_error.resize(m_H + 1);
  m_output.resize(m_H + 2);

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k].reference(bob::core::array::ccopy(other.m_delta[k]));
    m_delta_bias[k].reference(bob::core::array::ccopy(other.m_delta_bias[k]));
    m_deriv[k].reference(bob::core::array::ccopy(other.m_deriv[k]));
    m_deriv_bias[k].reference(bob::core::array::ccopy(other.m_deriv_bias[k]));
    m_prev_deriv[k].reference(bob::core::array::ccopy(other.m_prev_deriv[k]));
    m_prev_deriv_bias[k].reference(bob::core::array::ccopy(other.m_prev_deriv_bias[k]));
    m_error[k].reference(bob::core::array::ccopy(other.m_error[k]));
    m_output[k].reference(bob::core::array::ccopy(other.m_output[k]));
  }
  m_output[m_H + 1].reference(bob::core::array::ccopy(other.m_output[m_H + 1]));

  return *this;
}

void train::MLPRPropTrainer::reset() {
  static const double DELTA0 = 0.1; ///< taken from the paper, section II.C

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k] = DELTA0;
    m_delta_bias[k] = DELTA0;
    m_prev_deriv[k] = 0;
    m_prev_deriv_bias[k] = 0;
  }
}

void train::MLPRPropTrainer::setBatchSize (size_t batch_size) {
  // m_output: values after the activation function; note that "output" will
  //           accomodate the input to ease on the calculations
  // m_target: sampled target values
  // m_error: error values;
  
  m_target.resize(batch_size, m_deriv.back().extent(1));

  m_output[0].resize(batch_size, m_deriv[0].extent(0));

  for (size_t k=1; k<m_output.size(); ++k) {
    m_output[k].resize(batch_size, m_deriv[k-1].extent(1));
  }

  for (size_t k=0; k<m_error.size(); ++k) {
    m_error[k].resize(batch_size, m_deriv[k].extent(1));
  }
}

bool train::MLPRPropTrainer::isCompatible(const mach::MLP& machine) const 
{
  if (m_H != machine.numOfHiddenLayers()) return false;
  
  if (m_target.extent(1) != (int)machine.outputSize()) return false;

  if (m_output[0].extent(1) != (int)machine.inputSize()) return false;

  //also, each layer should be of the same size
  for (size_t k=0; k<(m_H + 1); ++k) {
    if (!array::hasSameShape(m_deriv[k], machine.getWeights()[k])) return false;
  }

  //if you get to this point, you can only return true
  return true;
}

void train::MLPRPropTrainer::forward_step() {
  size_t batch_size = m_target.extent(0);
  for (size_t k=0; k<m_weight_ref.size(); ++k) { //for all layers
    math::prod_(m_output[k], m_weight_ref[k], m_output[k+1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<m_output[k+1].extent(1); ++j) { //for all variables
        m_output[k+1](i,j) = m_actfun(m_output[k+1](i,j) + m_bias_ref[k](j));
      }
    }
  }
}

void train::MLPRPropTrainer::backward_step() {
  size_t batch_size = m_target.extent(0);
  //last layer
  m_error[m_H] = m_output.back() - m_target;
  for (int i=0; i<(int)batch_size; ++i) { //for every example
    for (int j=0; j<m_error[m_H].extent(1); ++j) { //for all variables
      m_error[m_H](i,j) *= m_bwdfun(m_output[m_H+1](i,j));
    }
  }

  //all other layers
  for (size_t k=m_H; k>0; --k) {
    math::prod_(m_error[k], m_weight_ref[k].transpose(1,0), m_error[k-1]);
    for (int i=0; i<(int)batch_size; ++i) { //for every example
      for (int j=0; j<m_error[k-1].extent(1); ++j) { //for all variables
        m_error[k-1](i,j) *= m_bwdfun(m_output[k](i,j));
      }
    }
  }
}

/**
 * A function that returns the sign of a double number (zero if the value is
 * 0).
 */
static int8_t sign (double x) {
  if (x > 0) return +1;
  return (x == 0)? 0 : -1;
}

void train::MLPRPropTrainer::rprop_weight_update() {
  // constants taken from the paper.
  static const double ETA_MINUS = 0.5;
  static const double ETA_PLUS = 1.2;
  static const double DELTA_MAX = 50.0;
  static const double DELTA_MIN = 1e-6;

  for (size_t k=0; k<m_weight_ref.size(); ++k) { //for all layers
    math::prod_(m_output[k].transpose(1,0), m_error[k], m_deriv[k]);

    // Note that we don't need to estimate the mean since we are only
    // interested in the sign of the derivative and dividing by the mean makes
    // no difference on the final result as 'batch_size' is always > 0!
    // deriv[k] /= batch_size; //estimates the mean for the batch

    // Calculates the sign change as prescribed on the RProp paper. Depending
    // on the sign change, we update the "weight_update" matrix and apply the
    // updates on the respective weights.
    for (int i=0; i<m_deriv[k].extent(0); ++i) {
      for (int j=0; j<m_deriv[k].extent(1); ++j) {
        int8_t M = sign(m_deriv[k](i,j) * m_prev_deriv[k](i,j));
        // Implementations equations (4-6) on the RProp paper:
        if (M > 0) {
          m_delta[k](i,j) = std::min(m_delta[k](i,j)*ETA_PLUS, DELTA_MAX); 
          m_weight_ref[k](i,j) -= sign(m_deriv[k](i,j)) * m_delta[k](i,j); 
          m_prev_deriv[k](i,j) = m_deriv[k](i,j);
        }
        else if (M < 0) {
          m_delta[k](i,j) = std::max(m_delta[k](i,j)*ETA_MINUS, DELTA_MIN);
          m_prev_deriv[k](i,j) = 0;
        }
        else { //M == 0
          m_weight_ref[k](i,j) -= sign(m_deriv[k](i,j)) * m_delta[k](i,j);
          m_prev_deriv[k](i,j) = m_deriv[k](i,j);
        }
      }
    }

    // Here we decide if we should train the biases or not
    if (!m_train_bias) continue;

    // We do the same for the biases, with the exception that biases can be
    // considered as input neurons connecting the respective layers, with a
    // fixed input = +1. This means we only need to probe for the error at
    // layer k.
    blitz::secondIndex J;
    m_deriv_bias[k] = blitz::sum(m_error[k].transpose(1,0), J);
    for (int i=0; i<m_deriv_bias[k].extent(0); ++i) {
      int8_t M = sign(m_deriv_bias[k](i) * m_prev_deriv_bias[k](i));
      // Implementations equations (4-6) on the RProp paper:
      if (M > 0) {
        m_delta_bias[k](i) = std::min(m_delta_bias[k](i)*ETA_PLUS, DELTA_MAX); 
        m_bias_ref[k](i) -= sign(m_deriv_bias[k](i)) * m_delta_bias[k](i); 
        m_prev_deriv_bias[k](i) = m_deriv_bias[k](i);
      }
      else if (M < 0) {
        m_delta_bias[k](i) = std::max(m_delta_bias[k](i)*ETA_MINUS, DELTA_MIN);
        m_prev_deriv_bias[k](i) = 0;
      }
      else { //M == 0
        m_bias_ref[k](i) -= sign(m_deriv_bias[k](i)) * m_delta_bias[k](i);
        m_prev_deriv_bias[k](i) = m_deriv_bias[k](i);
      }
    }
  }
}

void train::MLPRPropTrainer::train(bob::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {
  if (!isCompatible(machine)) throw train::IncompatibleMachine();
  array::assertSameDimensionLength(getBatchSize(), input.extent(0));
  array::assertSameDimensionLength(getBatchSize(), target.extent(0));
  train_(machine, input, target);
}

void train::MLPRPropTrainer::train_(bob::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {

  m_output[0].reference(input);
  m_target.reference(target);

  // We refer to the machine's weights and biases
  for (size_t k=0;k<m_weight_ref.size();++k)
    m_weight_ref[k].reference(machine.getWeights()[k]);
  for (size_t k=0;k<m_bias_ref.size();++k)
    m_bias_ref[k].reference(machine.getBiases()[k]);

  // To be called in this sequence for a general backprop algorithm
  forward_step();
  backward_step();
  rprop_weight_update();
}
