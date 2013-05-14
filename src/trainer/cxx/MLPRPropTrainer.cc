/**
 * @file trainer/cxx/MLPRPropTrainer.cc
 * @date Mon Jul 11 16:19:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the RProp algorithm for MLP training.
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

#include <algorithm>
#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <bob/math/linear.h>
#include <bob/machine/MLPException.h>
#include <bob/trainer/Exception.h>
#include <bob/trainer/MLPRPropTrainer.h>

bob::trainer::MLPRPropTrainer::MLPRPropTrainer(const bob::machine::MLP& machine,
    size_t batch_size):
  bob::trainer::MLPBaseTrainer(machine, batch_size),
  m_deriv(m_H + 1),
  m_deriv_bias(m_H + 1),
  m_prev_deriv(m_H + 1),
  m_prev_deriv_bias(m_H + 1)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_prev_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_prev_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();
}

bob::trainer::MLPRPropTrainer::~MLPRPropTrainer() { }

bob::trainer::MLPRPropTrainer::MLPRPropTrainer(const MLPRPropTrainer& other):
  bob::trainer::MLPBaseTrainer(other),
  m_deriv(m_H + 1),
  m_deriv_bias(m_H + 1),
  m_prev_deriv(m_H + 1),
  m_prev_deriv_bias(m_H + 1)
{
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k].reference(bob::core::array::ccopy(other.m_deriv[k]));
    m_deriv_bias[k].reference(bob::core::array::ccopy(other.m_deriv_bias[k]));
    m_prev_deriv[k].reference(bob::core::array::ccopy(other.m_prev_deriv[k]));
    m_prev_deriv_bias[k].reference(bob::core::array::ccopy(other.m_prev_deriv_bias[k]));
  }
}

bob::trainer::MLPRPropTrainer& bob::trainer::MLPRPropTrainer::operator=
(const bob::trainer::MLPRPropTrainer& other) {
  if (this != &other)
  {
    bob::trainer::MLPBaseTrainer::operator=(other);
    m_deriv.resize(m_H + 1);
    m_deriv_bias.resize(m_H + 1);
    m_prev_deriv.resize(m_H + 1);
    m_prev_deriv_bias.resize(m_H + 1);

    for (size_t k=0; k<(m_H + 1); ++k) {
      m_deriv[k].reference(bob::core::array::ccopy(other.m_deriv[k]));
      m_deriv_bias[k].reference(bob::core::array::ccopy(other.m_deriv_bias[k]));
      m_prev_deriv[k].reference(bob::core::array::ccopy(other.m_prev_deriv[k]));
      m_prev_deriv_bias[k].reference(bob::core::array::ccopy(other.m_prev_deriv_bias[k]));
    }
  }
  return *this;
}

void bob::trainer::MLPRPropTrainer::reset() {
  static const double DELTA0 = 0.1; ///< taken from the paper, section II.C

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_delta[k] = DELTA0;
    m_delta_bias[k] = DELTA0;
    m_prev_deriv[k] = 0;
    m_prev_deriv_bias[k] = 0;
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

void bob::trainer::MLPRPropTrainer::rprop_weight_update() {
  // constants taken from the paper.
  static const double ETA_MINUS = 0.5;
  static const double ETA_PLUS = 1.2;
  static const double DELTA_MAX = 50.0;
  static const double DELTA_MIN = 1e-6;

  for (size_t k=0; k<m_weight_ref.size(); ++k) { //for all layers
    bob::math::prod_(m_output[k].transpose(1,0), m_error[k], m_deriv[k]);

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

void bob::trainer::MLPRPropTrainer::train(bob::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {
  if (!isCompatible(machine)) throw bob::trainer::IncompatibleMachine();
  bob::core::array::assertSameDimensionLength(getBatchSize(), input.extent(0));
  bob::core::array::assertSameDimensionLength(getBatchSize(), target.extent(0));
  train_(machine, input, target);
}

void bob::trainer::MLPRPropTrainer::train_(bob::machine::MLP& machine,
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
