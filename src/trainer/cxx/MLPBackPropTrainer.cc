/**
 * @file trainer/cxx/MLPBackPropTrainer.cc
 * @date Mon Jul 18 18:11:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the BackProp algorithm for MLP training.
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
#include <bob/math/linear.h>
#include <bob/machine/MLPException.h>
#include <bob/trainer/Exception.h>
#include <bob/trainer/MLPBackPropTrainer.h>

bob::trainer::MLPBackPropTrainer::MLPBackPropTrainer(const bob::machine::MLP& machine,
    size_t batch_size):
  bob::trainer::MLPBaseTrainer(machine, batch_size),
  m_learning_rate(0.1),
  m_momentum(0.0),
  m_prev_delta(m_H + 1),
  m_prev_delta_bias(m_H + 1)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_prev_delta[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_prev_delta_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();
}

bob::trainer::MLPBackPropTrainer::~MLPBackPropTrainer() { }

bob::trainer::MLPBackPropTrainer::MLPBackPropTrainer(const MLPBackPropTrainer& other):
  bob::trainer::MLPBaseTrainer(other),
  m_learning_rate(other.m_learning_rate),
  m_momentum(other.m_momentum),
  m_prev_delta(m_H + 1),
  m_prev_delta_bias(m_H + 1)
{
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_prev_delta[k].reference(bob::core::array::ccopy(other.m_prev_delta[k]));
    m_prev_delta_bias[k].reference(bob::core::array::ccopy(other.m_prev_delta_bias[k]));
  }
}

bob::trainer::MLPBackPropTrainer& bob::trainer::MLPBackPropTrainer::operator=
(const bob::trainer::MLPBackPropTrainer& other) {
  if (this != &other)
  {
    bob::trainer::MLPBaseTrainer::operator=(other);
    m_learning_rate = other.m_learning_rate;
    m_momentum = other.m_momentum;
    m_prev_delta.resize(m_H + 1);
    m_prev_delta_bias.resize(m_H + 1);

    for (size_t k=0; k<(m_H + 1); ++k) {
      m_prev_delta[k].reference(bob::core::array::ccopy(other.m_prev_delta[k]));
      m_prev_delta_bias[k].reference(bob::core::array::ccopy(other.m_prev_delta_bias[k]));
    }
  }
  return *this;
}

void bob::trainer::MLPBackPropTrainer::reset() {
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_prev_delta[k] = 0;
    m_prev_delta_bias[k] = 0;
  }
}

void bob::trainer::MLPBackPropTrainer::backprop_weight_update(bob::machine::MLP& machine,
  const blitz::Array<double,2>& input)
{
  std::vector<blitz::Array<double,2> >& machine_weight =
    machine.updateWeights();
  std::vector<blitz::Array<double,1> >& machine_bias =
    machine.updateBiases();
  for (size_t k=0; k<machine_weight.size(); ++k) { //for all layers
    if (k == 0) bob::math::prod_(input.transpose(1,0), m_error[k], m_delta[k]);
    else bob::math::prod_(m_output[k-1].transpose(1,0), m_error[k], m_delta[k]);
    m_delta[k] *= m_learning_rate / m_batch_size;
    machine_weight[k] += ((1-m_momentum)*m_delta[k]) + 
      (m_momentum*m_prev_delta[k]);
    m_prev_delta[k] = m_delta[k];

    // Here we decide if we should train the biases or not
    if (!m_train_bias) continue;

    // We do the same for the biases, with the exception that biases can be
    // considered as input neurons connecting the respective layers, with a
    // fixed input = +1. This means we only need to probe for the error at
    // layer k.
    blitz::secondIndex J;
    m_delta_bias[k] = m_learning_rate * 
      blitz::mean(m_error[k].transpose(1,0), J);
    machine_bias[k] += ((1-m_momentum)*m_delta_bias[k]) + 
      (m_momentum*m_prev_delta_bias[k]);
    m_prev_delta_bias[k] = m_delta_bias[k];
  }
}

void bob::trainer::MLPBackPropTrainer::train(bob::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {
  if (!isCompatible(machine)) throw bob::trainer::IncompatibleMachine();
  bob::core::array::assertSameDimensionLength(getBatchSize(), input.extent(0));
  bob::core::array::assertSameDimensionLength(getBatchSize(), target.extent(0));
  train_(machine, input, target);
}

void bob::trainer::MLPBackPropTrainer::train_(bob::machine::MLP& machine,
    const blitz::Array<double,2>& input,
    const blitz::Array<double,2>& target) {

  // Initialize the training
  init_train(machine, input, target);

  // To be called in this sequence for a general backprop algorithm
  forward_step(machine, input);
  backward_step(machine, target);
  backprop_weight_update(machine, input);
}
