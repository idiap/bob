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
#include <bob/trainer/Exception.h>
#include <bob/trainer/MLPRPropTrainer.h>

bob::trainer::MLPRPropTrainer::MLPRPropTrainer(size_t batch_size,
    boost::shared_ptr<bob::trainer::Cost> cost):
  bob::trainer::MLPBaseTrainer(batch_size, cost),
  m_eta_minus(0.5),
  m_eta_plus(1.2),
  m_delta_zero(0.1),
  m_delta_min(1e-6),
  m_delta_max(50.0),
  m_delta(numberOfHiddenLayers() + 1),
  m_delta_bias(numberOfHiddenLayers() + 1),
  m_prev_deriv(numberOfHiddenLayers() + 1),
  m_prev_deriv_bias(numberOfHiddenLayers() + 1)
{
  reset();
}


bob::trainer::MLPRPropTrainer::MLPRPropTrainer(size_t batch_size,
    boost::shared_ptr<bob::trainer::Cost> cost,
    const bob::machine::MLP& machine):
  bob::trainer::MLPBaseTrainer(batch_size, cost, machine),
  m_eta_minus(0.5),
  m_eta_plus(1.2),
  m_delta_zero(0.1),
  m_delta_min(1e-6),
  m_delta_max(50.0),
  m_delta(numberOfHiddenLayers() + 1),
  m_delta_bias(numberOfHiddenLayers() + 1),
  m_prev_deriv(numberOfHiddenLayers() + 1),
  m_prev_deriv_bias(numberOfHiddenLayers() + 1)
{
  initialize(machine);
}

bob::trainer::MLPRPropTrainer::MLPRPropTrainer(size_t batch_size,
    boost::shared_ptr<bob::trainer::Cost> cost,
    const bob::machine::MLP& machine,
    bool train_biases):
  bob::trainer::MLPBaseTrainer(batch_size, cost, machine, train_biases),
  m_eta_minus(0.5),
  m_eta_plus(1.2),
  m_delta_zero(0.1),
  m_delta_min(1e-6),
  m_delta_max(50.0),
  m_delta(numberOfHiddenLayers() + 1),
  m_delta_bias(numberOfHiddenLayers() + 1),
  m_prev_deriv(numberOfHiddenLayers() + 1),
  m_prev_deriv_bias(numberOfHiddenLayers() + 1)
{
  initialize(machine);
}

bob::trainer::MLPRPropTrainer::~MLPRPropTrainer() { }

bob::trainer::MLPRPropTrainer::MLPRPropTrainer(const MLPRPropTrainer& other):
  bob::trainer::MLPBaseTrainer(other),
  m_eta_minus(other.m_eta_minus),
  m_eta_plus(other.m_eta_plus),
  m_delta_zero(other.m_delta_zero),
  m_delta_min(other.m_delta_min),
  m_delta_max(other.m_delta_max),
  m_delta(numberOfHiddenLayers() + 1),
  m_delta_bias(numberOfHiddenLayers() + 1),
  m_prev_deriv(numberOfHiddenLayers() + 1),
  m_prev_deriv_bias(numberOfHiddenLayers() + 1)
{
  bob::core::array::ccopy(other.m_delta, m_delta);
  bob::core::array::ccopy(other.m_delta_bias, m_delta_bias);
  bob::core::array::ccopy(other.m_prev_deriv, m_prev_deriv);
  bob::core::array::ccopy(other.m_prev_deriv_bias, m_prev_deriv_bias);
}

bob::trainer::MLPRPropTrainer& bob::trainer::MLPRPropTrainer::operator=
(const bob::trainer::MLPRPropTrainer& other) {
  if (this != &other)
  {
    bob::trainer::MLPBaseTrainer::operator=(other);

    m_eta_minus = other.m_eta_minus;
    m_eta_plus = other.m_eta_plus;
    m_delta_zero = other.m_delta_zero;
    m_delta_min = other.m_delta_min;
    m_delta_max = other.m_delta_max;

    bob::core::array::ccopy(other.m_delta, m_delta);
    bob::core::array::ccopy(other.m_delta_bias, m_delta_bias);
    bob::core::array::ccopy(other.m_prev_deriv, m_prev_deriv);
    bob::core::array::ccopy(other.m_prev_deriv_bias, m_prev_deriv_bias);
  }
  return *this;
}

void bob::trainer::MLPRPropTrainer::reset() {
  for (size_t k=0; k<(numberOfHiddenLayers() + 1); ++k) {
    m_delta[k] = m_delta_zero;
    m_delta_bias[k] = m_delta_zero;
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

void bob::trainer::MLPRPropTrainer::rprop_weight_update(bob::machine::MLP& machine,
  const blitz::Array<double,2>& input)
{
  std::vector<blitz::Array<double,2> >& machine_weight = machine.updateWeights();
  std::vector<blitz::Array<double,1> >& machine_bias = machine.updateBiases();
  const std::vector<blitz::Array<double,2> >& deriv = getDerivatives();

  for (size_t k=0; k<machine_weight.size(); ++k) { //for all layers
    // Calculates the sign change as prescribed on the RProp paper. Depending
    // on the sign change, we update the "weight_update" matrix and apply the
    // updates on the respective weights.
    for (int i=0; i<deriv[k].extent(0); ++i) {
      for (int j=0; j<deriv[k].extent(1); ++j) {
        int8_t M = sign(deriv[k](i,j) * m_prev_deriv[k](i,j));
        // Implementations equations (4-6) on the RProp paper:
        if (M > 0) {
          m_delta[k](i,j) = std::min(m_delta[k](i,j)*m_eta_plus, m_delta_max);
          machine_weight[k](i,j) -= sign(deriv[k](i,j)) * m_delta[k](i,j);
          m_prev_deriv[k](i,j) = deriv[k](i,j);
        }
        else if (M < 0) {
          m_delta[k](i,j) = std::max(m_delta[k](i,j)*m_eta_minus, m_delta_min);
          m_prev_deriv[k](i,j) = 0;
        }
        else { //M == 0
          machine_weight[k](i,j) -= sign(deriv[k](i,j)) * m_delta[k](i,j);
          m_prev_deriv[k](i,j) = deriv[k](i,j);
        }
      }
    }

    // Here we decide if we should train the biases or not
    if (!getTrainBiases()) continue;

    const std::vector<blitz::Array<double,1> >& deriv_bias = getBiasDerivatives();

    // We do the same for the biases, with the exception that biases can be
    // considered as input neurons connecting the respective layers, with a
    // fixed input = +1. This means we only need to probe for the error at
    // layer k.
    for (int i=0; i<deriv_bias[k].extent(0); ++i) {
      int8_t M = sign(deriv_bias[k](i) * m_prev_deriv_bias[k](i));
      // Implementations equations (4-6) on the RProp paper:
      if (M > 0) {
        m_delta_bias[k](i) = std::min(m_delta_bias[k](i)*m_eta_plus, m_delta_max);
        machine_bias[k](i) -= sign(deriv_bias[k](i)) * m_delta_bias[k](i);
        m_prev_deriv_bias[k](i) = deriv_bias[k](i);
      }
      else if (M < 0) {
        m_delta_bias[k](i) = std::max(m_delta_bias[k](i)*m_eta_minus, m_delta_min);
        m_prev_deriv_bias[k](i) = 0;
      }
      else { //M == 0
        machine_bias[k](i) -= sign(deriv_bias[k](i)) * m_delta_bias[k](i);
        m_prev_deriv_bias[k](i) = deriv_bias[k](i);
      }
    }
  }
}

void bob::trainer::MLPRPropTrainer::initialize(const bob::machine::MLP& machine)
{
  bob::trainer::MLPBaseTrainer::initialize(machine);

  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  m_delta.resize(numberOfHiddenLayers() + 1);
  m_delta_bias.resize(numberOfHiddenLayers() + 1);
  m_prev_deriv.resize(numberOfHiddenLayers() + 1);
  m_prev_deriv_bias.resize(numberOfHiddenLayers() + 1);
  for (size_t k=0; k<(numberOfHiddenLayers() + 1); ++k) {
    m_delta[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_delta_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_prev_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_prev_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();
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

  // To be called in this sequence for a general backprop algorithm
  forward_step(machine, input);
  backward_step(machine, input, target);
  rprop_weight_update(machine, input);
}

void bob::trainer::MLPRPropTrainer::setPreviousDerivatives(const std::vector<blitz::Array<double,2> >& v) {
  bob::core::array::assertSameDimensionLength(v.size(), m_prev_deriv.size());
  for (size_t k=0; k<v.size(); ++k) {
    bob::core::array::assertSameShape(v[k], m_prev_deriv[k]);
    m_prev_deriv[k] = v[k];
  }
}

void bob::trainer::MLPRPropTrainer::setPreviousDerivative(const blitz::Array<double,2>& v, const size_t k) {
  if (k >= m_prev_deriv.size()) {
    boost::format m("MLPRPropTrainer: index for setting derivative array %lu is not on the expected range of [0, %lu]");
    m % k % (m_prev_deriv.size()-1);
    throw std::runtime_error(m.str());
  }
  bob::core::array::assertSameShape(v, m_prev_deriv[k]);
  m_prev_deriv[k] = v;
}

void bob::trainer::MLPRPropTrainer::setPreviousBiasDerivatives(const std::vector<blitz::Array<double,1> >& v) {
  bob::core::array::assertSameDimensionLength(v.size(), m_prev_deriv_bias.size());
  for (size_t k=0; k<v.size(); ++k)
  {
    bob::core::array::assertSameShape(v[k], m_prev_deriv_bias[k]);
    m_prev_deriv_bias[k] = v[k];
  }
}

void bob::trainer::MLPRPropTrainer::setPreviousBiasDerivative(const blitz::Array<double,1>& v, const size_t k) {
  if (k >= m_prev_deriv_bias.size()) {
    boost::format m("MLPRPropTrainer: index for setting derivative bias array %lu is not on the expected range of [0, %lu]");
    m % k % (m_prev_deriv_bias.size()-1);
    throw std::runtime_error(m.str());
  }
  bob::core::array::assertSameShape(v, m_prev_deriv_bias[k]);
  m_prev_deriv_bias[k] = v;
}

void bob::trainer::MLPRPropTrainer::setDeltas(const std::vector<blitz::Array<double,2> >& v) {
  bob::core::array::assertSameDimensionLength(v.size(), m_delta.size());
  for (size_t k=0; k<v.size(); ++k) {
    bob::core::array::assertSameShape(v[k], m_delta[k]);
    m_delta[k] = v[k];
  }
}

void bob::trainer::MLPRPropTrainer::setDelta(const blitz::Array<double,2>& v, const size_t k) {
  if (k >= m_delta.size()) {
    boost::format m("MLPRPropTrainer: index for setting delta array %lu is not on the expected range of [0, %lu]");
    m % k % (m_delta.size()-1);
    throw std::runtime_error(m.str());
  }
  bob::core::array::assertSameShape(v, m_delta[k]);
  m_delta[k] = v;
}

void bob::trainer::MLPRPropTrainer::setBiasDeltas(const std::vector<blitz::Array<double,1> >& v) {
  bob::core::array::assertSameDimensionLength(v.size(), m_delta_bias.size());
  for (size_t k=0; k<v.size(); ++k)
  {
    bob::core::array::assertSameShape(v[k], m_delta_bias[k]);
    m_delta_bias[k] = v[k];
  }
}

void bob::trainer::MLPRPropTrainer::setBiasDelta(const blitz::Array<double,1>& v, const size_t k) {
  if (k >= m_delta_bias.size()) {
    boost::format m("MLPRPropTrainer: index for setting delta bias array %lu is not on the expected range of [0, %lu]");
    m % k % (m_delta_bias.size()-1);
    throw std::runtime_error(m.str());
  }
  bob::core::array::assertSameShape(v, m_delta_bias[k]);
  m_delta_bias[k] = v;
}
