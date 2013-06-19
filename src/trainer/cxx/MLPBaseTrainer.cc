/**
 * @file trainer/cxx/MLPBaseTrainer.cc
 * @date Tue May 14 12:04:51 CEST 2013
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#include <algorithm>
#include <bob/core/assert.h>
#include <bob/core/check.h>
#include <bob/core/Exception.h>
#include <bob/math/linear.h>
#include <bob/trainer/Exception.h>
#include <bob/trainer/MLPBaseTrainer.h>

bob::trainer::MLPBaseTrainer::MLPBaseTrainer(size_t batch_size,
    boost::shared_ptr<bob::trainer::Cost> cost):
  m_batch_size(batch_size),
  m_cost(cost),
  m_train_bias(true),
  m_H(0), ///< handy!
  m_deriv(1),
  m_deriv_bias(1),
  m_error(1),
  m_output(1)
{
  m_deriv[0].reference(blitz::Array<double,2>(0,0));
  m_deriv_bias[0].reference(blitz::Array<double,1>(0));
  m_error[0].reference(blitz::Array<double,2>(0,0));
  m_output[0].reference(blitz::Array<double,2>(0,0));

  reset();
}

bob::trainer::MLPBaseTrainer::MLPBaseTrainer(size_t batch_size, 
    boost::shared_ptr<bob::trainer::Cost> cost,
    const bob::machine::MLP& machine):
  m_batch_size(batch_size),
  m_cost(cost),
  m_train_bias(true),
  m_H(machine.numOfHiddenLayers()), ///< handy!
  m_deriv(m_H + 1),
  m_deriv_bias(m_H + 1),
  m_error(m_H + 1),
  m_output(m_H + 1)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
  }

  reset();

  setBatchSize(batch_size);
}

bob::trainer::MLPBaseTrainer::~MLPBaseTrainer() { }

bob::trainer::MLPBaseTrainer::MLPBaseTrainer(const MLPBaseTrainer& other):
  m_batch_size(other.m_batch_size),
  m_cost(other.m_cost),
  m_train_bias(other.m_train_bias),
  m_H(other.m_H)
{
  bob::core::array::ccopy(other.m_deriv, m_deriv);
  bob::core::array::ccopy(other.m_deriv_bias, m_deriv_bias);
  bob::core::array::ccopy(other.m_error, m_error);
  bob::core::array::ccopy(other.m_output, m_output);
}

bob::trainer::MLPBaseTrainer& bob::trainer::MLPBaseTrainer::operator=
(const bob::trainer::MLPBaseTrainer& other) {
  if (this != &other)
  {
    m_batch_size = other.m_batch_size;
    m_cost = other.m_cost;
    m_train_bias = other.m_train_bias;
    m_H = other.m_H;

    bob::core::array::ccopy(other.m_deriv, m_deriv);
    bob::core::array::ccopy(other.m_deriv_bias, m_deriv_bias);
    bob::core::array::ccopy(other.m_error, m_error);
    bob::core::array::ccopy(other.m_output, m_output);
  }
  return *this;
}

void bob::trainer::MLPBaseTrainer::setBatchSize (size_t batch_size) {
  // m_output: values after the activation function
  // m_error: error values;
 
  m_batch_size = batch_size;
   
  for (size_t k=0; k<m_output.size(); ++k) {
    m_output[k].resize(batch_size, m_deriv[k].extent(1));
    m_output[k] = 0.;
  }

  for (size_t k=0; k<m_error.size(); ++k) {
    m_error[k].resize(batch_size, m_deriv[k].extent(1));
    m_error[k] = 0.;
  }
}

bool bob::trainer::MLPBaseTrainer::isCompatible(const bob::machine::MLP& machine) const 
{
  if (m_H != machine.numOfHiddenLayers()) return false;
  
  if (m_deriv.back().extent(1) != (int)machine.outputSize()) return false;

  if (m_deriv[0].extent(0) != (int)machine.inputSize()) return false;

  //also, each layer should be of the same size
  for (size_t k=0; k<(m_H + 1); ++k) {
    if (!bob::core::array::hasSameShape(m_deriv[k], machine.getWeights()[k])) return false;
  }

  //if you get to this point, you can only return true
  return true;
}

void bob::trainer::MLPBaseTrainer::forward_step(const bob::machine::MLP& machine, 
  const blitz::Array<double,2>& input)
{
  const std::vector<blitz::Array<double,2> >& machine_weight = machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias = machine.getBiases();

  boost::shared_ptr<bob::machine::Activation> hidden_actfun = machine.getHiddenActivation();
  boost::shared_ptr<bob::machine::Activation> output_actfun = machine.getOutputActivation();

  for (size_t k=0; k<machine_weight.size(); ++k) { //for all layers
    if (k == 0) bob::math::prod_(input, machine_weight[k], m_output[k]);
    else bob::math::prod_(m_output[k-1], machine_weight[k], m_output[k]);
    boost::shared_ptr<bob::machine::Activation> cur_actfun = 
      (k == (machine_weight.size()-1) ? output_actfun : hidden_actfun );
    for (int i=0; i<(int)m_batch_size; ++i) { //for every example
      for (int j=0; j<m_output[k].extent(1); ++j) { //for all variables
        m_output[k](i,j) = cur_actfun->f(m_output[k](i,j) + machine_bias[k](j));
      }
    }
  }
}

void bob::trainer::MLPBaseTrainer::backward_step(const bob::machine::MLP& machine,
  const blitz::Array<double,2>& target)
{
  const std::vector<blitz::Array<double,2> >& machine_weight = machine.getWeights();

  //last layer
  boost::shared_ptr<bob::machine::Activation> output_actfun = machine.getOutputActivation();
  for (int i=0; i<(int)m_batch_size; ++i) { //for every example
    for (int j=0; j<m_error[m_H].extent(1); ++j) { //for all variables
      m_error[m_H](i,j) = m_cost->error(m_output[m_H](i,j), target(i,j));
    }
  }

  //all other layers
  boost::shared_ptr<bob::machine::Activation> hidden_actfun = machine.getHiddenActivation();
  for (size_t k=m_H; k>0; --k) {
    bob::math::prod_(m_error[k], machine_weight[k].transpose(1,0), m_error[k-1]);
    for (int i=0; i<(int)m_batch_size; ++i) { //for every example
      for (int j=0; j<m_error[k-1].extent(1); ++j) { //for all variables
        m_error[k-1](i,j) *= hidden_actfun->f_prime_from_f(m_output[k-1](i,j));
      }
    }
  }
}

void bob::trainer::MLPBaseTrainer::cost_derivatives_step(const bob::machine::MLP& machine, 
  const blitz::Array<double,2>& input)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();

  for (size_t k=0; k<machine_weight.size(); ++k) { //for all layers
    // For the weights
    if (k == 0) bob::math::prod_(input.transpose(1,0), m_error[k], m_deriv[k]);
    else bob::math::prod_(m_output[k-1].transpose(1,0), m_error[k], m_deriv[k]);
    m_deriv[k] /= m_batch_size;
    // For the biases
    blitz::secondIndex bj;
    m_deriv_bias[k] = blitz::mean(m_error[k].transpose(1,0), bj);
  }
}

double bob::trainer::MLPBaseTrainer::average_cost
(const blitz::Array<double,2>& target) const {
  bob::core::array::assertSameShape(m_output[m_H], target);
  uint64_t counter = 0;
  double retval = 0.0;
  for (int i=0; i<target.extent(0); ++i) { //for every example
    for (int j=0; j<target.extent(1); ++j) { //for all variables
      retval += m_cost->f(m_output[m_H](i,j), target(i,j));
      ++counter;
    }
  }
  return retval / counter;
}

double bob::trainer::MLPBaseTrainer::average_cost
(const bob::machine::MLP& machine, const blitz::Array<double,2>& input,
 const blitz::Array<double,2>& target) {
  forward_step(machine, input);
  backward_step(machine, target);
  return average_cost(target);
}

void bob::trainer::MLPBaseTrainer::initialize(const bob::machine::MLP& machine)
{
  const std::vector<blitz::Array<double,2> >& machine_weight =
    machine.getWeights();
  const std::vector<blitz::Array<double,1> >& machine_bias =
    machine.getBiases();

  m_H = machine.numOfHiddenLayers();
  m_deriv.resize(m_H + 1);
  m_deriv_bias.resize(m_H + 1);
  m_output.resize(m_H + 1);
  m_error.resize(m_H + 1);
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k].reference(blitz::Array<double,2>(machine_weight[k].shape()));
    m_deriv_bias[k].reference(blitz::Array<double,1>(machine_bias[k].shape()));
    m_output[k].resize(m_batch_size, m_deriv[k].extent(1));
    m_error[k].resize(m_batch_size, m_deriv[k].extent(1));
  }

  reset();
}

void bob::trainer::MLPBaseTrainer::setError(const std::vector<blitz::Array<double,2> >& error) {
  bob::core::array::assertSameDimensionLength(error.size(), m_error.size());
  for (size_t k=0; k<error.size(); ++k)
  {
    bob::core::array::assertSameShape(error[k], m_error[k]);
    m_error[k] = error[k];
  }
}

void bob::trainer::MLPBaseTrainer::setError(const blitz::Array<double,2>& error, const size_t id) {
  if (id >= m_error.size())
    throw bob::core::InvalidArgumentException("MLPBaseTrainer: Index in error array",
      (int)id, 0, (int)(m_error.size()-1));
  bob::core::array::assertSameShape(error, m_error[id]);
  m_error[id] = error;
}

void bob::trainer::MLPBaseTrainer::setOutput(const std::vector<blitz::Array<double,2> >& output) {
  bob::core::array::assertSameDimensionLength(output.size(), m_output.size());
  for (size_t k=0; k<output.size(); ++k)
  {
    bob::core::array::assertSameShape(output[k], m_output[k]);
    m_output[k] = output[k];
  }
}

void bob::trainer::MLPBaseTrainer::setOutput(const blitz::Array<double,2>& output, const size_t id) {
  if (id >= m_output.size())
    throw bob::core::InvalidArgumentException("MLPBaseTrainer: Index in output array", 
      (int)id, 0, (int)(m_output.size()-1));
  bob::core::array::assertSameShape(output, m_output[id]);
  m_output[id] = output;
}

void bob::trainer::MLPBaseTrainer::setDeriv(const std::vector<blitz::Array<double,2> >& deriv) {
  bob::core::array::assertSameDimensionLength(deriv.size(), m_deriv.size());
  for (size_t k=0; k<deriv.size(); ++k)
  {
    bob::core::array::assertSameShape(deriv[k], m_deriv[k]);
    m_deriv[k] = deriv[k];
  }
}

void bob::trainer::MLPBaseTrainer::setDeriv(const blitz::Array<double,2>& deriv, const size_t id) {
  if (id >= m_deriv.size())
    throw bob::core::InvalidArgumentException("MLPBaseTrainer: Index in deriv array", 
      (int)id, 0, (int)(m_deriv.size()-1));
  bob::core::array::assertSameShape(deriv, m_deriv[id]);
  m_deriv[id] = deriv;
}

void bob::trainer::MLPBaseTrainer::setDerivBias(const std::vector<blitz::Array<double,1> >& deriv_bias) {
  bob::core::array::assertSameDimensionLength(deriv_bias.size(), m_deriv_bias.size());
  for (size_t k=0; k<deriv_bias.size(); ++k)
  {
    bob::core::array::assertSameShape(deriv_bias[k], m_deriv_bias[k]);
    m_deriv_bias[k] = deriv_bias[k];
  }
}

void bob::trainer::MLPBaseTrainer::setDerivBias(const blitz::Array<double,1>& deriv_bias, const size_t id) {
  if (id >= m_deriv_bias.size())
    throw bob::core::InvalidArgumentException("MLPBaseTrainer: Index in deriv_bias array", 
      (int)id, 0, (int)(m_deriv_bias.size()-1));
  bob::core::array::assertSameShape(deriv_bias, m_deriv_bias[id]);
  m_deriv_bias[id] = deriv_bias;
}

void bob::trainer::MLPBaseTrainer::reset() {
  for (size_t k=0; k<(m_H + 1); ++k) {
    m_deriv[k] = 0.;
    m_deriv_bias[k] = 0.;
    m_error[k] = 0.;
    m_output[k] = 0.;
  }
}
