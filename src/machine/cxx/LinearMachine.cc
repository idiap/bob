/**
 * @file machine/cxx/LinearMachine.cc
 * @date Tue May 31 09:22:33 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a LinearMachine
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

#include <cmath>

#include <bob/core/array_copy.h>
#include <bob/machine/LinearMachine.h>
#include <bob/machine/Exception.h>
#include <bob/math/linear.h>

bob::machine::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight)
  : m_input_sub(weight.extent(0)),
    m_input_div(weight.extent(0)),
    m_bias(weight.extent(1)),
    m_activation(bob::machine::LINEAR),
    m_actfun(linear),
    m_buffer(weight.extent(0))
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_bias = 0.0;
  m_weight.reference(bob::core::array::ccopy(weight));
}

bob::machine::LinearMachine::LinearMachine():
  m_input_sub(0),
  m_input_div(0),
  m_weight(0, 0),
  m_bias(0),
  m_activation(bob::machine::LINEAR),
  m_actfun(linear),
  m_buffer(0)
{
}

bob::machine::LinearMachine::LinearMachine(size_t n_input, size_t n_output):
  m_input_sub(n_input),
  m_input_div(n_input),
  m_weight(n_input, n_output),
  m_bias(n_output),
  m_activation(bob::machine::LINEAR),
  m_actfun(linear),
  m_buffer(n_input)
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_weight = 0.0;
  m_bias = 0.0;
}

bob::machine::LinearMachine::LinearMachine(const bob::machine::LinearMachine& other):
  m_input_sub(bob::core::array::ccopy(other.m_input_sub)),
  m_input_div(bob::core::array::ccopy(other.m_input_div)),
  m_weight(bob::core::array::ccopy(other.m_weight)),
  m_bias(bob::core::array::ccopy(other.m_bias)),
  m_activation(other.m_activation),
  m_actfun(other.m_actfun),
  m_buffer(m_input_sub.shape())
{
}

bob::machine::LinearMachine::LinearMachine (bob::io::HDF5File& config) {
  load(config);
}

bob::machine::LinearMachine::~LinearMachine() {}

bob::machine::LinearMachine& bob::machine::LinearMachine::operator=
(const bob::machine::LinearMachine& other) {
  if(this != &other)
  {
    m_input_sub.reference(bob::core::array::ccopy(other.m_input_sub));
    m_input_div.reference(bob::core::array::ccopy(other.m_input_div));
    m_weight.reference(bob::core::array::ccopy(other.m_weight));
    m_bias.reference(bob::core::array::ccopy(other.m_bias));
    m_activation = other.m_activation;
    m_actfun = other.m_actfun;
    m_buffer.resize(m_input_sub.shape());
  }
  return *this;
}

bool 
bob::machine::LinearMachine::operator==(const bob::machine::LinearMachine& b) const
{
  return (bob::core::array::isEqual(m_input_sub, b.m_input_sub) &&
          bob::core::array::isEqual(m_input_div, b.m_input_div) &&
          bob::core::array::isEqual(m_weight, b.m_weight) &&
          bob::core::array::isEqual(m_bias, b.m_bias) &&
          m_activation == b.m_activation);
}

bool 
bob::machine::LinearMachine::operator!=(const bob::machine::LinearMachine& b) const
{
  return !(this->operator==(b));
}

bool 
bob::machine::LinearMachine::is_similar_to(const bob::machine::LinearMachine& b,
  const double r_epsilon, const double a_epsilon) const
{
  return (bob::core::array::isClose(m_input_sub, b.m_input_sub, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_input_div, b.m_input_div, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_weight, b.m_weight, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_bias, b.m_bias, r_epsilon, a_epsilon) &&
          m_activation == b.m_activation);
}

void bob::machine::LinearMachine::load (bob::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_input_sub.reference(config.readArray<double,1>("input_sub"));
  m_input_div.reference(config.readArray<double,1>("input_div"));
  m_weight.reference(config.readArray<double,2>("weights"));
  m_bias.reference(config.readArray<double,1>("biases"));
  m_buffer.resize(m_input_sub.extent(0));

  //reads the activation function
  uint32_t act = config.read<uint32_t>("activation");
  setActivation(static_cast<bob::machine::Activation>(act));
}

void bob::machine::LinearMachine::resize (size_t input, size_t output) {
  m_input_sub.resizeAndPreserve(input);
  m_input_div.resizeAndPreserve(input);
  m_buffer.resizeAndPreserve(input);
  m_weight.resizeAndPreserve(input, output);
  m_bias.resizeAndPreserve(output);
}

void bob::machine::LinearMachine::save (bob::io::HDF5File& config) const {
  config.setArray("input_sub", m_input_sub);
  config.setArray("input_div", m_input_div);
  config.setArray("weights", m_weight);
  config.setArray("biases", m_bias);
  //bob's hdf5 implementation does not support enumerations yet...
  config.set("activation", static_cast<uint32_t>(m_activation));
}

void bob::machine::LinearMachine::forward_
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  m_buffer = (input - m_input_sub) / m_input_div;
  bob::math::prod_(m_buffer, m_weight, output);
  for (int i=0; i<m_weight.extent(1); ++i)
    output(i) = m_actfun(output(i) + m_bias(i));
}

void bob::machine::LinearMachine::forward
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  if (m_weight.extent(0) != input.extent(0)) //checks input
    throw bob::machine::NInputsMismatch(m_weight.extent(0),
        input.extent(0));
  if (m_weight.extent(1) != output.extent(0)) //checks output
    throw bob::machine::NOutputsMismatch(m_weight.extent(1),
        output.extent(0));
  forward_(input, output);
}

void bob::machine::LinearMachine::setWeights
(const blitz::Array<double,2>& weight) {
  if (weight.extent(0) != m_input_sub.extent(0)) { //checks input
    throw bob::machine::NInputsMismatch(weight.extent(0), m_input_sub.extent(0));
  }
  if (weight.extent(1) != m_bias.extent(0)) { //checks output
    throw bob::machine::NOutputsMismatch(weight.extent(1), m_bias.extent(0));
  }
  m_weight.reference(bob::core::array::ccopy(weight));
}

void bob::machine::LinearMachine::setBiases
(const blitz::Array<double,1>& bias) {
  if (m_weight.extent(1) != bias.extent(0)) {
    throw bob::machine::NOutputsMismatch(m_weight.extent(1), bias.extent(0));
  }
  m_bias.reference(bob::core::array::ccopy(bias));
}

void bob::machine::LinearMachine::setInputSubtraction
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw bob::machine::NInputsMismatch(m_weight.extent(0), v.extent(0));
  }
  m_input_sub.reference(bob::core::array::ccopy(v));
}

void bob::machine::LinearMachine::setInputDivision
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw bob::machine::NInputsMismatch(m_weight.extent(0), v.extent(0));
  }
  m_input_div.reference(bob::core::array::ccopy(v));
}

void bob::machine::LinearMachine::setActivation (bob::machine::Activation a) {
  switch (a) {
    case bob::machine::LINEAR:
      m_actfun = bob::machine::linear;
      break;
    case bob::machine::TANH:
      m_actfun = std::tanh;
      break;
    case bob::machine::LOG:
      m_actfun = bob::machine::logistic;
      break;
  }
  m_activation = a;
}
