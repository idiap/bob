/**
 * @file cxx/machine/src/LinearMachine.cc
 * @date Tue May 31 09:22:33 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a LinearMachine
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

#include <cmath>

#include "core/array_copy.h"
#include "io/Arrayset.h"
#include "machine/LinearMachine.h"
#include "machine/Exception.h"
#include "math/linear.h"

namespace mach = bob::machine;
namespace math = bob::math;

mach::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight)
  : m_input_sub(weight.extent(0)),
    m_input_div(weight.extent(0)),
    m_bias(weight.extent(1)),
    m_activation(mach::LINEAR),
    m_actfun(linear),
    m_buffer(weight.extent(0))
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_bias = 0.0;
  m_weight.reference(bob::core::array::ccopy(weight));
}

mach::LinearMachine::LinearMachine():
  m_input_sub(0),
  m_input_div(0),
  m_weight(0, 0),
  m_bias(0),
  m_activation(mach::LINEAR),
  m_actfun(linear),
  m_buffer(0)
{
}

mach::LinearMachine::LinearMachine(size_t n_input, size_t n_output):
  m_input_sub(n_input),
  m_input_div(n_input),
  m_weight(n_input, n_output),
  m_bias(n_output),
  m_activation(mach::LINEAR),
  m_actfun(linear),
  m_buffer(n_input)
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_weight = 0.0;
  m_bias = 0.0;
}

mach::LinearMachine::LinearMachine(const mach::LinearMachine& other):
  m_input_sub(bob::core::array::ccopy(other.m_input_sub)),
  m_input_div(bob::core::array::ccopy(other.m_input_div)),
  m_weight(bob::core::array::ccopy(other.m_weight)),
  m_bias(bob::core::array::ccopy(other.m_bias)),
  m_activation(other.m_activation),
  m_actfun(other.m_actfun),
  m_buffer(m_input_sub.shape())
{
}

mach::LinearMachine::LinearMachine (bob::io::HDF5File& config) {
  load(config);
}

mach::LinearMachine::~LinearMachine() {}

mach::LinearMachine& mach::LinearMachine::operator=
(const mach::LinearMachine& other) {
  m_input_sub.reference(bob::core::array::ccopy(other.m_input_sub));
  m_input_div.reference(bob::core::array::ccopy(other.m_input_div));
  m_weight.reference(bob::core::array::ccopy(other.m_weight));
  m_bias.reference(bob::core::array::ccopy(other.m_bias));
  m_activation = other.m_activation;
  m_actfun = other.m_actfun;
  m_buffer.resize(m_input_sub.shape());
  return *this;
}

void mach::LinearMachine::load (bob::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_input_sub.reference(config.readArray<double,1>("input_sub"));
  m_input_div.reference(config.readArray<double,1>("input_div"));
  m_weight.reference(config.readArray<double,2>("weights"));
  m_bias.reference(config.readArray<double,1>("biases"));
  m_buffer.resize(m_input_sub.extent(0));

  //reads the activation function
  uint32_t act = config.read<uint32_t>("activation");
  setActivation(static_cast<mach::Activation>(act));
}

void mach::LinearMachine::resize (size_t input, size_t output) {
  m_input_sub.resizeAndPreserve(input);
  m_input_div.resizeAndPreserve(input);
  m_buffer.resizeAndPreserve(input);
  m_weight.resizeAndPreserve(input, output);
  m_bias.resizeAndPreserve(output);
}

void mach::LinearMachine::save (bob::io::HDF5File& config) const {
  config.setArray("input_sub", m_input_sub);
  config.setArray("input_div", m_input_div);
  config.setArray("weights", m_weight);
  config.setArray("biases", m_bias);
  //bob's hdf5 implementation does not support enumerations yet...
  config.set("activation", static_cast<uint32_t>(m_activation));
}

void mach::LinearMachine::forward_
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  m_buffer = (input - m_input_sub) / m_input_div;
  math::prod_(m_buffer, m_weight, output);
  for (int i=0; i<m_weight.extent(1); ++i)
    output(i) = m_actfun(output(i) + m_bias(i));
}

void mach::LinearMachine::forward
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  if (m_weight.extent(0) != input.extent(0)) //checks input
    throw mach::NInputsMismatch(m_weight.extent(0),
        input.extent(0));
  if (m_weight.extent(1) != output.extent(0)) //checks output
    throw mach::NOutputsMismatch(m_weight.extent(1),
        output.extent(0));
  forward_(input, output);
}

void mach::LinearMachine::setWeights
(const blitz::Array<double,2>& weight) {
  if (weight.extent(0) != m_input_sub.extent(0)) { //checks input
    throw mach::NInputsMismatch(weight.extent(0), m_input_sub.extent(0));
  }
  if (weight.extent(1) != m_bias.extent(0)) { //checks output
    throw mach::NOutputsMismatch(weight.extent(1), m_bias.extent(0));
  }
  m_weight.reference(bob::core::array::ccopy(weight));
}

void mach::LinearMachine::setBiases
(const blitz::Array<double,1>& bias) {
  if (m_weight.extent(1) != bias.extent(0)) {
    throw mach::NOutputsMismatch(m_weight.extent(1), bias.extent(0));
  }
  m_bias.reference(bob::core::array::ccopy(bias));
}

void mach::LinearMachine::setInputSubtraction
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(0), v.extent(0));
  }
  m_input_sub.reference(bob::core::array::ccopy(v));
}

void mach::LinearMachine::setInputDivision
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(0), v.extent(0));
  }
  m_input_div.reference(bob::core::array::ccopy(v));
}

void mach::LinearMachine::setActivation (mach::Activation a) {
  switch (a) {
    case mach::LINEAR:
      m_actfun = mach::linear;
      break;
    case mach::TANH:
      m_actfun = std::tanh;
      break;
    case mach::LOG:
      m_actfun = mach::logistic;
      break;
  }
  m_activation = a;
}
