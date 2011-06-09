/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 30 May 2011 13:01:33 CEST
 *
 * @brief Implements a LinearMachine
 */

#include <cmath>

#include "database/Arrayset.h"
#include "machine/LinearMachine.h"
#include "machine/Exception.h"

namespace mach = Torch::machine;

inline static double linear(double x) { return x; }
inline static double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }
      
mach::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight, 
    const blitz::Array<double,1>& bias)
  : m_input_sub(weight.extent(1)),
    m_input_div(weight.extent(1)),
    m_activation(mach::LinearMachine::LINEAR),
    m_actfun(linear),
    m_buffer(weight.extent(1))
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  if (weight.extent(0) != bias.extent(0)) {
    throw mach::NInputsMismatch(weight.extent(0), bias.extent(0));
  }
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}

mach::LinearMachine::LinearMachine(size_t n_input, size_t n_output):
  m_input_sub(n_input),
  m_input_div(n_input),
  m_weight(n_output, n_input),
  m_bias(n_output),
  m_activation(mach::LinearMachine::LINEAR),
  m_actfun(linear),
  m_buffer(n_input)
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_weight = 0;
  m_bias = 0;
}

mach::LinearMachine::LinearMachine(const mach::LinearMachine& other):
  m_input_sub(other.m_input_sub.copy()),
  m_input_div(other.m_input_div.copy()),
  m_weight(other.m_weight.copy()),
  m_bias(other.m_bias.copy()),
  m_activation(other.m_activation),
  m_actfun(other.m_actfun),
  m_buffer(m_input_sub.shape())
{
}

mach::LinearMachine::LinearMachine (Torch::database::HDF5File& config) {
  load(config);
}

mach::LinearMachine::~LinearMachine() {}

mach::LinearMachine& mach::LinearMachine::operator=
(const mach::LinearMachine& other) {
  m_input_sub.reference(other.m_input_sub.copy());
  m_input_div.reference(other.m_input_div.copy());
  m_weight.reference(other.m_weight.copy());
  m_bias.reference(other.m_bias.copy());
  m_activation = other.m_activation;
  m_actfun = other.m_actfun;
  m_buffer.resize(m_input_sub.shape());
  return *this;
}

void mach::LinearMachine::load (Torch::database::HDF5File& config) {
  //query linear machine shape
  const Torch::database::HDF5Type& t = config.describe("weights");
  size_t n_output = t.shape()[0];
  size_t n_input = t.shape()[1];

  //reset all members to prepare the data copy using HDF5
  m_input_sub.resize(n_input);
  m_input_div.resize(n_input);
  m_buffer.resize(n_input);
  m_weight.resize(n_output, n_input);
  m_bias.resize(n_output);

  //reads all data directly into the member variables
  config.readArray("input_sub", m_input_sub);
  config.readArray("input_div", m_input_div);
  config.readArray("weights", m_weight);
  config.readArray("biases", m_bias);

  //reads the activation function
  uint32_t act = 0;
  config.read("activation", act);
  setActivation(static_cast<mach::LinearMachine::Activation>(act));
}

void mach::LinearMachine::save (Torch::database::HDF5File& config) const {
  config.appendArray("input_sub", m_input_sub);
  config.appendArray("input_div", m_input_div);
  config.appendArray("weights", m_weight);
  config.appendArray("biases", m_bias);
  //torch's hdf5 implementation does not support enumerations yet...
  config.append("activation", static_cast<uint32_t>(m_activation));
}

void mach::LinearMachine::forward
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  if (m_weight.extent(0) != output.extent(0))
    throw mach::NOutputsMismatch(m_weight.extent(0),
        output.extent(0));
  if (m_weight.extent(1) != input.extent(0))
    throw mach::NInputsMismatch(m_weight.extent(1),
        input.extent(0));

  m_buffer = (input - m_input_sub) / m_input_div;
  blitz::Range a = blitz::Range::all();
  for (int i=0; i<m_weight.extent(0); ++i)
    output(i) = m_actfun(blitz::sum(m_weight(i,a)*m_buffer) + m_bias(i));
}

void mach::LinearMachine::setWeights
(const blitz::Array<double,2>& weight) {
  if (weight.extent(0) != m_bias.extent(0)) {
    throw mach::NOutputsMismatch(weight.extent(0), m_bias.extent(0));
  }
  m_weight.reference(weight.copy());
}

void mach::LinearMachine::setBiases
(const blitz::Array<double,1>& bias) {
  if (m_weight.extent(0) != bias.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(0), bias.extent(0));
  }
  m_bias.reference(bias.copy());
}

void mach::LinearMachine::setInputSubtraction
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(1) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(1), v.extent(0));
  }
  m_input_sub.reference(v.copy());
}

void mach::LinearMachine::setInputDivision
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(1) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(1), v.extent(0));
  }
  m_input_div.reference(v.copy());
}

void mach::LinearMachine::setActivation (mach::LinearMachine::Activation a) {
  switch (a) {
    case mach::LinearMachine::LINEAR:
      m_actfun = linear;
      break;
    case mach::LinearMachine::TANH:
      m_actfun = std::tanh;
      break;
    case mach::LinearMachine::LOG:
      m_actfun = logistic;
      break;
  }
  m_activation = a;
}

void mach::LinearMachine::setAll(const blitz::Array<double,1> input_sub,
    const blitz::Array<double,1> input_div,
    const blitz::Array<double,2>& weight, 
    const blitz::Array<double,1>& bias) {
  //some size checking...
  if (weight.extent(0) != bias.extent(0)) {
    throw mach::NOutputsMismatch(weight.extent(0), bias.extent(0));
  }
  if (input_sub.extent(0) != input_div.extent(0)) {
    throw mach::NInputsMismatch(input_sub.extent(0), input_div.extent(0));
  }
  if (input_sub.extent(0) != weight.extent(1)) {
    throw mach::NInputsMismatch(input_sub.extent(0), weight.extent(1));
  }
  m_input_sub.reference(input_sub.copy());
  m_input_div.reference(input_div.copy());
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
  m_buffer.resize(m_input_sub.shape());
}
