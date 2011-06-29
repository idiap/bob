/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 30 May 2011 13:01:33 CEST
 *
 * @brief Implements a LinearMachine
 */

#include <cmath>

#include "io/Arrayset.h"
#include "machine/LinearMachine.h"
#include "machine/Exception.h"
#include "math/linear.h"

namespace mach = Torch::machine;
namespace math = Torch::math;

inline static double linear(double x) { return x; }
inline static double logistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }
      
mach::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight)
  : m_input_sub(weight.extent(0)),
    m_input_div(weight.extent(0)),
    m_bias(weight.extent(1)),
    m_activation(mach::LinearMachine::LINEAR),
    m_actfun(linear),
    m_buffer(weight.extent(0))
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_bias = 0.0;
  m_weight.reference(weight.copy());
}

mach::LinearMachine::LinearMachine():
  m_input_sub(0),
  m_input_div(0),
  m_weight(0, 0),
  m_bias(0),
  m_activation(mach::LinearMachine::LINEAR),
  m_actfun(linear),
  m_buffer(0)
{
}

mach::LinearMachine::LinearMachine(size_t n_input, size_t n_output):
  m_input_sub(n_input),
  m_input_div(n_input),
  m_weight(n_input, n_output),
  m_bias(n_output),
  m_activation(mach::LinearMachine::LINEAR),
  m_actfun(linear),
  m_buffer(n_input)
{
  m_input_sub = 0.0;
  m_input_div = 1.0;
  m_weight = 0.0;
  m_bias = 0.0;
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

mach::LinearMachine::LinearMachine (Torch::io::HDF5File& config) {
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

void mach::LinearMachine::load (Torch::io::HDF5File& config) {
  //query linear machine shape
  const Torch::io::HDF5Type& t = config.describe("weights");
  size_t n_input = t.shape()[0];
  size_t n_output = t.shape()[1];

  //reset all members to prepare the data copy using HDF5
  m_input_sub.resize(n_input);
  m_input_div.resize(n_input);
  m_buffer.resize(n_input);
  m_weight.resize(n_input, n_output);
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

void mach::LinearMachine::resize (size_t input, size_t output) {
  m_input_sub.resizeAndPreserve(input);
  m_input_div.resizeAndPreserve(input);
  m_buffer.resizeAndPreserve(input);
  m_weight.resizeAndPreserve(input, output);
  m_bias.resizeAndPreserve(output);
}

void mach::LinearMachine::save (Torch::io::HDF5File& config) const {
  config.appendArray("input_sub", m_input_sub);
  config.appendArray("input_div", m_input_div);
  config.appendArray("weights", m_weight);
  config.appendArray("biases", m_bias);
  //torch's hdf5 implementation does not support enumerations yet...
  config.append("activation", static_cast<uint32_t>(m_activation));
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
    throw mach::NInputsMismatch(weight.extent(0), m_bias.extent(0));
  }
  if (weight.extent(1) != m_bias.extent(0)) { //checks output
    throw mach::NOutputsMismatch(weight.extent(1), m_bias.extent(0));
  }
  m_weight.reference(weight.copy());
}

void mach::LinearMachine::setBiases
(const blitz::Array<double,1>& bias) {
  if (m_weight.extent(1) != bias.extent(0)) {
    throw mach::NOutputsMismatch(m_weight.extent(1), bias.extent(0));
  }
  m_bias.reference(bias.copy());
}

void mach::LinearMachine::setInputSubtraction
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(0), v.extent(0));
  }
  m_input_sub.reference(v.copy());
}

void mach::LinearMachine::setInputDivision
(const blitz::Array<double,1>& v) {
  if (m_weight.extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.extent(0), v.extent(0));
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
