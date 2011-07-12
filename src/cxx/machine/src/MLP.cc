/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Jul 17:57:01 2011 
 *
 * @brief Implementation of MLPs 
 */

#include <cmath>
#include <boost/format.hpp>

#include "core/array_check.h"
#include "io/Arrayset.h"
#include "machine/MLP.h"
#include "machine/MLPException.h"
#include "math/linear.h"

namespace mach = Torch::machine;
namespace math = Torch::math;
namespace array = Torch::core::array;

mach::MLP::MLP (size_t input, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(1),
  m_bias(1),
  m_activation(mach::TANH),
  m_actfun(std::tanh),
  m_buffer(1)
{
  resize(input, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

mach::MLP::MLP (size_t input, size_t hidden, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(2),
  m_bias(2),
  m_activation(mach::TANH),
  m_actfun(std::tanh),
  m_buffer(2)
{
  resize(input, hidden, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

mach::MLP::MLP (size_t input, const std::vector<size_t>& hidden, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(hidden.size()+1),
  m_bias(hidden.size()+1),
  m_activation(mach::TANH),
  m_actfun(std::tanh),
  m_buffer(hidden.size()+1)
{
  resize(input, hidden, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

mach::MLP::MLP (const std::vector<size_t>& shape):
  m_activation(mach::TANH),
  m_actfun(std::tanh)
{
  resize(shape);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

mach::MLP::MLP (const mach::MLP& other):
  m_input_sub(other.m_input_sub.copy()),
  m_input_div(other.m_input_div.copy()),
  m_weight(other.m_weight.size()),
  m_bias(other.m_bias.size()),
  m_activation(other.m_activation),
  m_actfun(other.m_actfun),
  m_buffer(other.m_buffer.size())
{
  for (size_t i=0; i<other.m_weight.size(); ++i) {
    m_weight[i].reference(other.m_weight[i].copy());
    m_bias[i].reference(other.m_bias[i].copy());
    m_buffer[i].reference(other.m_buffer[i].copy());
  }
}

mach::MLP::MLP (Torch::io::HDF5File& config) {
  load(config);
}

mach::MLP::~MLP() { }

mach::MLP& mach::MLP::operator= (const MLP& other) {
  m_input_sub.reference(other.m_input_sub.copy());
  m_input_div.reference(other.m_input_div.copy());
  m_weight.resize(other.m_weight.size());
  m_bias.resize(other.m_bias.size());
  m_activation = other.m_activation;
  m_actfun = other.m_actfun;
  m_buffer.resize(other.m_buffer.size());
  for (size_t i=0; i<other.m_weight.size(); ++i) {
    m_weight[i].reference(other.m_weight[i].copy());
    m_bias[i].reference(other.m_bias[i].copy());
    m_buffer[i].reference(other.m_buffer[i].copy());
  }
  return *this;
}

void mach::MLP::load (Torch::io::HDF5File& config) {
  uint8_t nhidden;
  config.read("nhidden", nhidden);
  m_weight.resize(nhidden+1);
  m_bias.resize(nhidden+1);
  m_buffer.resize(nhidden+1);

  //configures the input
  m_input_sub.reference(config.readArray<double,1>("input_sub"));
  m_input_div.reference(config.readArray<double,1>("input_div"));

  boost::format weight("weight_%d");
  boost::format bias("bias_%d");
  ++nhidden;
  for (size_t i=0; i<nhidden; ++i) {
    weight % i;
    m_weight[i].reference(config.readArray<double,2>(weight.str()));
    bias % i;
    m_bias[i].reference(config.readArray<double,1>(bias.str()));
  }

  //reads the activation function
  uint32_t act = 0;
  config.read("activation", act);
  setActivation(static_cast<mach::Activation>(act));

  //setup buffers: first, input
  m_buffer[0].reference(blitz::Array<double,1>(m_input_sub.shape()));
  for (size_t i=1; i<m_weight.size(); ++i) {
    //buffers have to be sized the same as the input for the next layer
    m_buffer[i].reference(blitz::Array<double,1>(m_weight[i].extent(0)));
  }
}

void mach::MLP::save (Torch::io::HDF5File& config) const {
  config.setArray("input_sub", m_input_sub);
  config.setArray("input_div", m_input_div);
  config.set("nhidden", (uint8_t)(m_weight.size()-1));
  boost::format weight("weight_%d");
  boost::format bias("bias_%d");
  for (size_t i=0; i<m_weight.size(); ++i) {
    weight % i;
    bias % i;
    config.setArray(weight.str(), m_weight[i]);
    config.setArray(bias.str(), m_bias[i]);
  }
  //torch's hdf5 implementation does not support enumerations yet...
  config.set("activation", static_cast<uint32_t>(m_activation));
}

void mach::MLP::forward_ (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) const {

  //doesn't check input, just computes
  m_buffer[0] = (input - m_input_sub) / m_input_div;

  //input -> hidden[0]; hidden[0] -> hidden[1], ..., hidden[N-2] -> hidden[N-1]
  for (size_t j=1; j<m_weight.size(); ++j) {
    math::prod_(m_buffer[j-1], m_weight[j-1], m_buffer[j]);
    for (int i=0; i<m_buffer[j].extent(0); ++i) {
      m_buffer[j](i) = m_actfun(m_buffer[j](i) + m_bias[j-1](i));
    }
  }

  //hidden[N-1] -> output
  math::prod_(m_buffer.back(), m_weight.back(), output);
  const blitz::Array<double,1>& last_bias = m_bias.back(); //opt. access
  for (int i=0; i<output.extent(0); ++i) {
    output(i) = m_actfun(output(i) + last_bias(i));
  }
}

void mach::MLP::forward (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) const {

  //checks input
  if (m_weight.front().extent(0) != input.extent(0)) //checks input
    throw mach::NInputsMismatch(m_weight.front().extent(0),
        input.extent(0));
  if (m_weight.back().extent(1) != output.extent(0)) //checks output
    throw mach::NOutputsMismatch(m_weight.back().extent(1),
        output.extent(0));
  forward_(input, output); 
}

void mach::MLP::resize (size_t input, size_t output) {
  m_input_sub.resize(input);
  m_input_sub = 0;
  m_input_div.resize(input);
  m_input_div = 1;
  m_weight.resize(1);
  m_weight[0].reference(blitz::Array<double,2>(input, output));
  m_bias.resize(1);
  m_bias[0].reference(blitz::Array<double,1>(output));
  m_buffer.resize(1);
  m_buffer[0].reference(blitz::Array<double,1>(input));
}

void mach::MLP::resize (size_t input, size_t hidden, size_t output) {
  std::vector<size_t> vhidden(1, hidden);
  resize(input, vhidden, output);
}

void mach::MLP::resize (size_t input, const std::vector<size_t>& hidden,
    size_t output) {

  if (hidden.size() == 0) {
    resize(input, output);
    return;
  }

  m_input_sub.resize(input);
  m_input_sub = 0;
  m_input_div.resize(input);
  m_input_div = 1;
  m_weight.resize(hidden.size()+1);
  m_bias.resize(hidden.size()+1);
  m_buffer.resize(hidden.size()+1);
  
  //initializes first layer
  m_weight[0].reference(blitz::Array<double,2>(input, hidden[0]));
  m_bias[0].reference(blitz::Array<double,1>(hidden[0]));
  m_buffer[0].reference(blitz::Array<double,1>(input));

  //initializes hidden layers
  for (size_t i=0; i<(hidden.size()-1); ++i) {
    m_weight[i+1].reference(blitz::Array<double,2>(hidden[i], hidden[i+1]));
    m_bias[i+1].reference(blitz::Array<double,1>(hidden[i+1]));
    m_buffer[i+1].reference(blitz::Array<double,1>(hidden[i]));
  }

  //initializes the last layer
  m_weight.back().reference(blitz::Array<double,2>(hidden.back(), output));
  m_bias.back().reference(blitz::Array<double,1>(output));
}

void mach::MLP::resize (const std::vector<size_t>& shape) {

  if (shape.size() < 2) throw mach::InvalidShape();
  
  if (shape.size() == 2) {
    resize(shape[0], shape[1]);
    return;
  }

  //falls back to the normal case
  size_t input = shape.front();
  size_t output = shape.back();
  std::vector<size_t> vhidden(shape.size()-2);
  for (size_t i=1; i<(shape.size()-1); ++i) vhidden[i-1] = shape[i];
  resize(input, vhidden, output);
}

void mach::MLP::setInputSubtraction(const blitz::Array<double,1>& v) {
  if (m_weight.front().extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.front().extent(0), v.extent(0));
  }
  m_input_sub.reference(v.copy());
}

void mach::MLP::setInputDivision(const blitz::Array<double,1>& v) {
  if (m_weight.front().extent(0) != v.extent(0)) {
    throw mach::NInputsMismatch(m_weight.front().extent(0), v.extent(0));
  }
  m_input_div.reference(v.copy());
}

void mach::MLP::setWeights(const std::vector<blitz::Array<double,2> >& weight) {
  if (m_weight.size() != weight.size()) 
    throw mach::NumberOfLayersMismatch(m_weight.size(), weight.size());
  for (size_t i=0; i<m_weight.size(); ++i) {
    if (!array::hasSameShape(m_weight[i], weight[i])) {
      throw mach::WeightShapeMismatch(i, m_weight[i].shape(), weight[i].shape());
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = weight[i];
}

void mach::MLP::setWeights(double v) { 
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = v;
}

void mach::MLP::setBiases(const std::vector<blitz::Array<double,1> >& bias) {
  if (m_bias.size() != bias.size()) 
    throw mach::NumberOfLayersMismatch(m_bias.size(), bias.size());
  for (size_t i=0; i<m_bias.size(); ++i) {
    if (!array::hasSameShape(m_bias[i], bias[i])) {
      throw mach::BiasShapeMismatch(i, m_bias[i].shape()[0], bias[i].shape()[0]);
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = bias[i];
}

void mach::MLP::setBiases(double v) {
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = v;
}

void mach::MLP::setActivation(mach::Activation a) {
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
    default:
      throw mach::UnsupportedActivation(a);
  }
  m_activation = a;
}
