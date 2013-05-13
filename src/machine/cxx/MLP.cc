/**
 * @file machine/cxx/MLP.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of MLPs
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

#include <sys/time.h>
#include <cmath>
#include <boost/format.hpp>

#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <bob/core/assert.h>
#include <bob/machine/MLP.h>
#include <bob/machine/MLPException.h>
#include <bob/math/linear.h>

bob::machine::MLP::MLP (size_t input, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(1),
  m_bias(1),
  m_activation(bob::machine::TANH),
  m_actfun(std::tanh),
  m_output_activation(bob::machine::TANH),
  m_output_actfun(std::tanh),
  m_z(1),
  m_a(1),
  m_b(1)
{
  resize(input, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

bob::machine::MLP::MLP (size_t input, size_t hidden, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(2),
  m_bias(2),
  m_activation(bob::machine::TANH),
  m_actfun(std::tanh),
  m_output_activation(bob::machine::TANH),
  m_output_actfun(std::tanh),
  m_z(2),
  m_a(2),
  m_b(2)
{
  resize(input, hidden, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

bob::machine::MLP::MLP (size_t input, const std::vector<size_t>& hidden, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(hidden.size()+1),
  m_bias(hidden.size()+1),
  m_activation(bob::machine::TANH),
  m_actfun(std::tanh),
  m_output_activation(bob::machine::TANH),
  m_output_actfun(std::tanh),
  m_z(hidden.size()+1),
  m_a(hidden.size()+1),
  m_b(hidden.size()+1)
{
  resize(input, hidden, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

bob::machine::MLP::MLP (const std::vector<size_t>& shape):
  m_activation(bob::machine::TANH),
  m_actfun(std::tanh),
  m_output_activation(bob::machine::TANH),
  m_output_actfun(std::tanh)
{
  resize(shape);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

bob::machine::MLP::MLP (const bob::machine::MLP& other):
  m_input_sub(bob::core::array::ccopy(other.m_input_sub)),
  m_input_div(bob::core::array::ccopy(other.m_input_div)),
  m_weight(other.m_weight.size()),
  m_bias(other.m_bias.size()),
  m_activation(other.m_activation),
  m_actfun(other.m_actfun),
  m_output_activation(other.m_output_activation),
  m_output_actfun(other.m_output_actfun),
  m_z(other.m_z.size()),
  m_a(other.m_a.size()),
  m_b(other.m_b.size())
{
  for (size_t i=0; i<other.m_weight.size(); ++i) {
    m_weight[i].reference(bob::core::array::ccopy(other.m_weight[i]));
    m_bias[i].reference(bob::core::array::ccopy(other.m_bias[i]));
    m_z[i].reference(bob::core::array::ccopy(other.m_z[i]));
    m_a[i].reference(bob::core::array::ccopy(other.m_a[i]));
    m_b[i].reference(bob::core::array::ccopy(other.m_b[i]));
  }
}

bob::machine::MLP::MLP (bob::io::HDF5File& config) {
  load(config);
}

bob::machine::MLP::~MLP() { }

bob::machine::MLP& bob::machine::MLP::operator= (const MLP& other) {
  m_input_sub.reference(bob::core::array::ccopy(other.m_input_sub));
  m_input_div.reference(bob::core::array::ccopy(other.m_input_div));
  m_weight.resize(other.m_weight.size());
  m_bias.resize(other.m_bias.size());
  m_activation = other.m_activation;
  m_actfun = other.m_actfun;
  m_output_activation = other.m_output_activation;
  m_output_actfun = other.m_output_actfun;
  m_z.resize(other.m_z.size());
  m_a.resize(other.m_a.size());
  m_b.resize(other.m_b.size());
  for (size_t i=0; i<other.m_weight.size(); ++i) {
    m_weight[i].reference(bob::core::array::ccopy(other.m_weight[i]));
    m_bias[i].reference(bob::core::array::ccopy(other.m_bias[i]));
    m_z[i].reference(bob::core::array::ccopy(other.m_z[i]));
    m_a[i].reference(bob::core::array::ccopy(other.m_a[i]));
    m_b[i].reference(bob::core::array::ccopy(other.m_b[i]));
  }
  return *this;
}

void bob::machine::MLP::load (bob::io::HDF5File& config) {
  uint8_t nhidden = config.read<uint8_t>("nhidden");
  m_weight.resize(nhidden+1);
  m_bias.resize(nhidden+1);
  m_z.resize(nhidden+1);
  m_a.resize(nhidden+1);
  m_b.resize(nhidden+1);

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
  uint32_t act = config.read<uint32_t>("activation");
  setActivation(static_cast<bob::machine::Activation>(act));
  if (config.contains("output_activation"))
  {
    uint32_t out_act = config.read<uint32_t>("output_activation");
    setOutputActivation(static_cast<bob::machine::Activation>(out_act));
  }
  else
    setOutputActivation(static_cast<bob::machine::Activation>(act));

  //setup buffers: first, input
  m_a[0].reference(blitz::Array<double,1>(m_input_sub.shape()));
  for (size_t i=1; i<m_weight.size(); ++i) {
    //buffers have to be sized the same as the input for the next layer
    m_z[i-1].reference(blitz::Array<double,1>(m_weight[i].extent(0)));
    m_a[i].reference(blitz::Array<double,1>(m_weight[i].extent(0)));
    m_b[i-1].reference(blitz::Array<double,1>(m_weight[i].extent(0)));
  }
  m_z.back().reference(blitz::Array<double,1>(m_bias.back().shape()));
  m_b.back().reference(blitz::Array<double,1>(m_bias.back().shape()));
}

void bob::machine::MLP::save (bob::io::HDF5File& config) const {
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
  //bob's hdf5 implementation does not support enumerations yet...
  config.set("activation", static_cast<uint32_t>(m_activation));
  config.set("output_activation", static_cast<uint32_t>(m_output_activation));
}

void bob::machine::MLP::forward_ (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) {

  //doesn't check input, just computes
  m_a[0] = (input - m_input_sub) / m_input_div;

  //input -> hidden[0]; hidden[0] -> hidden[1], ..., hidden[N-2] -> hidden[N-1]
  for (size_t j=1; j<m_weight.size(); ++j) {
    bob::math::prod_(m_a[j-1], m_weight[j-1], m_z[j-1]);
    m_z[j-1] += m_bias[j-1];
    for (int i=0; i<m_a[j].extent(0); ++i) {
      m_a[j](i) = m_actfun(m_z[j-1](i));
    }
  }

  //hidden[N-1] -> output
  bob::math::prod_(m_a.back(), m_weight.back(), m_z.back());
  m_z.back() += m_bias.back();
  for (int i=0; i<output.extent(0); ++i) {
    output(i) = m_output_actfun(m_z.back()(i));
  }
}

void bob::machine::MLP::forward (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) {

  //checks input
  if (m_weight.front().extent(0) != input.extent(0)) //checks input
    throw bob::machine::NInputsMismatch(m_weight.front().extent(0),
        input.extent(0));
  if (m_weight.back().extent(1) != output.extent(0)) //checks output
    throw bob::machine::NOutputsMismatch(m_weight.back().extent(1),
        output.extent(0));
  forward_(input, output); 
}

void bob::machine::MLP::forward_ (const blitz::Array<double,2>& input,
    blitz::Array<double,2>& output) {

  blitz::Range all = blitz::Range::all();
  for (int i=0; i<input.extent(0); ++i) {
    blitz::Array<double,1> inref(input(i,all));
    blitz::Array<double,1> outref(output(i,all));
    forward_(inref, outref);
  }
}

void bob::machine::MLP::forward (const blitz::Array<double,2>& input,
    blitz::Array<double,2>& output) {

  //checks input
  if (m_weight.front().extent(0) != input.extent(1)) //checks input
    throw bob::machine::NInputsMismatch(m_weight.front().extent(0),
        input.extent(1));
  if (m_weight.back().extent(1) != output.extent(1)) //checks output
    throw bob::machine::NOutputsMismatch(m_weight.back().extent(1),
        output.extent(1));
  //checks output
  bob::core::array::assertSameDimensionLength(input.extent(0), output.extent(0));
  forward_(input, output); 
}

void bob::machine::MLP::resize (size_t input, size_t output) {
  m_input_sub.resize(input);
  m_input_sub = 0;
  m_input_div.resize(input);
  m_input_div = 1;
  m_weight.resize(1);
  m_weight[0].reference(blitz::Array<double,2>(input, output));
  m_bias.resize(1);
  m_bias[0].reference(blitz::Array<double,1>(output));
  m_z.resize(1);
  m_z[0].reference(blitz::Array<double,1>(output));
  m_a.resize(1);
  m_a[0].reference(blitz::Array<double,1>(input));
  m_b.resize(1);
  m_b[0].reference(blitz::Array<double,1>(output));
}

void bob::machine::MLP::resize (size_t input, size_t hidden, size_t output) {
  std::vector<size_t> vhidden(1, hidden);
  resize(input, vhidden, output);
}

void bob::machine::MLP::resize (size_t input, const std::vector<size_t>& hidden,
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
  m_z.resize(hidden.size()+1);
  m_a.resize(hidden.size()+1);
  m_b.resize(hidden.size()+1);
  
  //initializes first layer
  m_weight[0].reference(blitz::Array<double,2>(input, hidden[0]));
  m_bias[0].reference(blitz::Array<double,1>(hidden[0]));
  m_a[0].reference(blitz::Array<double,1>(input));

  //initializes hidden layers
  const size_t NH1 = hidden.size()-1;
  for (size_t i=0; i<NH1; ++i) {
    m_weight[i+1].reference(blitz::Array<double,2>(hidden[i], hidden[i+1]));
    m_bias[i+1].reference(blitz::Array<double,1>(hidden[i+1]));
    m_z[i].reference(blitz::Array<double,1>(hidden[i]));
    m_a[i+1].reference(blitz::Array<double,1>(hidden[i]));
    m_b[i].reference(blitz::Array<double,1>(hidden[i]));
  }
  m_z[NH1].reference(blitz::Array<double,1>(hidden[NH1]));
  m_b[NH1].reference(blitz::Array<double,1>(hidden[NH1]));

  //initializes the last layer
  m_weight.back().reference(blitz::Array<double,2>(hidden.back(), output));
  m_bias.back().reference(blitz::Array<double,1>(output));
  m_z.back().reference(blitz::Array<double,1>(output));
  m_a.back().reference(blitz::Array<double,1>(hidden.back()));
  m_b.back().reference(blitz::Array<double,1>(output));
}

void bob::machine::MLP::resize (const std::vector<size_t>& shape) {

  if (shape.size() < 2) throw bob::machine::InvalidShape();
  
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

void bob::machine::MLP::setInputSubtraction(const blitz::Array<double,1>& v) {
  if (m_weight.front().extent(0) != v.extent(0)) {
    throw bob::machine::NInputsMismatch(m_weight.front().extent(0), v.extent(0));
  }
  m_input_sub.reference(bob::core::array::ccopy(v));
}

void bob::machine::MLP::setInputDivision(const blitz::Array<double,1>& v) {
  if (m_weight.front().extent(0) != v.extent(0)) {
    throw bob::machine::NInputsMismatch(m_weight.front().extent(0), v.extent(0));
  }
  m_input_div.reference(bob::core::array::ccopy(v));
}

void bob::machine::MLP::setWeights(const std::vector<blitz::Array<double,2> >& weight) {
  if (m_weight.size() != weight.size()) 
    throw bob::machine::NumberOfLayersMismatch(m_weight.size(), weight.size());
  for (size_t i=0; i<m_weight.size(); ++i) {
    if (!bob::core::array::hasSameShape(m_weight[i], weight[i])) {
      throw bob::machine::WeightShapeMismatch(i, weight[i].shape(), m_weight[i].shape());
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = weight[i];
}

void bob::machine::MLP::setWeights(double v) { 
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = v;
}

void bob::machine::MLP::setBiases(const std::vector<blitz::Array<double,1> >& bias) {
  if (m_bias.size() != bias.size()) 
    throw bob::machine::NumberOfLayersMismatch(m_bias.size(), bias.size());
  for (size_t i=0; i<m_bias.size(); ++i) {
    if (!bob::core::array::hasSameShape(m_bias[i], bias[i])) {
      throw bob::machine::BiasShapeMismatch(i, m_bias[i].shape()[0], bias[i].shape()[0]);
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = bias[i];
}

void bob::machine::MLP::setBiases(double v) {
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = v;
}

void bob::machine::MLP::setActivation(bob::machine::Activation a) {
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
    default:
      throw bob::machine::UnsupportedActivation(a);
  }
  m_activation = a;
}

void bob::machine::MLP::setOutputActivation(bob::machine::Activation a) {
  switch (a) {
    case bob::machine::LINEAR:
      m_output_actfun = bob::machine::linear;
      break;
    case bob::machine::TANH:
      m_output_actfun = std::tanh;
      break;
    case bob::machine::LOG:
      m_output_actfun = bob::machine::logistic;
      break;
    default:
      throw bob::machine::UnsupportedActivation(a);
  }
  m_output_activation = a;
}


void bob::machine::MLP::randomize(boost::mt19937& rng, double lower_bound, double upper_bound) {
  boost::uniform_real<double> draw(lower_bound, upper_bound);

  for (size_t k=0; k<m_weight.size(); ++k) {
    for (int i=0; i<m_weight[k].extent(0); ++i) {
      for (int j=0; j<m_weight[k].extent(1); ++j) {
        m_weight[k](i,j) = draw(rng);
      }
    }
    for (int i=0; i<m_bias[k].extent(0); ++i) m_bias[k](i) = draw(rng);
  }
}

void bob::machine::MLP::randomize(double lower_bound, double upper_bound) {
  struct timeval tv;
  gettimeofday(&tv, 0);
  boost::mt19937 rng(tv.tv_sec + tv.tv_usec);
  randomize(rng, lower_bound, upper_bound);
}
