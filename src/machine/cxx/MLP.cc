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
#include <boost/make_shared.hpp>

#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <bob/core/assert.h>
#include <bob/machine/MLP.h>
#include <bob/math/linear.h>

bob::machine::MLP::MLP (size_t input, size_t output):
  m_input_sub(input),
  m_input_div(input),
  m_weight(1),
  m_bias(1),
  m_hidden_activation(boost::make_shared<bob::machine::HyperbolicTangentActivation>()),
  m_output_activation(m_hidden_activation),
  m_buffer(1)
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
  m_hidden_activation(boost::make_shared<bob::machine::HyperbolicTangentActivation>()),
  m_output_activation(m_hidden_activation),
  m_buffer(2)
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
  m_hidden_activation(boost::make_shared<bob::machine::HyperbolicTangentActivation>()),
  m_output_activation(m_hidden_activation),
  m_buffer(hidden.size()+1)
{
  resize(input, hidden, output);
  m_input_sub = 0;
  m_input_div = 1;
  setWeights(0);
  setBiases(0);
}

bob::machine::MLP::MLP (const std::vector<size_t>& shape):
  m_hidden_activation(boost::make_shared<bob::machine::HyperbolicTangentActivation>()),
  m_output_activation(m_hidden_activation)
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
  m_hidden_activation(other.m_hidden_activation),
  m_output_activation(other.m_output_activation),
  m_buffer(other.m_buffer.size())
{
  for (size_t i=0; i<other.m_weight.size(); ++i) {
    m_weight[i].reference(bob::core::array::ccopy(other.m_weight[i]));
    m_bias[i].reference(bob::core::array::ccopy(other.m_bias[i]));
    m_buffer[i].reference(bob::core::array::ccopy(other.m_buffer[i]));
  }
}

bob::machine::MLP::MLP (bob::io::HDF5File& config) {
  load(config);
}

bob::machine::MLP::~MLP() { }

bob::machine::MLP& bob::machine::MLP::operator= (const MLP& other) {
  if (this != &other)
  {
    m_input_sub.reference(bob::core::array::ccopy(other.m_input_sub));
    m_input_div.reference(bob::core::array::ccopy(other.m_input_div));
    m_weight.resize(other.m_weight.size());
    m_bias.resize(other.m_bias.size());
    m_hidden_activation = other.m_hidden_activation;
    m_output_activation = other.m_output_activation;
    m_buffer.resize(other.m_buffer.size());
    for (size_t i=0; i<other.m_weight.size(); ++i) {
      m_weight[i].reference(bob::core::array::ccopy(other.m_weight[i]));
      m_bias[i].reference(bob::core::array::ccopy(other.m_bias[i]));
      m_buffer[i].reference(bob::core::array::ccopy(other.m_buffer[i]));
    }
  }
  return *this;
}

bool bob::machine::MLP::operator== (const MLP& other) const {
  return (bob::core::array::isEqual(m_input_sub, other.m_input_sub) &&
          bob::core::array::isEqual(m_input_div, other.m_input_div) &&
          bob::core::array::isEqual(m_weight, other.m_weight) &&
          bob::core::array::isEqual(m_bias, other.m_bias) &&
          m_hidden_activation->str() == other.m_hidden_activation->str() &&
          m_output_activation->str() == other.m_output_activation->str());
}

bool bob::machine::MLP::operator!= (const MLP& other) const {
  return !(this->operator==(other));
}

bool bob::machine::MLP::is_similar_to(const bob::machine::MLP& other,
    const double r_epsilon, const double a_epsilon) const
{
  return (bob::core::array::isClose(m_input_sub, other.m_input_sub, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_input_div, other.m_input_div, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_weight, other.m_weight, r_epsilon, a_epsilon) &&
          bob::core::array::isClose(m_bias, other.m_bias, r_epsilon, a_epsilon) &&
          m_hidden_activation->str() == other.m_hidden_activation->str() &&
          m_output_activation->str() == other.m_output_activation->str());
}


void bob::machine::MLP::load (bob::io::HDF5File& config) {
  uint8_t nhidden = config.read<uint8_t>("nhidden");
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

  //switch between different versions - support for version 2
  if (config.hasAttribute(".", "version")) { //new version
    config.cd("hidden_activation");
    m_hidden_activation = bob::machine::load_activation(config);
    config.cd("../output_activation");
    m_output_activation = bob::machine::load_activation(config);
    config.cd("..");
  }
  else { //old version
    uint32_t act = config.read<uint32_t>("activation");
    m_hidden_activation = bob::machine::make_deprecated_activation(act);
    m_output_activation = m_hidden_activation;
  }

  //setup buffers: first, input
  m_buffer[0].reference(blitz::Array<double,1>(m_input_sub.shape()));
  for (size_t i=1; i<m_weight.size(); ++i) {
    //buffers have to be sized the same as the input for the next layer
    m_buffer[i].reference(blitz::Array<double,1>(m_weight[i].extent(0)));
  }
}

void bob::machine::MLP::save (bob::io::HDF5File& config) const {
  config.setAttribute(".", "version", 1);
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
  config.createGroup("hidden_activation");
  config.cd("hidden_activation");
  m_hidden_activation->save(config);
  config.cd("..");
  config.createGroup("output_activation");
  config.cd("output_activation");
  m_output_activation->save(config);
  config.cd("..");
}

void bob::machine::MLP::forward_ (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) {

  //doesn't check input, just computes
  m_buffer[0] = (input - m_input_sub) / m_input_div;

  //input -> hidden[0]; hidden[0] -> hidden[1], ..., hidden[N-2] -> hidden[N-1]
  for (size_t j=1; j<m_weight.size(); ++j) {
    bob::math::prod_(m_buffer[j-1], m_weight[j-1], m_buffer[j]);
    m_buffer[j] += m_bias[j-1];
    for (int i=0; i<m_buffer[j].extent(0); ++i) {
      m_buffer[j](i) = m_hidden_activation->f(m_buffer[j](i));
    }
  }

  //hidden[N-1] -> output
  bob::math::prod_(m_buffer.back(), m_weight.back(), output);
  output += m_bias.back();
  for (int i=0; i<output.extent(0); ++i) {
    output(i) = m_output_activation->f(output(i));
  }
}

void bob::machine::MLP::forward (const blitz::Array<double,1>& input,
    blitz::Array<double,1>& output) {

  //checks input
  if (m_weight.front().extent(0) != input.extent(0)) {//checks input
    boost::format m("mismatch on the input dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.front().extent(0) % input.extent(0);
    throw std::runtime_error(m.str());
  }
  if (m_weight.back().extent(1) != output.extent(0)) {//checks output
    boost::format m("mismatch on the output dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.back().extent(1) % output.extent(0);
    throw std::runtime_error(m.str());
  }
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
  if (m_weight.front().extent(0) != input.extent(1)) {//checks input
    boost::format m("mismatch on the input dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.front().extent(0) % input.extent(1);
    throw std::runtime_error(m.str());
  }
  if (m_weight.back().extent(1) != output.extent(1)) {//checks output
    boost::format m("mismatch on the output dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.back().extent(1) % output.extent(1);
    throw std::runtime_error(m.str());
  }
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
  m_buffer.resize(1);
  m_buffer[0].reference(blitz::Array<double,1>(input));
  setWeights(0);
  setBiases(0);
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
  m_buffer.resize(hidden.size()+1);
  
  //initializes first layer
  m_weight[0].reference(blitz::Array<double,2>(input, hidden[0]));
  m_bias[0].reference(blitz::Array<double,1>(hidden[0]));
  m_buffer[0].reference(blitz::Array<double,1>(input));

  //initializes hidden layers
  const size_t NH1 = hidden.size()-1;
  for (size_t i=0; i<NH1; ++i) {
    m_weight[i+1].reference(blitz::Array<double,2>(hidden[i], hidden[i+1]));
    m_bias[i+1].reference(blitz::Array<double,1>(hidden[i+1]));
    m_buffer[i+1].reference(blitz::Array<double,1>(hidden[i]));
  }

  //initializes the last layer
  m_weight.back().reference(blitz::Array<double,2>(hidden.back(), output));
  m_bias.back().reference(blitz::Array<double,1>(output));
  m_buffer.back().reference(blitz::Array<double,1>(hidden.back()));
  
  setWeights(0);
  setBiases(0);
}

void bob::machine::MLP::resize (const std::vector<size_t>& shape) {

  if (shape.size() < 2) {
    boost::format m("invalid shape for MLP: %d");
    m % shape.size();
    throw std::runtime_error(m.str());
  }
  
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
    boost::format m("mismatch on the input subtraction dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.front().extent(0) % v.extent(0);
    throw std::runtime_error(m.str());
  }
  m_input_sub.reference(bob::core::array::ccopy(v));
}

void bob::machine::MLP::setInputDivision(const blitz::Array<double,1>& v) {
  if (m_weight.front().extent(0) != v.extent(0)) {
    boost::format m("mismatch on the input division dimension: expected a vector with %d positions, but you input %d");
    m % m_weight.front().extent(0) % v.extent(0);
    throw std::runtime_error(m.str());
  }
  m_input_div.reference(bob::core::array::ccopy(v));
}

void bob::machine::MLP::setWeights(const std::vector<blitz::Array<double,2> >& weight) {
  if (m_weight.size() != weight.size()) {
    boost::format m("mismatch on the number of weight layers to set: expected %d layers, but you input %d");
    m % m_weight.size() % weight.size();
  }
  for (size_t i=0; i<m_weight.size(); ++i) {
    if (!bob::core::array::hasSameShape(m_weight[i], weight[i])) {
      boost::format m("mismatch on the shape of weight layer %d");
      m % i;
      throw std::runtime_error(m.str());
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = weight[i];
}

void bob::machine::MLP::setWeights(double v) { 
  for (size_t i=0; i<m_weight.size(); ++i) m_weight[i] = v;
}

void bob::machine::MLP::setBiases(const std::vector<blitz::Array<double,1> >& bias) {
  if (m_bias.size() != bias.size()) {
    boost::format m("mismatch on the number of bias layers to set: expected %d layers, but you input %d");
    m % m_bias.size() % bias.size();
    throw std::runtime_error(m.str());
  }
  for (size_t i=0; i<m_bias.size(); ++i) {
    if (!bob::core::array::hasSameShape(m_bias[i], bias[i])) {
      boost::format m("mismatch on the shape of bias layer %d: expected a vector with length %d, but you input %d");
      m % i % m_bias[i].shape()[0] % bias[i].shape()[0];
      throw std::runtime_error(m.str());
    }
  }
  //if you got to this point, the sizes are correct, just set
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = bias[i];
}

void bob::machine::MLP::setBiases(double v) {
  for (size_t i=0; i<m_bias.size(); ++i) m_bias[i] = v;
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
