/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 30 May 2011 13:01:33 CEST
 *
 * @brief Implements a LinearMachine
 */

#include "database/Arrayset.h"
#include "machine/LinearMachine.h"
#include "machine/Exception.h"

namespace mach = Torch::machine;
      
mach::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight, 
    const blitz::Array<double,1>& bias)
{
  if (weight.extent(0) != bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(weight.extent(0), bias.extent(0));
  }
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}

mach::LinearMachine::LinearMachine(size_t n_input, size_t n_output):
  m_weight(n_output, n_input),
  m_bias(n_output)
{
  m_weight = 0;
  m_bias = 0;
}

mach::LinearMachine::LinearMachine(const mach::LinearMachine& other) {
  //no need to check, it was done at the other machine.
  m_weight.reference(other.m_weight.copy());
  m_bias.reference(other.m_bias.copy());
}

mach::LinearMachine::LinearMachine (const Torch::config::Configuration& config) {
  load(config);
}

mach::LinearMachine::~LinearMachine() {}

mach::LinearMachine& mach::LinearMachine::operator=
(const mach::LinearMachine& other) {
  m_weight.reference(other.m_weight.copy());
  m_bias.reference(other.m_bias.copy());
  return *this;
}

void mach::LinearMachine::load (const Torch::config::Configuration& config) {
  setWeightsAndBiases
    (config.get<Torch::database::Arrayset>("weights").get<double,2>(1),
     config.get<Torch::database::Arrayset>("biases").get<double,1>(1));
}

void mach::LinearMachine::save (Torch::config::Configuration& config) const {
  config.set("weights", Torch::database::Array(m_weight));
  config.set("biases", Torch::database::Array(m_bias));
}

void Torch::machine::LinearMachine::forward
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  if (m_weight.extent(0) != output.extent(0))
    throw Torch::machine::NOutputsMismatch(m_weight.extent(0),
        output.extent(0));
  if (m_weight.extent(1) != input.extent(0))
    throw Torch::machine::NInputsMismatch(m_weight.extent(1),
        input.extent(0));

  blitz::Range a = blitz::Range::all();
  for (int i=0; i<m_weight.extent(0); ++i)
    output(i) = blitz::sum(m_weight(i,a)*input) + m_bias(i);
}

void Torch::machine::LinearMachine::setWeights
(const blitz::Array<double,2>& weight) {
  if (weight.extent(0) != m_bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(weight.extent(0), m_bias.extent(0));
  }
  m_weight.reference(weight.copy());
}

void Torch::machine::LinearMachine::setBiases
(const blitz::Array<double,1>& bias) {
  if (m_weight.extent(0) != bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(m_weight.extent(0), bias.extent(0));
  }
  m_bias.reference(bias.copy());
}

void Torch::machine::LinearMachine::setWeightsAndBiases
(const blitz::Array<double,2>& weight, const blitz::Array<double,1>& bias) {
  if (weight.extent(0) != bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(weight.extent(0), bias.extent(0));
  }
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}
