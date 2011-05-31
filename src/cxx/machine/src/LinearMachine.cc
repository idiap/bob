/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 30 May 2011 13:01:33 CEST
 *
 * @brief Implements a LinearMachine
 */

#include "machine/LinearMachine.h"
#include "machine/Exception.h"

namespace mach = Torch::machine;
      
mach::LinearMachine::LinearMachine(const blitz::Array<double,2>& weight, 
    const blitz::Array<double,1>& bias)
{
  if (weight.extent(1) != bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(weight.extent(1), bias.extent(0));
  }
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}

mach::LinearMachine::LinearMachine(int n_input, int n_output):
  m_weight(n_output, n_input),
  m_bias(n_output)
{
  m_weight = 0;
  m_bias = 0;
}

mach::LinearMachine::LinearMachine(const mach::LinearMachine& other) {
  //no need to check, it was done at the other machine.
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}

mach::LinearMachine::~LinearMachine() {}

mach::LinearMachine& mach::LinearMachine::operator=(const mach::LinearMachine& other) {
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
  return *this;
}

void Torch::machine::LinearMachine::forward
(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const {
  if (m_weight.extent(0) != output.extent(0))
    throw Torch::machine::NOutputsMismatch(m_weight.extent(0), output.extent(0));
  if (m_weight.extent(1) != input.extent(0))
    throw Torch::machine::NInputsMismatch(m_weight.extent(1), input.extent(0));

  blitz::Range a = blitz::Range::all();
  for (int i=0; i<m_weight.extent(0); ++i)
    output(i) = blitz::sum(m_weight(i,a)*m_input) + m_bias(i);
}

void Torch::machine::LinearMachine::setWeightsAndBiases
(const blitz::Array<double,2>& weight, const blitz::Array<double,1>& bias) {
  if (weight.extent(1) != bias.extent(0)) {
    throw Torch::machine::NInputsMismatch(weight.extent(1), bias.extent(0));
  }
  m_weight.reference(weight.copy());
  m_bias.reference(bias.copy());
}
