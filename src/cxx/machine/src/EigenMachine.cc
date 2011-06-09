/**
 * @file src/cxx/machine/src/EigenMachine.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to project input on a given subspace.
 */

#include "machine/EigenMachine.h"
#include "machine/Exception.h"
#include "machine/EigenMachineException.h"
#include "core/cast.h"
#include "core/logging.h"
#include "math/linear.h"

Torch::machine::EigenMachine::EigenMachine():
  m_p_variance(0.), m_n_outputs(0), m_eigenvalues(0), m_eigenvectors(0), 
  m_pre_mean(0)
{
}

Torch::machine::EigenMachine::EigenMachine(
    const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors):
  m_p_variance(0.), m_n_outputs(0)
{
  setEigenvaluesvectors(eigenvalues, eigenvectors);
  m_n_outputs = m_eigenvectors.extent(0);
}

Torch::machine::EigenMachine::EigenMachine(
    const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors, int n_outputs):
  m_p_variance(0.)
{
  setEigenvaluesvectors(eigenvalues, eigenvectors);
  if( n_outputs > m_eigenvectors.extent(0))
    throw Torch::machine::EigenMachineNOutputsTooLarge(n_outputs, m_eigenvectors.extent(0));
  else
    m_n_outputs = n_outputs;
}

Torch::machine::EigenMachine::EigenMachine(
    const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors, double p_variance):
  m_p_variance(p_variance)
{
  setEigenvaluesvectors(eigenvalues, eigenvectors);

  // Determine number of outputs to get p_variance
  setPVariance(p_variance);
}

Torch::machine::EigenMachine::EigenMachine(const EigenMachine& other): 
  Machine<blitz::Array<double,1>, blitz::Array<double,1> >(other) 
{
  copy(other);
}

Torch::machine::EigenMachine& Torch::machine::EigenMachine::operator=(const EigenMachine &other) 
{
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // by convention, always return *this
  return *this;
}

void Torch::machine::EigenMachine::copy(const EigenMachine& other) 
{
  m_p_variance = other.m_p_variance;
  m_n_outputs = other.m_n_outputs;
  setEigenvaluesvectors(other.m_eigenvalues, other.m_eigenvectors);
}

Torch::machine::EigenMachine::~EigenMachine() 
{
}

void Torch::machine::EigenMachine::setNOutputs(int n_outputs) 
{
  if( n_outputs > m_eigenvectors.extent(0))
    throw Torch::machine::EigenMachineNOutputsTooLarge(n_outputs, m_eigenvectors.extent(0));
  else
    m_n_outputs = n_outputs;
}

void Torch::machine::EigenMachine::setPVariance(double p_variance) 
{
  double current_var = 0.;
  int current_index = 0;
  while(current_var < m_p_variance)
  {
    if( current_index >= m_eigenvalues.extent(0) )
      throw Torch::machine::EigenMachineNOutputsTooLarge(current_index+1, m_eigenvectors.extent(0));
    current_var += m_eigenvalues(current_index);
    ++current_index;
  }

  setNOutputs(current_index);
}

void Torch::machine::EigenMachine::setEigenvaluesvectors( 
  const blitz::Array<double,1>& eigenvalues, 
  const blitz::Array<double,2>& eigenvectors)
{
  if( eigenvectors.extent(0) != eigenvalues.extent(0) )
    throw Torch::machine::NOutputsMismatch(eigenvectors.extent(0), eigenvalues.extent(0));
  m_eigenvalues.resize(eigenvalues.shape());
  m_eigenvalues = eigenvalues;
  m_eigenvectors.resize(eigenvectors.shape());
  m_eigenvectors = eigenvectors;

  m_pre_mean.resize(eigenvectors.extent(1));
  m_pre_mean = 0.;

  if( m_n_outputs == 0 || m_n_outputs>m_eigenvectors.extent(0) )
    m_n_outputs = m_eigenvectors.extent(0);
}

int Torch::machine::EigenMachine::getNOutputs() const 
{
  return m_n_outputs;
}

double Torch::machine::EigenMachine::getPVariance() const 
{
  return m_p_variance;
}

const blitz::Array<double,1>& Torch::machine::EigenMachine::getEigenvalues() const
{
  return m_eigenvalues;
}

const blitz::Array<double,2>& Torch::machine::EigenMachine::getEigenvectors() const
{
  return m_eigenvectors;
}

void Torch::machine::EigenMachine::setPreMean( const blitz::Array<double,1>& pre_mean)
{
  m_pre_mean.resize(pre_mean.shape());
  if( m_eigenvectors.extent(1) != m_pre_mean.extent(0) )
    throw Torch::machine::NInputsMismatch(m_eigenvectors.extent(1), m_pre_mean.extent(0));
  m_pre_mean = pre_mean;
}
 
const blitz::Array<double,1>& Torch::machine::EigenMachine::getPreMean() const
{
  return m_pre_mean;
}

void Torch::machine::EigenMachine::forward(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const
{
  const blitz::Array<double,2> mat=m_eigenvectors(blitz::Range(0,m_n_outputs-1),blitz::Range::all());
  output.resize(m_n_outputs);
  blitz::Array<double,1> input_nomean(m_pre_mean.extent(0));
  input_nomean = input - m_pre_mean;
  Torch::math::prod(mat, input_nomean, output);
}

namespace Torch {
  namespace machine {
    std::ostream& operator<<(std::ostream& os, const EigenMachine& machine) {
      os << "Output dimensionality = " << machine.m_n_outputs << std::endl;
      os << "Eigenvalues = " << std::endl << machine.m_eigenvalues << std::endl;
      os << "Eigenvectors = " << std::endl << machine.m_eigenvectors << std::endl;

      return os;
    }
  }
}
