/**
 * @file machine/cxx/TwoDPCAMachine.cc
 * @date Wed May 18 21:51:16 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a 2DPCA machine implementation
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

#include "bob/machine/TwoDPCAMachine.h"
#include "bob/machine/Exception.h"
#include "bob/machine/EigenMachineException.h"
#include "bob/core/cast.h"
#include "bob/core/logging.h"
#include "bob/math/linear.h"

bob::machine::TwoDPCAMachine::TwoDPCAMachine():
  m_dim_outputs(0), m_p_variance(0.), m_n_outputs(0), m_eigenvalues(0), 
  m_eigenvectors(0), m_pre_mean(0)
{
}

bob::machine::TwoDPCAMachine::TwoDPCAMachine(int dim_outputs):
  m_dim_outputs(dim_outputs), m_p_variance(0.), m_n_outputs(0), 
  m_eigenvalues(0), m_eigenvectors(0), m_pre_mean(0)
{
}

bob::machine::TwoDPCAMachine::TwoDPCAMachine(
    int dim_outputs, const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors):
  m_dim_outputs(dim_outputs), m_p_variance(0.), m_n_outputs(0)
{
  setEigenvaluesvectors(eigenvalues, eigenvectors);
  m_n_outputs = m_eigenvectors.extent(1);
}

bob::machine::TwoDPCAMachine::TwoDPCAMachine(
    int dim_outputs, const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors, int n_outputs):
  m_dim_outputs(dim_outputs), m_p_variance(0.)
{
  if( n_outputs > m_eigenvectors.extent(1))
    throw bob::machine::EigenMachineNOutputsTooLarge(n_outputs, m_eigenvectors.extent(1));
  else
    m_n_outputs = n_outputs;
  setEigenvaluesvectors(eigenvalues, eigenvectors);
}

bob::machine::TwoDPCAMachine::TwoDPCAMachine(
    int dim_outputs, const blitz::Array<double,1>& eigenvalues, 
    const blitz::Array<double,2>& eigenvectors, double p_variance):
  m_dim_outputs(dim_outputs), m_p_variance(p_variance)
{
  setEigenvaluesvectors(eigenvalues, eigenvectors);

  // Determine number of outputs to get p_variance
  setPVariance(p_variance);
}

bob::machine::TwoDPCAMachine::TwoDPCAMachine(const TwoDPCAMachine& other): 
  Machine<blitz::Array<double,2>, blitz::Array<double,2> >(other) 
{
  copy(other);
}

bob::machine::TwoDPCAMachine& bob::machine::TwoDPCAMachine::operator=(const TwoDPCAMachine &other) 
{
  // protect against invalid self-assignment
  if (this != &other) {
    copy(other);
  }
  
  // by convention, always return *this
  return *this;
}

void bob::machine::TwoDPCAMachine::copy(const TwoDPCAMachine& other) 
{
  m_dim_outputs = other.m_dim_outputs;
  m_p_variance = other.m_p_variance;
  m_n_outputs = other.m_n_outputs;
  setEigenvaluesvectors(other.m_eigenvalues, other.m_eigenvectors);
}

bob::machine::TwoDPCAMachine::~TwoDPCAMachine() 
{
}

void bob::machine::TwoDPCAMachine::setDimOutputs(int dim_outputs) 
{
  m_dim_outputs = dim_outputs;
}

void bob::machine::TwoDPCAMachine::setNOutputs(int n_outputs) 
{
  if( n_outputs > m_eigenvectors.extent(0))
    throw bob::machine::EigenMachineNOutputsTooLarge(n_outputs, m_eigenvectors.extent(0));
  else
    m_n_outputs = n_outputs;
}

void bob::machine::TwoDPCAMachine::setPVariance(double p_variance) 
{
  double current_var = 0.;
  int current_index = 0;
  while(current_var < m_p_variance)
  {
    if( current_index >= m_eigenvalues.extent(0) )
      throw bob::machine::EigenMachineNOutputsTooLarge(current_index+1, m_eigenvalues.extent(0));
    current_var += m_eigenvalues(current_index);
    ++current_index;
  }

  setNOutputs(current_index);
}

void bob::machine::TwoDPCAMachine::setEigenvaluesvectors( 
  const blitz::Array<double,1>& eigenvalues, 
  const blitz::Array<double,2>& eigenvectors)
{
  if( eigenvectors.extent(1) != eigenvalues.extent(0) )
    throw bob::machine::NOutputsMismatch(eigenvectors.extent(1), eigenvalues.extent(0));
  m_eigenvalues.resize(eigenvalues.shape());
  m_eigenvalues = eigenvalues;
  m_eigenvectors.resize(eigenvectors.shape());
  m_eigenvectors = eigenvectors;

  m_pre_mean.resize(m_pre_mean.extent(0),eigenvectors.extent(0));
  m_pre_mean = 0.;

  if( m_n_outputs == 0 || m_n_outputs>m_eigenvectors.extent(1) )
    m_n_outputs = m_eigenvectors.extent(1);
}

int bob::machine::TwoDPCAMachine::getDimOutputs() const 
{
  return m_dim_outputs;
}

int bob::machine::TwoDPCAMachine::getNOutputs() const 
{
  return m_n_outputs;
}

double bob::machine::TwoDPCAMachine::getPVariance() const 
{
  return m_p_variance;
}

const blitz::Array<double,1>& bob::machine::TwoDPCAMachine::getEigenvalues() const
{
  return m_eigenvalues;
}

const blitz::Array<double,2>& bob::machine::TwoDPCAMachine::getEigenvectors() const
{
  return m_eigenvectors;
}

void bob::machine::TwoDPCAMachine::setPreMean( const blitz::Array<double,2>& pre_mean)
{
  m_pre_mean.resize(pre_mean.shape());
  if( m_dim_outputs != m_pre_mean.extent(1) )
    throw bob::machine::NInputsMismatch(m_dim_outputs, m_pre_mean.extent(0));
  if( m_eigenvectors.extent(0) != m_pre_mean.extent(1) )
    throw bob::machine::NInputsMismatch(m_eigenvectors.extent(0), m_pre_mean.extent(1));
  m_pre_mean = pre_mean;
}
 
const blitz::Array<double,2>& bob::machine::TwoDPCAMachine::getPreMean() const
{
  return m_pre_mean;
}

void bob::machine::TwoDPCAMachine::forward(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const
{
  output.resize(m_dim_outputs, m_n_outputs);
  const blitz::Array<double,2> mat=m_eigenvectors(blitz::Range::all(), blitz::Range(0,m_n_outputs-1));
  blitz::Array<double,2> input_nomean(m_pre_mean.extent(0), m_pre_mean.extent(1));
  input_nomean = input - m_pre_mean;
  bob::math::prod(input_nomean, mat, output);
}

void bob::machine::TwoDPCAMachine::print() const 
{
  bob::core::info << "Output dimensionality = " << m_dim_outputs << "x" << m_n_outputs << std::endl;
  bob::core::info << "Eigenvalues = " << std::endl << m_eigenvalues << std::endl;
  bob::core::info << "Eigenvectors = " << std::endl << m_eigenvectors << std::endl;
}

