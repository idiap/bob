/**
 * @file cxx/machine/src/BICMachine.cc
 * @date Tue Jun  5 16:54:27 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * A machine that implements the liner projection of input to the output using
 * weights, biases and sums:
 * output = sum(inputs * weights) + bias
 * It is possible to setup the machine to previously normalize the input taking
 * into consideration some input bias and division factor. It is also possible
 * to set it up to have an activation function.
 * A linear classifier. See C. M. Bishop, "Pattern Recognition and Machine
 * Learning", chapter 4
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

#include <machine/BICMachine.h>
#include <math/linear.h>
#include <core/array_assert.h>

static double sqr(const double& x){
  return x*x;
}

static double sqr_norm(const blitz::Array<double,1> array){
  return blitz::sum(blitz::pow2(array));
}

/**
 * Initializes an empty BIC Machine
 *
 * @param use_DFFS  Add the Distance From Feature Space during score computation?
 */
bob::machine::BICMachine::BICMachine(bool use_DFFS)
:
  m_project_data(use_DFFS),
  m_use_DFFS(use_DFFS)
{}


void bob::machine::BICMachine::initialize(bool clazz, int input_length, int projected_length){
  blitz::Array<double,1>& diff = clazz ? m_diff_E : m_diff_I;
  blitz::Array<double,1>& proj = clazz ? m_proj_E : m_proj_I;
  diff.resize(input_length);
  proj.resize(projected_length);
}

/**
 * Sets the parameters of the given class that are required for computing the IEC scores (Guenther, Wuertz)
 *
 * @param  clazz   false for the intrapersonal class, true for the extrapersonal one.
 * @param  mean    The mean vector of the training data
 * @param  variances  The variances of the training data
 */
void bob::machine::BICMachine::setIEC(
    bool clazz,
    const blitz::Array<double,1>& mean,
    const blitz::Array<double,1>& variances
){
  m_project_data = false;
  // select the right matrices to write
  blitz::Array<double,1>& mu = clazz ? m_mu_E : m_mu_I;
  blitz::Array<double,1>& lambda = clazz ? m_lambda_E : m_lambda_I;

  // copy mean and variances
  mu.reference(mean);
  lambda.reference(variances);
}

/**
 * Sets the parameters of the given class that are required for computing the BIC scores (Teixeira)
 *
 * @param  clazz   false for the intrapersonal class, true for the extrapersonal one.
 * @param  mean    The mean vector of the training data
 * @param  variances  The eigenvalues of the training data
 * @param  projection  The PCA projection matrix
 * @param  rho     The residual eigenvalues, used for DFFS calculation
 */
void bob::machine::BICMachine::setBIC(
    bool clazz,
    const blitz::Array<double,1>& mean,
    const blitz::Array<double,1>& variances,
    const blitz::Array<double,2>& projection,
    const double rho
){
  m_project_data = true;
  // select the right matrices to write
  blitz::Array<double,1>& mu = clazz ? m_mu_E : m_mu_I;
  blitz::Array<double,1>& lambda = clazz ? m_lambda_E : m_lambda_I;
  blitz::Array<double,2>& Phi = clazz ? m_Phi_E : m_Phi_I;
  double& rho_ = clazz ? m_rho_E : m_rho_I;

  // copy information
  mu.reference(mean);
  lambda.reference(variances);
  Phi.resize(projection.shape());
  Phi = projection;
  Phi.transposeSelf(1,0);
  rho_ = rho;

  // check that rho has a reasonable value (if it is used)
  if (m_use_DFFS && rho_ < 1e-12) throw bob::machine::ZeroEigenvalueException();

  // initialize temporaries
  initialize(clazz, Phi.shape()[1], Phi.shape()[0]);
}

/**
 * Set or unset the usage of the Distance From Feature Space
 *
 * @param  The new value of use_DFFS
 */
void bob::machine::BICMachine::use_DFFS(bool use_DFFS){
  m_use_DFFS = use_DFFS;
  if (m_project_data && m_use_DFFS && (m_rho_E < 1e-12 || m_rho_I < 1e-12)) throw bob::machine::ZeroEigenvalueException();
}

/**
 * Loads the BICMachine from the given hdf5 file.
 *
 * @param  config  The hdf5 file containing the required information.
 */
void bob::machine::BICMachine::load(bob::io::HDF5File& config){
  //reads all data directly into the member variables
  m_project_data = config.read<bool>("project_data");
  m_mu_I.reference(config.readArray<double,1>("intra_mean"));
  m_lambda_I.reference(config.readArray<double,1>("intra_variance"));
  if (m_project_data){
    m_use_DFFS = config.read<bool>("use_DFFS");
    m_Phi_I.reference(config.readArray<double,2>("intra_subspace"));
    initialize(false, m_Phi_I.shape()[1], m_Phi_I.shape()[0]);
     m_rho_I = config.read<double>("intra_rho");
  }

  m_mu_E.reference(config.readArray<double,1>("extra_mean"));
  m_lambda_E.reference(config.readArray<double,1>("extra_variance"));
  if (m_project_data){
    m_Phi_E.reference(config.readArray<double,2>("extra_subspace"));
    initialize(true, m_Phi_E.shape()[1], m_Phi_E.shape()[0]);
    m_rho_E = config.read<double>("extra_rho");
  }
  // check that rho has reasonable values
  if (m_project_data && m_use_DFFS && (m_rho_E < 1e-12 || m_rho_I < 1e-12)) throw bob::machine::ZeroEigenvalueException();

}

/**
 * Saves the parameters of the BICMachine to the given hdf5 file.
 *
 * @param  config  The hdf5 file to write the configuration into.
 */
void bob::machine::BICMachine::save(bob::io::HDF5File& config) const{
  config.set("project_data", m_project_data);
  config.setArray("intra_mean", m_mu_I);
  config.setArray("intra_variance", m_lambda_I);
  if (m_project_data){
    config.set("use_DFFS", m_use_DFFS);
    config.setArray("intra_subspace", m_Phi_I);
    config.set("intra_rho", m_rho_I);
  }

  config.setArray("extra_mean", m_mu_E);
  config.setArray("extra_variance", m_lambda_E);
  if (m_project_data){
    config.setArray("extra_subspace", m_Phi_E);
    config.set("extra_rho", m_rho_E);
  }
}

/**
 * Computes the BIC or IEC score for the given input vector.
 * The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class.
 * No sanity checks of input and output are performed.
 *
 * @param  input  A vector (of difference values) to compute the BIC or IEC score for.
 * @param  output The one-element array that will contain the score afterwards.
 */
void bob::machine::BICMachine::forward_(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const{
  double& res = output(0) = 0.;
  if (m_project_data){
    // subtract mean
    m_diff_I = input - m_mu_I;
    m_diff_E = input - m_mu_E;
    // project data to intrapersonal and extrapersonal subspace
    bob::math::prod(m_Phi_I, m_diff_I, m_proj_I);
    bob::math::prod(m_Phi_E, m_diff_E, m_proj_E);

    // compute Mahalanobis distance
    for (int i = m_proj_I.shape()[0]; i--;){
      res += sqr(m_proj_E(i)) / m_lambda_E(i)
          -  sqr(m_proj_I(i)) / m_lambda_I(i);
    }

    // add the DFFS?
    if (m_use_DFFS){
      res += (sqr_norm(m_diff_E) - sqr_norm(m_proj_E)) / m_rho_E
          -  (sqr_norm(m_diff_I) - sqr_norm(m_proj_I)) / m_rho_I;
    }
  } else {
    // forward without projection
    for (int i = input.shape()[0]; i--;){
      res += sqr(input(i) - m_mu_E(i)) / m_lambda_E(i)
          -  sqr(input(i) - m_mu_I(i)) / m_lambda_I(i);
    }
  }
}

/**
 * Computes the BIC or IEC score for the given input vector.
 * The score itself is the log-likelihood score of the given input vector belonging to the intrapersonal class.
 * Sanity checks of input and output shape are performed.
 *
 * @param  input  A vector (of difference values) to compute the BIC or IEC score for.
 * @param  output The one-element array that will contain the score afterwards.
 */
void bob::machine::BICMachine::forward(const blitz::Array<double,1>& input, blitz::Array<double,1>& output) const{
  // perform some checks
  bob::core::array::assertSameShape(input, m_mu_E);
  bob::core::array::assertSameShape(output, blitz::TinyVector<int,1>(1));

  // call the actual method
  forward_(input, output);
}

