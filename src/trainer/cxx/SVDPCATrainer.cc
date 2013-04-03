/**
 * @file trainer/cxx/SVDPCATrainer.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Principal Component Analysis implemented with Singular Value
 * Decomposition (lapack). Implementation.
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

#include <bob/trainer/SVDPCATrainer.h>
#include <bob/machine/Exception.h>
#include <bob/math/svd.h>
#include <algorithm>
#include <blitz/array.h>

bob::trainer::SVDPCATrainer::SVDPCATrainer()
{
}

bob::trainer::SVDPCATrainer::SVDPCATrainer(const bob::trainer::SVDPCATrainer& other)
{
}

bob::trainer::SVDPCATrainer::~SVDPCATrainer() {}

bob::trainer::SVDPCATrainer& bob::trainer::SVDPCATrainer::operator=
(const bob::trainer::SVDPCATrainer& other) 
{
  return *this;
}

bool bob::trainer::SVDPCATrainer::operator==
  (const bob::trainer::SVDPCATrainer& other) const
{
  return true;
}

bool bob::trainer::SVDPCATrainer::operator!=
  (const bob::trainer::SVDPCATrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::SVDPCATrainer::is_similar_to
  (const bob::trainer::SVDPCATrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return true;
}

void bob::trainer::SVDPCATrainer::train(bob::machine::LinearMachine& machine, 
  blitz::Array<double,1>& eigen_values, const blitz::Array<double,2>& ar)
{
  // data is checked now and conforms, just proceed w/o any further checks.
  const size_t n_samples = ar.extent(0);
  const size_t n_features = ar.extent(1);
  const int n_sigma = (int)std::min(n_features, n_samples);

  // Checks that the dimensions are matching 
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();
  const int n_eigenvalues = eigen_values.extent(0);

  // Checks that the dimensions are matching
  if (n_inputs != n_features)
    throw bob::machine::NInputsMismatch(n_inputs, n_features);
  if (n_outputs != (size_t)n_sigma)
    throw bob::machine::NOutputsMismatch(n_outputs, (size_t)n_sigma);
  if (n_eigenvalues != n_sigma)
    throw bob::machine::NOutputsMismatch(n_eigenvalues, n_sigma);

  // removes the empirical mean from the training data
  blitz::Array<double,2> data(n_features, n_samples);
  blitz::Range a = blitz::Range::all();
  for (size_t i=0; i<n_samples; ++i) data(a,i) = ar(i,a);

  // computes the mean of the training data
  blitz::secondIndex j;
  blitz::Array<double,1> mean(n_features);
  mean = blitz::mean(data, j);

  // applies the training data mean
  for (size_t i=0; i<n_samples; ++i) data(a,i) -= mean;

  /**
   * computes the singular value decomposition using lapack
   *
   * note: Lapack already organizes the U,Sigma,V**T matrixes so that the
   * singular values in Sigma are organized by decreasing order of magnitude.
   * You **don't** need sorting after this.
   */
  blitz::Array<double,2> U(n_features, n_sigma);
  blitz::Array<double,1> sigma(n_sigma);
  bob::math::svd_(data, U, sigma);

  /**
   * sets the linear machine with the results:
   *
   * note: eigen values are sigma^2/n_samples diagonal
   *       eigen vectors are the rows of U
   */
  machine.resize(n_features, n_sigma);
  machine.setInputSubtraction(mean);
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
  machine.setWeights(U);

  //weight normalization (if necessary):
  //norm_factor = blitz::sum(blitz::pow2(V(all,i)))

  // finally, we set also the eigen values in this version
  eigen_values = blitz::pow2(sigma)/(n_samples-1);
}

void bob::trainer::SVDPCATrainer::train(bob::machine::LinearMachine& machine, 
  const blitz::Array<double,2>& ar)
{
  const int n_sigma = std::min(ar.extent(0),ar.extent(1));
  blitz::Array<double,1> throw_away(n_sigma);
  train(machine, throw_away, ar);
}
