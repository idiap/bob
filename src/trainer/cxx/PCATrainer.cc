/**
 * @file trainer/cxx/PCATrainer.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Principal Component Analysis implemented with Singular Value
 * Decomposition or using the Covariance Method. Both are implemented using
 * LAPACK. Implementation.
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

#include <algorithm>
#include <blitz/array.h>
#include <boost/format.hpp>
#include <bob/math/stats.h>
#include <bob/math/svd.h>
#include <bob/math/eig.h>
#include <bob/trainer/PCATrainer.h>

bob::trainer::PCATrainer::PCATrainer(bool use_svd)
  : m_use_svd(use_svd)
{
}

bob::trainer::PCATrainer::PCATrainer(const bob::trainer::PCATrainer& other)
  : m_use_svd(other.m_use_svd)
{
}

bob::trainer::PCATrainer::~PCATrainer() {}

bob::trainer::PCATrainer& bob::trainer::PCATrainer::operator=
(const bob::trainer::PCATrainer& other)
{
  m_use_svd = other.m_use_svd;
  return *this;
}

bool bob::trainer::PCATrainer::operator==
  (const bob::trainer::PCATrainer& other) const
{
  return m_use_svd == other.m_use_svd;
}

bool bob::trainer::PCATrainer::operator!=
  (const bob::trainer::PCATrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::PCATrainer::is_similar_to
  (const bob::trainer::PCATrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return this->operator==(other);
}

/**
 * Sets up the machine calculating the PC's via the Covariance Matrix
 */
static void pca_via_covmat(
    bob::machine::LinearMachine& machine,
    blitz::Array<double,1>& eigen_values, 
    const blitz::Array<double,2>& X,
    int rank
    ) {
  /**
   * computes the covariance matrix (X-mu)(X-mu)^T / (len(X)-1) and then solves
   * the generalized eigen-value problem taking into consideration the
   * covariance matrix is symmetric (and, by extension, hermitian).
   */
  blitz::Array<double,1> mean(X.extent(1));
  blitz::Array<double,2> Sigma(X.extent(1), X.extent(1));
  bob::math::scatter_(X.transpose(1,0), Sigma, mean);
  Sigma /= (X.extent(0)-1); //unbiased variance estimator

  blitz::Array<double,2> U(X.extent(1), X.extent(1));
  blitz::Array<double,1> e(X.extent(1));
  bob::math::eigSym_(Sigma, U, e);
  e.reverseSelf(0);
  U.reverseSelf(1);

  /**
   * sets the linear machine with the results:
   */
  machine.setInputSubtraction(mean);
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
  if (e.size() == eigen_values.size()) {
    eigen_values = e;
    machine.setWeights(U);
  }
  else {
    eigen_values = e(blitz::Range(0,rank-1));
    machine.setWeights(U(blitz::Range::all(), blitz::Range(0,rank-1)));
  }
}

/**
 * Sets up the machine calculating the PC's via SVD
 */
static void pca_via_svd(
    bob::machine::LinearMachine& machine,
    blitz::Array<double,1>& eigen_values, 
    const blitz::Array<double,2>& X,
    int rank
    ) {

  // removes the empirical mean from the training data
  blitz::Array<double,2> data(X.extent(1), X.extent(0));
  blitz::Range a = blitz::Range::all();
  for (int i=0; i<X.extent(0); ++i) data(a,i) = X(i,a);

  // computes the mean of the training data
  blitz::secondIndex j;
  blitz::Array<double,1> mean(X.extent(1));
  mean = blitz::mean(data, j);

  // applies the training data mean
  for (int i=0; i<X.extent(0); ++i) data(a,i) -= mean;

  /**
   * computes the singular value decomposition using lapack
   *
   * note: Lapack already organizes the U,Sigma,V**T matrixes so that the
   * singular values in Sigma are organized by decreasing order of magnitude.
   * You **don't** need sorting after this.
   */
  const int rank_1 = (rank == (int)X.extent(1))? X.extent(1) : X.extent(0);
  blitz::Array<double,2> U(X.extent(1), rank_1);
  blitz::Array<double,1> sigma(rank_1);
  bob::math::svd_(data, U, sigma);

  /**
   * sets the linear machine with the results:
   *
   * note: eigen values are sigma^2/X.extent(0) diagonal
   *       eigen vectors are the rows of U
   */
  machine.setInputSubtraction(mean);
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
  blitz::Range up_to_rank(0, rank-1);
  machine.setWeights(U(a,up_to_rank));

  //weight normalization (if necessary):
  //norm_factor = blitz::sum(blitz::pow2(V(all,i)))

  // finally, we set also the eigen values in this version
  eigen_values = (blitz::pow2(sigma)/(X.extent(0)-1))(up_to_rank);
}

void bob::trainer::PCATrainer::train(bob::machine::LinearMachine& machine,
  blitz::Array<double,1>& eigen_values, const blitz::Array<double,2>& X) const
{
  // data is checked now and conforms, just proceed w/o any further checks.
  const int rank = output_size(X);

  // Checks that the dimensions are matching
  if (machine.inputSize() != (size_t)X.extent(1)) {
    boost::format m("Number of features at input data set (%d columns) does not match machine input size (%d)");
    m % X.extent(1) % machine.inputSize();
    throw std::runtime_error(m.str());
  }
  if (machine.outputSize() != (size_t)rank) {
    boost::format m("Number of outputs of the given machine (%d) does not match the maximum covariance rank, i.e., min(#samples-1,#features) = min(%d, %d) = %d");
    m % machine.outputSize() % (X.extent(0)-1) % X.extent(1) % rank;
    throw std::runtime_error(m.str());
  }
  if (eigen_values.extent(0) != rank) {
    boost::format m("Number of eigenvalues on the given 1D array (%d) does not match the maximum covariance rank, i.e., min(#samples-1,#features) = min(%d,%d) = %d");
    m % eigen_values.extent(0) % (X.extent(0)-1) % X.extent(1) % rank;
    throw std::runtime_error(m.str());
  }

  if (m_use_svd) pca_via_svd(machine, eigen_values, X, rank);
  else pca_via_covmat(machine, eigen_values, X, rank);
}

void bob::trainer::PCATrainer::train(bob::machine::LinearMachine& machine,
  const blitz::Array<double,2>& X) const
{
  blitz::Array<double,1> throw_away_eigen_values(output_size(X));
  train(machine, throw_away_eigen_values, X);
}

size_t bob::trainer::PCATrainer::output_size
(const blitz::Array<double,2>& X) const{
  return (size_t)std::min(X.extent(0)-1,X.extent(1));
}
