/**
 * @file trainer/cxx/CovMatrixPCATrainer.cc
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

#include <algorithm>
#include <blitz/array.h>
#include <boost/format.hpp>

#include <bob/math/stats.h>
#include <bob/math/eig.h>
#include <bob/trainer/CovMatrixPCATrainer.h>

bob::trainer::CovMatrixPCATrainer::CovMatrixPCATrainer()
{
}

bob::trainer::CovMatrixPCATrainer::CovMatrixPCATrainer(const bob::trainer::CovMatrixPCATrainer& other)
{
}

bob::trainer::CovMatrixPCATrainer::~CovMatrixPCATrainer() {}

bob::trainer::CovMatrixPCATrainer& bob::trainer::CovMatrixPCATrainer::operator=
(const bob::trainer::CovMatrixPCATrainer& other) 
{
  return *this;
}

bool bob::trainer::CovMatrixPCATrainer::operator==
  (const bob::trainer::CovMatrixPCATrainer& other) const
{
  return true;
}

bool bob::trainer::CovMatrixPCATrainer::operator!=
  (const bob::trainer::CovMatrixPCATrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::CovMatrixPCATrainer::is_similar_to
  (const bob::trainer::CovMatrixPCATrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return true;
}

/**
 * Returns the indexes for sorting a given blitz::Array<double,1>
 */
struct compare_1d_blitz {
  const blitz::Array<double,1>& v_;
  compare_1d_blitz(const blitz::Array<double,1>& v): v_(v) { }
  bool operator() (size_t i, size_t j) { return v_(i) >= v_(j); }
};

static std::vector<size_t> sort_indexes(const blitz::Array<double,1>& v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), compare_1d_blitz(v));

  return idx;
}

void bob::trainer::CovMatrixPCATrainer::train(bob::machine::LinearMachine& machine, blitz::Array<double,1>& eigen_values, const blitz::Array<double,2>& ar) {

  // data is checked now and conforms, just proceed w/o any further checks.
  const size_t n_samples = ar.extent(0);
  const size_t n_features = ar.extent(1);
  const int n_sigma = (int)std::min(n_samples-1, n_features);

  // Checks that the dimensions are matching 
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();
  const int n_eigenvalues = eigen_values.extent(0);

  // Checks that the dimensions are matching
  if (n_inputs != n_features) {
    boost::format m("Number of features at input data set (%d columns) does not match machine input size (%d)");
    m % n_features % n_inputs;
    throw std::runtime_error(m.str());
  }
  if (n_outputs != (size_t)n_sigma) {
    boost::format m("Number of outputs (%d) does not match min(#features-1,#samples) = min(%d, %d) = %d");
    m % n_outputs % n_features % n_samples % n_sigma;
    throw std::runtime_error(m.str());
  }
  if (n_eigenvalues != n_sigma) {
    boost::format m("Number of eigenvalues placeholder (%d) does not match min(#features-1,#samples) = min(%d,%d) = %d");
    m % n_eigenvalues % n_features % n_samples % n_sigma;
    throw std::runtime_error(m.str());
  }

  /**
   * computes the covariance matrix (X-mu)(X-mu)^T / (len(X)-1) and then solves
   * the generalized eigen-value problem taking into consideration the
   * covariance matrix is symmetric (and, by extension, hermitian).
   */
  blitz::Array<double,1> mean(n_features);
  blitz::Array<double,2> Sigma(n_features, n_features);
  bob::math::scatter_(ar.transpose(1,0), Sigma, mean);
  Sigma /= (n_samples-1); //unbiased variance estimator

  blitz::Array<double,2> U(n_features, n_features);
  blitz::Array<double,1> e(n_features);
  bob::math::eigSym_(Sigma, U, e);

  /**
   * sorts eigenvectors using a decreasing eigen value priority
   */
  blitz::Array<double,2> SortedU(n_features, n_sigma);
  blitz::Range a = blitz::Range::all();
  std::vector<size_t> order = sort_indexes(e);
  size_t j = 0;
  for (int i=0; i<n_sigma; ++i) {
    eigen_values(j) = e(order[i]);
    SortedU(a,j) = U(a,order[i]);
    ++j;
  }

  /**
   * sets the linear machine with the results:
   */
  machine.setInputSubtraction(mean);
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
  machine.setWeights(SortedU);
}

void bob::trainer::CovMatrixPCATrainer::train(bob::machine::LinearMachine& machine, 
  const blitz::Array<double,2>& ar)
{
  const int n_sigma = std::min(ar.extent(0),ar.extent(1));
  blitz::Array<double,1> throw_away(n_sigma);
  train(machine, throw_away, ar);
}
