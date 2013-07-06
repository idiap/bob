/**
 * @file trainer/cxx/WhiteningTrainer.cc
 * @date Tue Apr 2 21:08:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <boost/make_shared.hpp>

#include <bob/trainer/WhiteningTrainer.h>
#include <bob/math/inv.h>
#include <bob/math/lu.h>
#include <bob/math/stats.h>

bob::trainer::WhiteningTrainer::WhiteningTrainer()
{
}

bob::trainer::WhiteningTrainer::WhiteningTrainer(const bob::trainer::WhiteningTrainer& other)
{
}

bob::trainer::WhiteningTrainer::~WhiteningTrainer() {}

bob::trainer::WhiteningTrainer& bob::trainer::WhiteningTrainer::operator=
(const bob::trainer::WhiteningTrainer& other)
{
  return *this;
}

bool bob::trainer::WhiteningTrainer::operator==
  (const bob::trainer::WhiteningTrainer& other) const
{
  return true;
}

bool bob::trainer::WhiteningTrainer::operator!=
  (const bob::trainer::WhiteningTrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::WhiteningTrainer::is_similar_to
  (const bob::trainer::WhiteningTrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return true;
}

void bob::trainer::WhiteningTrainer::train(bob::machine::LinearMachine& machine, 
  const blitz::Array<double,2>& ar)
{
  // training data dimensions
  const size_t n_samples = ar.extent(0);
  const size_t n_features = ar.extent(1);
  // machine dimensions
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();

  // Checks that the dimensions are matching
  if (n_inputs != n_features) {
    boost::format m("machine input size (%u) does not match the number of columns in input array (%d)");
    m % n_inputs % n_features;
    throw std::runtime_error(m.str());
  }
  if (n_outputs != n_features) {
    boost::format m("machine output size (%u) does not match the number of columns in output array (%d)");
    m % n_outputs % n_features;
    throw std::runtime_error(m.str());
  }

  // 1. Computes the mean vector and the covariance matrix of the training set
  blitz::Array<double,1> mean(n_features);
  blitz::Array<double,2> cov(n_features,n_features);
  bob::math::scatter(ar, cov, mean);
  cov /= (double)(n_samples-1);

  // 2. Computes the inverse of the covariance matrix
  blitz::Array<double,2> icov(n_features,n_features);
  bob::math::inv(cov, icov);

  // 3. Computes the Cholesky decomposition of the inverse covariance matrix 
  blitz::Array<double,2> whiten(n_features,n_features);
  bob::math::chol(icov, whiten);

  // 4. Updates the linear machine
  machine.setInputSubtraction(mean);
  machine.setInputDivision(1.);
  machine.setWeights(whiten);
  machine.setBiases(0);
  machine.setActivation(boost::make_shared<bob::machine::IdentityActivation>());
}
