/**
 * @file trainer/cxx/WCCNTrainer.cc
 * @date Tue Apr 2 21:08:00 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
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

#include <bob/trainer/WCCNTrainer.h>
#include <bob/math/inv.h>
#include <bob/math/lu.h>
#include <bob/math/stats.h>
#include <boost/make_shared.hpp>


bob::trainer::WCCNTrainer::WCCNTrainer()
{
}

bob::trainer::WCCNTrainer::WCCNTrainer(const bob::trainer::WCCNTrainer& other)
{
}

bob::trainer::WCCNTrainer::~WCCNTrainer() {}

bob::trainer::WCCNTrainer& bob::trainer::WCCNTrainer::operator=
(const bob::trainer::WCCNTrainer& other)
{
  return *this;
}

bool bob::trainer::WCCNTrainer::operator==
  (const bob::trainer::WCCNTrainer& other) const
{
  return true;
}

bool bob::trainer::WCCNTrainer::operator!=
  (const bob::trainer::WCCNTrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::WCCNTrainer::is_similar_to
  (const bob::trainer::WCCNTrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return true;
}


void bob::trainer::WCCNTrainer::train(bob::machine::LinearMachine& machine,
    const std::vector<blitz::Array<double, 2> >& data)
{
  const size_t n_classes = data.size();
  // if #classes < 2, then throw
  if (n_classes < 2) {
    boost::format m("number of classes should be >= 2, but you passed %u");
    m % n_classes;
    throw std::runtime_error(m.str());
  }

  // checks for data type and shape once
  const int n_features = data[0].extent(1);

  for (size_t cl=0; cl<n_classes; ++cl) {
    if (data[cl].extent(1) != n_features) {
      boost::format m("number of features (columns) of array for class %u (%d) does not match that of array for class 0 (%d)");
      m % cl % data[cl].extent(1) % n_features;
      throw std::runtime_error(m.str());
    }
  }

  // machine dimensions
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();

  // Checks that the dimensions are matching
  if ((int)n_inputs != n_features) {
    boost::format m("machine input size (%u) does not match the number of columns in input array (%d)");
    m % n_inputs % n_features;
    throw std::runtime_error(m.str());
  }
  if ((int)n_outputs != n_features) {
    boost::format m("machine output size (%u) does not match the number of columns in output array (%d)");
    m % n_outputs % n_features;
    throw std::runtime_error(m.str());
  }

  // 1. Computes the mean vector and the Scatter matrix Sw and Sb
  blitz::Array<double,1> mean(n_features);
  blitz::Array<double,2> buf1(n_features, n_features); // Sw
  blitz::Array<double,2> buf2(n_features, n_features); // Sb
  bob::math::scatters(data, buf1, buf2, mean); // buf1 = Sw; buf2 = Sb

  // 2. Computes the inverse of (1/N * Sw), Sw is the within-class covariance matrix
  buf1 /= n_classes;
  bob::math::inv(buf1, buf2); // buf2 = (1/N * Sw)^{-1}

  // 3. Computes the Cholesky decomposition of the inverse covariance matrix 
  bob::math::chol(buf2, buf1); //  buf1 = cholesky(buf2)

  // 4. Updates the linear machine
  machine.setInputSubtraction(0); // we do not substract the mean
  machine.setInputDivision(1.);
  machine.setWeights(buf1);
  machine.setBiases(0);
  machine.setActivation(boost::make_shared<bob::machine::IdentityActivation>());
}
