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

#include <boost/make_shared.hpp>

#include <bob/trainer/WCCNTrainer.h>
#include <bob/machine/Exception.h>
#include <bob/trainer/Exception.h>
#include <bob/math/inv.h>
#include <bob/math/lu.h>
#include <bob/math/stats.h>

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

/**
 * Evaluates, in a single loop, the overall (or grand) mean 'm', the individual
 * class means 'm_k' and computes the total number of elements in each class
 * 'N'. (This is same function used in FisherLDATrainer.cc)
 */
static void evalMeans(const std::vector<blitz::Array<double,2> >& data,
  blitz::Array<double,1>& m, blitz::Array<double,2>& m_k,
  blitz::Array<double,1>& N)
{
  blitz::Range a = blitz::Range::all();
  for (size_t k=0; k<data.size(); ++k) { //class loop
    N(k) = data[k].extent(0);
    for (int example=0; example<data[k].extent(0); ++example) {
      blitz::Array<double,1> buffer(data[k](example,a));
      m_k(a,k) += buffer;
      m += buffer;
    }

    m_k(a,k) /= N(k);
  }

  m /= sum(N);
}

/**
 * Calculates the within and between class scatter matrices Sw and Sb. Returns
 * those matrices and the overall means vector (m).
 *
 * Strategy implemented:
 * 1. Evaluate the overall mean (m), class means (m_k) and the total class
 *    counts (N).
 * 2. Evaluate Sw and Sb using normal loops.
 *
 * Note that Sw and Sb, in this implementation, will be normalized by N-1
 * (number of samples) and K (number of classes). This procedure makes
 * the eigen values scaled by (N-1)/K, effectively increasing their values. The
 * main motivation for this normalization are numerical precision concerns with
 * the increasing number of samples causing a rather large Sw matrix. A
 * normalization strategy mitigates this problem. The eigen vectors will see
 * no effect on this normalization as they are normalized in the euclidean
 * sense (||a|| = 1) so that does not change those.
 *
 * This method was designed based on the previous design at bob3Vision 2.1,
 * by SM.
 */


/**
 * Evaluates, for testing purposes, the St scatter which is the total scatter
 * for the LDA problem. The total scatter St relates to the within class
 * scatter Sw and the between class scatter Sb in the following manner:
 *
 * St = Sw + Sb (Bishop's Equation 4.45).
 *
 * This code is useless out of a testing scenario.
 */
static void evalTotalScatter(const std::vector<blitz::Array<double, 2> >& data,
    blitz::Array<double,1>& m, blitz::Array<double,2>& St)
{
  const int n_features = data[0].extent(1);
  blitz::Array<double,1> buffer(n_features);

  blitz::firstIndex i;
  blitz::secondIndex j;

  // within class scatter Sw
  St = 0;
  blitz::Range a = blitz::Range::all();
  for (size_t k=0; k<data.size(); ++k) { //class loop
    for (int example=0; example<data[k].extent(0); ++example) {
      buffer = data[k](example,a) - m;
      St += buffer(i) * buffer(j); //outer product
    }
  }
}

static void evalScatters(const std::vector<blitz::Array<double, 2> >& data,
  blitz::Array<double,1>& m,
  blitz::Array<double,2>& Sw, blitz::Array<double,2>& Sb)
{
  // checks for data shape should have been done before...
  const int n_features = data[0].extent(1);

  m = 0; //overall mean
  blitz::Array<double,2> m_k(n_features, data.size());
  m_k = 0; //class means
  blitz::Array<double,1> N(data.size());
  N = 0; //class counts
  blitz::Array<double,1> buffer(n_features); //tmp buffer for speed-up

  evalMeans(data, m, m_k, N);

  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Range a = blitz::Range::all();

  // between class scatter Sb
  Sb = 0;
  for (size_t k=0; k<data.size(); ++k) { //class loop
    buffer = m - m_k(a,k);
    Sb += N(k) * buffer(i) * buffer(j); //Bishop's Eq. 4.46
  }
  Sb /= data.size(); //limit numerical precision problems (not tested)

  // within class scatter Sw
  Sw = 0;
  for (size_t k=0; k<data.size(); ++k) { //class loop
    for (int example=0; example<data[k].extent(0); ++example) {
      buffer = data[k](example,a) - m_k(a,k);
      Sw += buffer(i) * buffer(j); //outer product
    }
  }
  Sw /= (sum(N) - 1); //use cov. matrix to limit precision problems (untested)

  //Code for testing purposes only: prints St - (Sw + Sb) which should be
  //approximately zero:
  //
  //blitz::Array<double,2> St(Sw.shape());
  //evalTotalScatter(data, m, St);
  //St -= (Sw + Sb);
  (void)evalTotalScatter; //< silences gcc, does nothing.
}


void bob::trainer::WCCNTrainer::train(bob::machine::LinearMachine& machine,
    const std::vector<blitz::Array<double, 2> >& data)
{
  const size_t n_classes = data.size();
  // if #classes < 2, then throw
  if (n_classes < 2) throw bob::trainer::WrongNumberOfClasses(data.size());

  // checks for data type and shape once
  const int n_features = data[0].extent(1);

  for (size_t cl=0; cl<n_classes; ++cl) {
    if (data[cl].extent(1) != n_features) {
      throw bob::trainer::WrongNumberOfFeatures(data[cl].extent(1),
          n_features, cl);
    }
  }

  // machine dimensions
  const size_t n_inputs = machine.inputSize();
  const size_t n_outputs = machine.outputSize();

  // Checks that the dimensions are matching
  if ((int)n_inputs != n_features)
    throw bob::machine::NInputsMismatch(n_inputs, n_features);
  if ((int)n_outputs != n_features)
    throw bob::machine::NOutputsMismatch(n_outputs, n_features);

  // 1. Computes the mean vector and the Scatter matrix Sw and Sb
  blitz::Array<double,1> preMean(n_features);
  blitz::Array<double,2> Sw(n_features, n_features);
  blitz::Array<double,2> Sb(n_features, n_features);
  evalScatters(data, preMean, Sw, Sb);

  // 2. Computes the inverse of (1/N * Sw), Sw is the within-class covariance matrix
  Sw /= n_classes;
  blitz::Array<double,2> icov(n_features,n_features);
  bob::math::inv(Sw, icov);

  // 3. Computes the Cholesky decomposition of the inverse covariance matrix 
  blitz::Array<double,2> B(n_features,n_features);
  bob::math::chol(icov, B);

  // 4. Updates the linear machine
  machine.setInputSubtraction(0); // we do not substract the mean
  machine.setInputDivision(1.);
  machine.setWeights(B);
  machine.setBiases(0);
  machine.setActivation(boost::make_shared<bob::machine::IdentityActivation>());
}
