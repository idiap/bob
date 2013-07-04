/**
 * @file trainer/cxx/FisherLDATrainer.cc
 * @date Sat Jun 4 21:38:59 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
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

#include <boost/format.hpp>

#include <bob/core/blitz_compat.h>
#include <bob/math/pinv.h>
#include <bob/math/eig.h>
#include <bob/math/linear.h>
#include <bob/trainer/Exception.h>
#include <bob/trainer/FisherLDATrainer.h>

bob::trainer::FisherLDATrainer::FisherLDATrainer
  (bool use_pinv, bool strip_to_rank)
: m_use_pinv(use_pinv),
  m_strip_to_rank(strip_to_rank)
{
}

bob::trainer::FisherLDATrainer::FisherLDATrainer
  (const bob::trainer::FisherLDATrainer& other)
: m_use_pinv(other.m_use_pinv),
  m_strip_to_rank(other.m_strip_to_rank)
{
}

bob::trainer::FisherLDATrainer::~FisherLDATrainer()
{
}

bob::trainer::FisherLDATrainer& bob::trainer::FisherLDATrainer::operator=
  (const bob::trainer::FisherLDATrainer& other)
{
  m_use_pinv = other.m_use_pinv;
  m_strip_to_rank = other.m_strip_to_rank;
  return *this;
}

bool bob::trainer::FisherLDATrainer::operator==
  (const bob::trainer::FisherLDATrainer& other) const
{
  return m_use_pinv == other.m_use_pinv && \
                     m_strip_to_rank == other.m_strip_to_rank;
}

bool bob::trainer::FisherLDATrainer::operator!=
  (const bob::trainer::FisherLDATrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::FisherLDATrainer::is_similar_to
  (const bob::trainer::FisherLDATrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return this->operator==(other);
}

/**
 * Evaluates, in a single loop, the overall (or grand) mean 'm', the individual
 * class means 'm_k' and computes the total number of elements in each class
 * 'N'.
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
  int n_features = data[0].extent(1);
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
static void evalScatters(const std::vector<blitz::Array<double, 2> >& data,
  blitz::Array<double,1>& m,
  blitz::Array<double,2>& Sw, blitz::Array<double,2>& Sb)
{
  // checks for data shape should have been done before...
  int n_features = data[0].extent(1);

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

  // within class scatter Sw
  Sw = 0;
  for (size_t k=0; k<data.size(); ++k) { //class loop
    for (int example=0; example<data[k].extent(0); ++example) {
      buffer = data[k](example,a) - m_k(a,k);
      Sw += buffer(i) * buffer(j); //outer product
    }
  }

  //Code for testing purposes only: prints St - (Sw + Sb) which should be
  //approximately zero:
  //
  //blitz::Array<double,2> St(Sw.shape());
  //evalTotalScatter(data, m, St);
  //St -= (Sw + Sb);
  (void)evalTotalScatter; //< silences gcc, does nothing.
}

/**
 * Returns the indexes for sorting a given blitz::Array<double,1>
 */
struct compare_1d_blitz {
  const blitz::Array<double,1>& v_;
  compare_1d_blitz(const blitz::Array<double,1>& v): v_(v) { }
  bool operator() (size_t i, size_t j) { return v_(i) < v_(j); }
};

static std::vector<size_t> sort_indexes(const blitz::Array<double,1>& v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), compare_1d_blitz(v));

  return idx;
}

void bob::trainer::FisherLDATrainer::train
(bob::machine::LinearMachine& machine, blitz::Array<double,1>& eigen_values,
  const std::vector<blitz::Array<double, 2> >& data) const
{
  // if #classes < 2, then throw
  if (data.size() < 2) {
    boost::format m("The number of arrays in the input data == %d whereas for LDA you should provide at least 2");
    m % data.size();
    throw std::runtime_error(m.str());
  }

  // checks for arrayset data type and shape once
  int n_features = data[0].extent(1);

  for (size_t cl=0; cl<data.size(); ++cl) {
    if (data[cl].extent(1) != n_features) {
      boost::format m("The number of features/columns (%d) in array at position %d of your input differs from that of array at position 0 (%d)");
      m % data[cl].extent(1) % n_features;
      throw std::runtime_error(m.str());
    }
  }

  int osize = output_size(data);

  // Checks that the dimensions are matching
  if (machine.inputSize() != (size_t)data[0].extent(1)) {
    boost::format m("Number of features at input data set (%d columns) does not match machine input size (%d)");
    m % data[0].extent(1) % machine.inputSize();
    throw std::runtime_error(m.str());
  }
  if (machine.outputSize() != (size_t)osize) {
    boost::format m("Number of outputs of the given machine (%d) does not match the expected number of outputs calculated by this trainer = %d");
    m % machine.outputSize() % osize;
    throw std::runtime_error(m.str());
  }
  if (eigen_values.extent(0) != osize) {
    boost::format m("Number of eigenvalues on the given 1D array (%d) does not match the expected number of outputs calculated by this trainer = %d");
    m % eigen_values.extent(0) % osize;
    throw std::runtime_error(m.str());
  }

  blitz::Array<double,1> preMean(n_features);
  blitz::Array<double,2> Sw(n_features, n_features);
  blitz::Array<double,2> Sb(n_features, n_features);
  evalScatters(data, preMean, Sw, Sb);

  // computes the generalized eigenvalue decomposition
  // so to find the eigen vectors/values of Sw^(-1) * Sb
  blitz::Array<double,2> V(Sw.shape());
  blitz::Array<double,1> eigen_values_(n_features);

  if (m_use_pinv) {

    //note: misuse V and Sw as temporary place holders for data
    bob::math::pinv_(Sw, V); //V now contains Sw^-1
    bob::math::prod_(V, Sb, Sw); //Sw now contains Sw^-1*Sb
    blitz::Array<std::complex<double>,1> Dtemp(eigen_values_.shape());
    blitz::Array<std::complex<double>,2> Vtemp(V.shape());
    bob::math::eig_(Sw, Vtemp, Dtemp); //V now contains eigen-vectors

    //sorting: we know this problem on has real eigen-values
    blitz::Range a = blitz::Range::all();
    blitz::Array<double,1> Dunordered(blitz::real(Dtemp));
    std::vector<size_t> order = sort_indexes(Dunordered);
    for (int i=0; i<n_features; ++i) {
      eigen_values_(i) = Dunordered(order[i]);
      V(a,i) = blitz::real(Vtemp(a,order[i]));
    }
  }
  else {
    bob::math::eigSym_(Sb, Sw, V, eigen_values_);
  }

  // Convert ascending order to descending order
  eigen_values_.reverseSelf(0);
  V.reverseSelf(1);

  // limit the dimensions of the resulting projection matrix and eigen values
  eigen_values = eigen_values_(blitz::Range(0,osize-1));
  V.resizeAndPreserve(V.extent(0), osize);

  // normalizes the eigen vectors so they have unit length
  blitz::Range a = blitz::Range::all();
  for (int column=0; column<V.extent(1); ++column) {
    math::normalizeSelf(V(a,column));
  }

  // updates the machine
  machine.setWeights(V);
  machine.setInputSubtraction(preMean);

  // also set input_div and biases to neutral values...
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
}

void bob::trainer::FisherLDATrainer::train(bob::machine::LinearMachine& machine,
    const std::vector<blitz::Array<double,2> >& data) const {
  blitz::Array<double,1> throw_away(output_size(data));
  train(machine, throw_away, data);
}

size_t bob::trainer::FisherLDATrainer::output_size(const std::vector<blitz::Array<double,2> >& data) const {
  return m_strip_to_rank? (data.size()-1) : data[0].extent(1);
}
