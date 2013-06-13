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

#include <bob/core/blitz_compat.h>
#include <bob/math/eig.h>
#include <bob/math/linear.h>
#include <bob/machine/Exception.h>
#include <bob/machine/EigenMachineException.h>
#include <bob/trainer/Exception.h>
#include <bob/trainer/FisherLDATrainer.h>

bob::trainer::FisherLDATrainer::FisherLDATrainer(int number_of_kept_lda_dimensions)
: m_lda_dimensions(number_of_kept_lda_dimensions)
{
}

bob::trainer::FisherLDATrainer::FisherLDATrainer(const bob::trainer::FisherLDATrainer& other)
: m_lda_dimensions(other.m_lda_dimensions)
{
}

bob::trainer::FisherLDATrainer::~FisherLDATrainer()
{
}

bob::trainer::FisherLDATrainer& bob::trainer::FisherLDATrainer::operator=
  (const bob::trainer::FisherLDATrainer& other)
{
  m_lda_dimensions = other.m_lda_dimensions;
  return *this;
}

bool bob::trainer::FisherLDATrainer::operator==
  (const bob::trainer::FisherLDATrainer& other) const
{
  return m_lda_dimensions == other.m_lda_dimensions;
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
  return m_lda_dimensions == other.m_lda_dimensions;
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

void bob::trainer::FisherLDATrainer::train(bob::machine::LinearMachine& machine,
  blitz::Array<double,1>& eigen_values,
  const std::vector<blitz::Array<double, 2> >& data)
{
  // if #classes < 2, then throw
  if (data.size() < 2) throw bob::trainer::WrongNumberOfClasses(data.size());

  // checks for arrayset data type and shape once
  int n_features = data[0].extent(1);

  for (size_t cl=0; cl<data.size(); ++cl) {
    if (data[cl].extent(1) != n_features) {
      throw bob::trainer::WrongNumberOfFeatures(data[cl].extent(1),
          n_features, cl);
    }
  }

  // Checks that the dimensions are matching
  const int n_inputs = (int)machine.inputSize();
  const int n_outputs = (int)machine.outputSize();
  const int n_eigenvalues = eigen_values.extent(0);
  const int output_dimension = lda_dimensions(data);

  // Checks that the dimensions are matching
  if (n_inputs != n_features)
    throw bob::machine::NInputsMismatch(n_inputs, n_features);
  if (n_outputs != output_dimension)
    throw bob::machine::NOutputsMismatch(n_outputs, output_dimension);
  if (n_eigenvalues != output_dimension)
    throw bob::machine::NOutputsMismatch(n_eigenvalues, output_dimension);

  blitz::Array<double,1> preMean(n_features);
  blitz::Array<double,2> Sw(n_features, n_features);
  blitz::Array<double,2> Sb(n_features, n_features);
  evalScatters(data, preMean, Sw, Sb);

  // computes the generalized eigenvalue decomposition
  // so to find the eigen vectors/values of Sw^(-1) * Sb
  blitz::Array<double,2> V(n_features, n_features);
  blitz::Array<double,1> eigen_values_(n_features);
  // eigSym returned the eigen_values in chronological order
  // reverts the vector and matrix before and after calling eig
  eigen_values_.reverseSelf(0);
  V.reverseSelf(1);
  eigen_values_ = 0;
  bob::math::eigSym(Sb, Sw, V, eigen_values_);
  // Convert ascending order to descending order
  eigen_values_.reverseSelf(0);
  V.reverseSelf(1);

  // limit the dimensions of the resulting projection matrix and eigen values
  eigen_values = eigen_values_(blitz::Range(0,output_dimension-1));
  V.resizeAndPreserve(V.extent(0), output_dimension);

  // updates the machine
  blitz::Range a = blitz::Range::all();
  // normalizes the eigen vectors so they have unit length
  for (int column=0; column<V.extent(1); ++column) {
    math::normalizeSelf(V(a,column));
  }
  machine.setWeights(V);
  machine.setInputSubtraction(preMean);

  // also set input_div and biases to neutral values...
  machine.setInputDivision(1.0);
  machine.setBiases(0.0);
}

void bob::trainer::FisherLDATrainer::train(bob::machine::LinearMachine& machine,
    const std::vector<blitz::Array<double,2> >& data)
{
  blitz::Array<double,1> throw_away(lda_dimensions(data));
  train(machine, throw_away, data);
}

int bob::trainer::FisherLDATrainer::lda_dimensions(const std::vector<blitz::Array<double,2> >& data) const{
  int feature_length = data[0].extent(1);
  if (m_lda_dimensions > feature_length)
    throw bob::machine::EigenMachineNOutputsTooLarge(m_lda_dimensions, feature_length);

  if (m_lda_dimensions > 0)
    return m_lda_dimensions;
  if (m_lda_dimensions == 0)
    return data.size() - 1; // the number of classes - 1
  return feature_length; // the feature length, i.e., all dimensions are kept
}
