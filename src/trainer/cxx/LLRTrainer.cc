/**
 * @file trainer/cxx/LLRTrainer.cc
 * @date Sat Sep 1 19:26:00 2012 +0100
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
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

#include "bob/trainer/LLRTrainer.h"
#include "bob/core/array_type.h"
#include "bob/math/linear.h"

#include "bob/core/logging.h"

#include <limits>

bob::trainer::LLRTrainer::LLRTrainer(const double prior, 
  const double convergence_threshold, const size_t max_iterations):
    m_prior(prior), m_convergence_threshold(convergence_threshold), 
    m_max_iterations(max_iterations)
{
  if(prior<=0. || prior>=1.) 
    throw bob::trainer::LLRPriorNotInRange(prior);
}

bob::trainer::LLRTrainer::LLRTrainer(const bob::trainer::LLRTrainer& other):
  m_prior(other.m_prior),
  m_convergence_threshold(other.m_convergence_threshold), 
  m_max_iterations(other.m_max_iterations)
{
}

bob::trainer::LLRTrainer::~LLRTrainer() {}

bob::trainer::LLRTrainer& bob::trainer::LLRTrainer::operator=
(const bob::trainer::LLRTrainer& other) 
{
  if(this != &other)
  {
    m_prior = other.m_prior;
    m_convergence_threshold = other.m_convergence_threshold;
    m_max_iterations = other.m_max_iterations;
  }
  return *this;
}

bool 
bob::trainer::LLRTrainer::operator==(const bob::trainer::LLRTrainer& b) const
{
  return (this->m_prior == b.m_prior &&
          this->m_convergence_threshold == b.m_convergence_threshold &&
          this->m_max_iterations == b.m_max_iterations);
}

bool 
bob::trainer::LLRTrainer::operator!=(const bob::trainer::LLRTrainer& b) const
{
  return !(this->operator==(b));
}

void bob::trainer::LLRTrainer::train(bob::machine::LinearMachine& machine, 
  const blitz::Array<double,2>& ar1, const blitz::Array<double,2>& ar2) const 
{
  // Checks for arraysets data type and shape once
  if(ar1.extent(1) != ar2.extent(1)) 
    throw bob::io::DimensionError(ar1.extent(1), ar2.extent(1));

  // Data is checked now and conforms, just proceed w/o any further checks.
  size_t n_samples1 = ar1.extent(0);
  size_t n_samples2 = ar2.extent(0);
  size_t n_samples = n_samples1 + n_samples2;
  size_t n_features = ar1.extent(1);

  // Defines useful ranges  
  blitz::Range rall = blitz::Range::all();
  blitz::Range rd = blitz::Range(0,n_features-1);
  blitz::Range r1 = blitz::Range(0,n_samples1-1);
  blitz::Range r2 = blitz::Range(n_samples1,n_samples-1);

  // Creates a large blitz::Array containing the samples
  // x = |ar1 -ar2|, of size (n_features+1,n_samples1+n_samples2)
  //     |1.  -1. |
  blitz::Array<double,2> x(n_features+1, n_samples);
  x(n_features,r1) = 1.;
  x(n_features,r2) = -1.;
  for(size_t i=0; i<n_samples1; ++i)
    x(rd,i) = ar1(i,rall);
  for(size_t i=0; i<n_samples2; ++i)
    x(rd,i+n_samples1) = -ar2(i,rall);

  // Ratio between the two classes and weights vector
  double prop = (double)n_samples1 / (double)n_samples;
  blitz::Array<double,1> weights(n_samples);
  weights(r1) = m_prior / prop;
  weights(r2) = (1.-m_prior) / (1.-prop);
  
  // Initializes offset vector
  blitz::Array<double,1> offset(n_samples);
  const double logit = log(m_prior/(1.-m_prior));
  offset(r1) = logit;
  offset(r2) = -logit;

  // Initializes gradient and w vectors
  blitz::Array<double,1> g_old(n_features+1);
  blitz::Array<double,1> w_old(n_features+1);
  blitz::Array<double,1> g(n_features+1);
  blitz::Array<double,1> w(n_features+1);
  g_old = 0.;
  w_old = 0.;
  g = 0.;
  w = 0.;

  // Initialize working arrays
  blitz::Array<double,1> s1(n_samples);
  blitz::Array<double,1> u(n_features+1);
  blitz::Array<double,1> tmp_n(n_samples);
  blitz::Array<double,1> tmp_d(n_features+1);

  // Iterates...
  blitz::firstIndex i;
  blitz::secondIndex j;
  static const double ten_epsilon = 10*std::numeric_limits<double>::epsilon();
  for(size_t iter=0; ; ++iter) 
  {
    // 1. Computes the non-weighted version of the likelihood
    // s1 = sum_{i=1}^{n}(1./(1.+exp(-y_i (w^T x_i + logit))
    //   where - the x blitz::Array contains -y_i x_i values
    //         - the offset blitz::Array contains -y_i logit values
    s1 = 1. / (1. + blitz::exp(blitz::sum(w(j)*x(j,i), j) + offset));
    // 2. Likelihood weighted by the prior/proportion
    tmp_n = s1 * weights;
    // 3. Gradient g of this weighted likelihood wrt. the weight vector w
    bob::math::prod(x, tmp_n, g);

    // 4. Conjugate gradient step
    if(iter == 0) 
      u = g;
    else
    {
      tmp_d = (g-g_old);
      double den = blitz::sum(u * tmp_d);
      if(den == 0) 
        u = 0.;
      else
      {
        // Hestenes-Stiefel formula: Heuristic to set the scale factor beta
        //   (chosen as it works well in practice)
        // beta = g^t(g-g_old) / (u_old^T (g - g_old))
        double beta = blitz::sum(tmp_d * g) / den;
        u = g - beta * u;
      }
    }

    // 5. Line search along the direction u
    // a. Compute ux
    bob::math::prod(u,x,tmp_n);
    // b. Compute u^T H u 
    //      = sum_{i} weights(i) sigmoid(w^T x_i) [1-sigmoid(w^T x_i)] (u^T x_i)
    double uhu = blitz::sum(blitz::pow2(tmp_n) * weights * s1 * (1.-s1));
    // Terminates if uhu is close to zero
    if(fabs(uhu) < ten_epsilon)
    {
      bob::core::info << ten_epsilon << std::endl;
      bob::core::info << "# LLR Training terminated: convergence after " << iter << " iterations (u^T H u == 0)." << std::endl;
      break;
    }
    // c. Compute w = w_old - (g^T u)/(u^T H u) u
    w = w + blitz::sum(u*g) / uhu * u;
    
    // Terminates if convergence has been reached
    if(blitz::max(blitz::fabs(w-w_old)) <= m_convergence_threshold) 
    {
      bob::core::info << "# LLR Training terminated: convergence after " << iter << " iterations." << std::endl;
      break;
    }
    // Terminates if maximum number of iterations has been reached
    if(m_max_iterations > 0 && iter+1 >= m_max_iterations) 
    {
      bob::core::info << "# EM terminated: maximum number of iterations (" << m_max_iterations << ") reached." << std::endl;
      break;
    }

    // Backup previous values
    g_old = g;
    w_old = w;
  }

  // Updates the LinearMachine
  machine.resize(n_features, 1);
  machine.setInputSubtraction(0.); // No subtraction
  machine.setInputDivision(1.); // No division
  blitz::Array<double,2>& w_ = machine.updateWeights();
  w_(rall,0) = w(rd); // Weights: first D values
  machine.setBiases(w(n_features)); // Bias: D+1 value
}

