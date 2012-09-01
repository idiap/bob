/**
 * @file cxx/trainer/src/LLRTrainer.cc
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

#include "trainer/LLRTrainer.h"
#include "io/Exception.h"
#include "core/array_type.h"
#include "math/linear.h"

#include "core/logging.h"

bob::trainer::LLRTrainer::LLRTrainer(const double prior, 
  const double convergence_threshold, const size_t max_iterations):
    m_prior(prior), m_convergence_threshold(convergence_threshold), 
    m_max_iterations(max_iterations)
{
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

void bob::trainer::LLRTrainer::train(bob::machine::LinearMachine& machine, 
  const bob::io::Arrayset& ar1, const bob::io::Arrayset& ar2) const 
{
  // checks for arraysets data type and shape once
  if(ar1.getElementType() != bob::core::array::t_float64) 
    throw bob::io::TypeError(ar1.getElementType(), bob::core::array::t_float64);
  if(ar1.getNDim() != 1) 
    throw bob::io::DimensionError(ar1.getNDim(), 1);
  if(ar2.getElementType() != bob::core::array::t_float64) 
    throw bob::io::TypeError(ar2.getElementType(), bob::core::array::t_float64);
  if(ar2.getNDim() != 1) 
    throw bob::io::DimensionError(ar2.getNDim(), 1);
  if(ar1.getShape()[0] != ar2.getShape()[0]) 
    throw bob::io::DimensionError(ar1.getShape()[0], ar2.getShape()[0]);

  // data is checked now and conforms, just proceed w/o any further checks.
  size_t n_samples1 = ar1.size();
  size_t n_samples2 = ar2.size();
  size_t n_samples = n_samples1 + n_samples2;
  size_t n_features = ar1.getShape()[0];

  // Defines useful ranges  
  blitz::Range rall = blitz::Range::all();
  blitz::Range rd = blitz::Range(0,n_features-1);
  blitz::Range r1 = blitz::Range(0,n_samples1-1);
  blitz::Range r2 = blitz::Range(n_samples1,n_samples-1);

  // Create a large Blitz array containing the samples
  // x = |ar1 -ar2|, of size (n_features+1,n_samples1+n_samples2)
  //     |1.  -1. |
  blitz::Array<double,2> x(n_features+1, n_samples);
  x(n_features,r1) = 1.;
  x(n_features,r2) = -1.;
  for(size_t i=0; i<n_samples1; ++i)
    x(rd,i) = ar1.get<double,1>(i);
  for(size_t i=0; i<n_samples2; ++i)
    x(rd,i+n_samples1) = -ar2.get<double,1>(i);

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
  double ug; 
  blitz::Array<double,1> ux(n_samples);
  blitz::Array<double,1> a(n_samples);
  double uhu;
  blitz::Array<double,1> tmp_n(n_samples);
  blitz::Array<double,1> tmp_d(n_features+1);

  // Iterates...
  blitz::firstIndex i;
  blitz::secondIndex j;
  for(size_t iter=0; ; ++iter) 
  {
    tmp_n = blitz::sum(w(j)*x(j,i), j);
    s1 = 1. / (1. + blitz::exp(tmp_n + offset));
    tmp_n = s1 * weights;
    bob::math::prod(x, tmp_n, g);

    // Conjugate gradient
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
        // Hestenes-Stiefel
        double beta = blitz::sum(tmp_d * g) / den;
        u = g - beta * u;
      }
    }

    // Line search along the direction u
    ug = blitz::sum(u*g);
    bob::math::prod(u,x,ux);
    a = weights * s1 * (1.-s1);
    tmp_n = blitz::pow2(ux);
    uhu = blitz::sum(tmp_n*a);
    w = w + (ug/uhu) * u;
    
    // Check if convergence has been reached
    if(blitz::max(blitz::fabs(w-w_old)) <= m_convergence_threshold) 
    {
      bob::core::info << "# LLR Training terminated: convergence" << std::endl;
      break;
    }
    // Terminates if maximum number of iterations has been reached
    if(m_max_iterations > 0 && iter+1 >= m_max_iterations) 
    {
      bob::core::info << "# EM terminated: maximum number of iterations reached." << std::endl;
      break;
    }

    // Backup previous values
    g_old = g;
    w_old = w;
  }

  // Update the LinearMachine
  machine.resize(n_features, 1);
  machine.setInputSubtraction(0.);
  machine.setInputDivision(1.);
  blitz::Array<double,2> w_(n_features, 1);
  w_(rall,0) = w(rd);
  machine.setWeights(w_);
  machine.setBiases(w(n_features));
}

