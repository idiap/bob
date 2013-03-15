/**
 * @file math/cxx/interiorpointLP.cc
 * @date Thu Mar 31 14:32:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines interior point methods which allow to solve a
 *        linear program (LP).
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

#include <bob/math/LPInteriorPoint.h>
#include <bob/math/linear.h>
#include <bob/math/linsolve.h>
#include <bob/math/Exception.h>
#include <bob/core/array_copy.h>
#include <bob/core/assert.h>
#include <bob/core/check.h>
#include <limits>


bob::math::LPInteriorPoint::LPInteriorPoint(const size_t M, const size_t N, 
    const double epsilon):
  m_M(M), m_N(N), m_epsilon(epsilon), m_lambda(M), m_mu(N)
{
  m_lambda = 0.;
  m_mu = 0.;
  resetCache();
}

bob::math::LPInteriorPoint::LPInteriorPoint(
  const bob::math::LPInteriorPoint &other):
  m_M(other.m_M), m_N(other.m_N), m_epsilon(other.m_epsilon), 
  m_lambda(bob::core::array::ccopy(other.m_lambda)),
  m_mu(bob::core::array::ccopy(other.m_mu))
{
  resetCache();
}

void bob::math::LPInteriorPoint::resetCache()
{
  m_cache_gradient.resize(m_M);
  m_cache_M.resize(m_M);
  m_cache_N.resize(m_N);

  m_cache_x.resize(m_N);
  m_cache_lambda.resize(m_M);
  m_cache_mu.resize(m_N);

  m_cache_A_large.resize(m_M+2*m_N, m_M+2*m_N);
  m_cache_b_large.resize(m_M+2*m_N);
  m_cache_x_large.resize(m_M+2*m_N);
}

bob::math::LPInteriorPoint& bob::math::LPInteriorPoint::operator=(
  const bob::math::LPInteriorPoint& other)
{
  if (this != &other)
  {
    m_M = other.m_M;
    m_N = other.m_N;
    m_epsilon = other.m_epsilon;
    m_lambda = bob::core::array::ccopy(other.m_lambda);
    m_mu = bob::core::array::ccopy(other.m_mu);
    resetCache();
  }
  return *this;
}

bool bob::math::LPInteriorPoint::operator==(
  const bob::math::LPInteriorPoint& other) const
{
  return (m_M == other.m_M && m_N == other.m_N && 
          m_epsilon == other.m_epsilon &&
          bob::core::array::isEqual(m_lambda, other.m_lambda) &&
          bob::core::array::isEqual(m_mu, other.m_mu));
}

bool bob::math::LPInteriorPoint::operator!=(
  const bob::math::LPInteriorPoint& other) const
{
  return !(this->operator==(other));
}

void bob::math::LPInteriorPoint::reset(const size_t M, const size_t N)
{
  m_M = M;
  m_N = N;
  m_lambda.resize(M);
  m_mu.resize(N);
  resetCache();
}

bool bob::math::LPInteriorPoint::isFeasible(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu) const
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(lambda.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(mu.extent(0), m_N);

  // x >= 0
  if (blitz::any(x < 0.))
    return false;

  // mu >= 0
  if (blitz::any(mu < 0.))
    return false;

  // A*x = b (abs(A*x-b)<=epsilon)
  bob::math::prod(A, x, m_cache_M);
  m_cache_M -= b;
  if (blitz::any(blitz::fabs(m_cache_M) > m_epsilon))
    return false;

  // A'*lambda + mu = c (abs(A'*lambda+mu-c)<=epsilon)
  // ugly fix for old blitz version
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0); 
  bob::math::prod(A_t, lambda, m_cache_N);
  m_cache_N += mu - c;
  return (!blitz::any(blitz::fabs(m_cache_N) > m_epsilon));
}


bool bob::math::LPInteriorPoint::isInV(const blitz::Array<double,1>& x, 
  const blitz::Array<double,1>& mu, const double theta) const
{
  // Check
  bob::core::array::assertSameDimensionLength(x.extent(0), mu.extent(0));

  // Check that the L2 norm is smaller than theta
  // 1) compute nu
  double nu = bob::math::dot(x, mu) / x.extent(0);

  // 2) compute the L2 norm
  double norm = sqrt( blitz::sum( blitz::pow2(x*mu - nu) ) ) / nu;

  // 3) check
  return (norm <= theta);
}

bool bob::math::LPInteriorPoint::isInVS(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double theta) const
{
  return (isFeasible(A, b, c, x, lambda, mu) && isInV(x, mu, theta));
}

double bob::math::LPInteriorPoint::logBarrierLP(
  const blitz::Array<double,2>& A_t, const blitz::Array<double,1>& c) const
{
  bob::math::prod( A_t, m_lambda, m_cache_N);
  if (blitz::any(c - m_cache_N <= 0.))
    return std::numeric_limits<double>::infinity();
  return blitz::sum( -blitz::log(c - m_cache_N));
}


void bob::math::LPInteriorPoint::gradientLogBarrierLP(
  const blitz::Array<double,2>& A, const blitz::Array<double,1>& c)
{
  // ugly fix for old blitz versions
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);
  bob::math::prod( A_t, m_lambda, m_cache_N);
  m_cache_N = c - m_cache_N; // c-transpose(A)*lambda
  const double eps = std::numeric_limits<double>::epsilon();
  m_cache_N = blitz::where( m_cache_N < eps, eps, m_cache_N);
  blitz::firstIndex i1;
  blitz::secondIndex i2;
  m_cache_gradient = -sum( A(i1,i2) / m_cache_N(i2), i2);
}


void bob::math::LPInteriorPoint::initializeDualLambdaMu(
  const blitz::Array<double,2>& A, const blitz::Array<double,1>& c)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);

  // Ugly fix for old blitz version
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);

  // Loop until we find a tuple (lambda,mu) which satisfies the constraint:
  //   transpose(A)*lambda + mu = c, with mu>=0
  while (true)
  {
    double alpha = std::numeric_limits<double>::epsilon();
    // Compute the value of the logarithm barrier function
    double f_old = logBarrierLP(A_t, c);

    // Find an alpha/lambda which decreases the barrier function
    while (alpha != std::numeric_limits<double>::infinity())
    {
      // Compute the gradient vector/direction d
      gradientLogBarrierLP( A, c);
      
      // Move lambda towards the d direction
      m_lambda += alpha * m_cache_gradient;
      
      // Compute the new value of the barrier
      double f_new = logBarrierLP( A_t, c);
    
      // Break if the value of the barrier decreases
      if (f_new < f_old)
        break;

      // Increase alpha
      alpha *= 2.;
    }

    // Update mu (= c - transpose(A)*lambda )
    bob::math::prod( A_t, m_lambda, m_cache_N);
    m_mu = c - m_cache_N; // c-transpose(A)*lambda
    
    // break if all the mu_i are positive
    if (blitz::all(m_mu >= 0.))
      break;
  }
}


void bob::math::LPInteriorPoint::centeringV(const blitz::Array<double,2>& A, 
  const double theta, blitz::Array<double,1>& x)
{
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  initializeLargeSystem(A);

  int k=0;  
  while (true)
  {
    // 1) Stopping criterion
    if (isInV(x, m_mu, theta) )
      break;

    // 2) Update the big system and solve it
    updateLargeSystem( x, 1., m);
    bob::math::linsolve( m_cache_A_large, m_cache_x_large, m_cache_b_large);

    // 4) Find alpha and update x, lamda and mu
    double alpha=1.;
    do {
      m_cache_lambda = m_lambda + alpha * m_cache_x_large(r_m+n);
      m_cache_x = x + alpha * m_cache_x_large(r_n);
      m_cache_mu = m_mu + alpha * m_cache_x_large(r_n+m+n);
      alpha /= 2.;
      if (alpha < 2*std::numeric_limits<double>::epsilon())
        throw bob::math::Exception();
    } while ( !(blitz::all(m_cache_x >= 0.) && blitz::all(m_cache_mu >= 0.)) );
    // Move content back
    m_lambda = m_cache_lambda;
    x = m_cache_x;
    m_mu = m_cache_mu;

    // 4) k = k + 1.
    ++k;
  }
}


void bob::math::LPInteriorPoint::initializeLargeSystem(
  const blitz::Array<double,2>& A) const
{
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);

  // 'Compute' transpose(A)
  // ugly fix for old blitz version
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);

  // Initialize
  m_cache_A_large = 0.;

  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // block 0x0: A
  m_cache_A_large(r_m, r_n) = A;
  // block 1x1: transpose(A)
  m_cache_A_large(r_n+m, r_m+n) = A_t;
  // block 1x2: I
  for (int i=0; i<n; ++i)
    m_cache_A_large(m+i, m+n+i) = 1.;

  m_cache_b_large = 0.;
}

void bob::math::LPInteriorPoint::updateLargeSystem(const blitz::Array<double,1>& x, 
  const double sigma, const int m) const
{
  // Get dimensions from the A matrix
  const int n = x.extent(0);
 
  // Compute nu*sigma
  double nu_sigma = sigma * bob::math::dot(x, m_mu) / n;

  // block 2x0: S_k
  for (int i=0; i<n; ++i)
    m_cache_A_large(m+n+i, i) = m_mu(i);
  // block 2x2: X_k
  for (int i=0; i<n; ++i)
    m_cache_A_large(m+n+i, m+n+i) = x(i);
  // block 2: -X S e + nu sigma e
  blitz::Range r_n(0,n-1);
  m_cache_b_large(r_n+m+n) = -x*m_mu + nu_sigma;
}



bob::math::LPInteriorPointShortstep::LPInteriorPointShortstep(
    const size_t M, const size_t N,
    const double theta, const double epsilon):
  bob::math::LPInteriorPoint(M, N, epsilon), 
  m_theta(theta) 
{
}

bob::math::LPInteriorPointShortstep::LPInteriorPointShortstep(
    const bob::math::LPInteriorPointShortstep& other):
  bob::math::LPInteriorPoint(other),
  m_theta(other.m_theta)
{
}

bob::math::LPInteriorPointShortstep& 
bob::math::LPInteriorPointShortstep::operator=(
    const bob::math::LPInteriorPointShortstep& other)
{
  if(this != &other)
  {
    bob::math::LPInteriorPoint::operator=(other);
    m_theta = other.m_theta;
  }
  return *this;
}

bool bob::math::LPInteriorPointShortstep::operator==(
  const bob::math::LPInteriorPointShortstep& other) const
{
  return (bob::math::LPInteriorPoint::operator==(other) &&
          m_theta == other.m_theta);
}

bool bob::math::LPInteriorPointShortstep::operator!=(
  const bob::math::LPInteriorPointShortstep& other) const
{
  return !(this->operator==(other));
}

void bob::math::LPInteriorPointShortstep::solve(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda, 
  const blitz::Array<double,1>& mu)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(lambda.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(mu.extent(0), m_N);

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize variables
  const double sigma = 1. - m_theta / sqrt(n);
  double nu;

  // Declare and initialize arrays for the large linear system
  initializeLargeSystem(A);
  m_lambda = lambda;
  m_mu = mu;

  int k=0;  
  while(true)
  {
    // 1) nu = 1/n <x,mu>
    nu = bob::math::dot(x, m_mu) / n;
    // Stopping criterion
    if( nu < m_epsilon )
      break;

    // 2) Update the big system and solve it
    updateLargeSystem( x, sigma, m);
    bob::math::linsolve( m_cache_A_large, m_cache_x_large, m_cache_b_large);

    // 3) Update x, lamda and mu
    m_lambda += m_cache_x_large( r_m+n);
    x += m_cache_x_large( r_n);
    m_mu += m_cache_x_large( r_n+m+n);

    // 4) k = k + 1.
    ++k;
  }
}

void bob::math::LPInteriorPointShortstep::solve(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);

  // Initialize dual variables
  m_lambda = 0.;
  m_mu = 0.;
  initializeDualLambdaMu( A, c);

  // Find an initial solution
  centeringV( A, m_theta, x);

  // Launch the short step algorithm
  solve( A, b, c, x, m_lambda, m_mu);
}



bob::math::LPInteriorPointPredictorCorrector::LPInteriorPointPredictorCorrector(
    const size_t M, const size_t N,
    const double theta_pred, const double theta_corr, const double epsilon):
  bob::math::LPInteriorPoint(M, N, epsilon), 
  m_theta_pred(theta_pred), m_theta_corr(theta_corr)
{
}

bob::math::LPInteriorPointPredictorCorrector::LPInteriorPointPredictorCorrector(
    const bob::math::LPInteriorPointPredictorCorrector& other):
  bob::math::LPInteriorPoint(other),
  m_theta_pred(other.m_theta_pred),
  m_theta_corr(other.m_theta_corr)
{
}

bob::math::LPInteriorPointPredictorCorrector& 
bob::math::LPInteriorPointPredictorCorrector::operator=(
    const bob::math::LPInteriorPointPredictorCorrector& other)
{
  if(this != &other)
  {
    bob::math::LPInteriorPoint::operator=(other);
    m_theta_pred = other.m_theta_pred;
    m_theta_corr = other.m_theta_corr;
  }
  return *this;
}

bool bob::math::LPInteriorPointPredictorCorrector::operator==(
  const bob::math::LPInteriorPointPredictorCorrector& other) const
{
  return (bob::math::LPInteriorPoint::operator==(other) &&
          m_theta_pred == other.m_theta_pred &&
          m_theta_corr == other.m_theta_corr);
}

bool bob::math::LPInteriorPointPredictorCorrector::operator!=(
  const bob::math::LPInteriorPointPredictorCorrector& other) const
{
  return !(this->operator==(other));
}

void bob::math::LPInteriorPointPredictorCorrector::solve(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda, 
  const blitz::Array<double,1>& mu)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(lambda.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(mu.extent(0), m_N);

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare variable
  double nu;

  // Declare and initialize arrays for the large linear system
  initializeLargeSystem(A);
  m_lambda = lambda;
  m_mu = mu;

  int k=0;  
  while (true)
  {
    /////////////////////////////
    // PREDICTION
    // 1) nu = 1/n <x,mu>
    nu = bob::math::dot(x, m_mu) / n;
    // Stopping criterion
    if (nu < m_epsilon)
      break;

    // 2) Update the big system and solve it
    updateLargeSystem( x, 0., m);
    bob::math::linsolve( m_cache_A_large, m_cache_x_large, m_cache_b_large);

    // 3) alpha=1
    double alpha = 1.;
 
    // 4) Find alpha and update x, lamda and mu
    do {
      m_cache_lambda = m_lambda + alpha * m_cache_x_large(r_m+n);
      m_cache_x = x + alpha * m_cache_x_large(r_n);
      m_cache_mu = m_mu + alpha * m_cache_x_large(r_n+m+n);
      alpha /= 2.;
      if (alpha<2*std::numeric_limits<double>::epsilon())
        throw bob::math::Exception();
    } while (!isInVS(A, b, c, m_cache_x, m_cache_lambda, m_cache_mu, m_theta_pred));
    // Move content back
    m_lambda = m_cache_lambda;
    x = m_cache_x;
    m_mu = m_cache_mu;

    // 5) k = k + 1.
    ++k;


    /////////////////////////////
    // CORRECTION
    // 6) Update nu
    nu = bob::math::dot(x, m_mu) / n;
    // Stopping criterion
    if( nu < m_epsilon )
      break;

    // 7) Update the big system and solve it
    updateLargeSystem( x, 1., m);
    bob::math::linsolve( m_cache_A_large, m_cache_x_large, m_cache_b_large);

    // 8) Update x
    m_lambda += m_cache_x_large(r_m+n);
    x += m_cache_x_large(r_n);
    m_mu += m_cache_x_large(r_n+m+n);

    // 9) k = k + 1
    ++k;
  } 
}

void bob::math::LPInteriorPointPredictorCorrector::solve(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);

  // Initialize dual variables
  m_lambda = 0.;
  m_mu = 0.;
  initializeDualLambdaMu(A, c);

  // Find an initial solution
  centeringV(A, m_theta_corr, x);

  // Launch the long step algorithm
  solve(A, b, c, x, m_lambda, m_mu);
}



bob::math::LPInteriorPointLongstep::LPInteriorPointLongstep(
    const size_t M, const size_t N,
    const double gamma, const double sigma, const double epsilon):
  bob::math::LPInteriorPoint(M, N, epsilon), 
  m_gamma(gamma), m_sigma(sigma)
{
}

bob::math::LPInteriorPointLongstep::LPInteriorPointLongstep(
    const bob::math::LPInteriorPointLongstep& other):
  bob::math::LPInteriorPoint(other),
  m_gamma(other.m_gamma),
  m_sigma(other.m_sigma)
{
}

bob::math::LPInteriorPointLongstep& 
bob::math::LPInteriorPointLongstep::operator=(
    const bob::math::LPInteriorPointLongstep& other)
{
  if(this != &other)
  {
    bob::math::LPInteriorPoint::operator=(other);
    m_gamma = other.m_gamma;
    m_sigma = other.m_sigma;
  }
  return *this;
}

bool bob::math::LPInteriorPointLongstep::operator==(
  const bob::math::LPInteriorPointLongstep& other) const
{
  return (bob::math::LPInteriorPoint::operator==(other) &&
          m_gamma == other.m_gamma && m_sigma == other.m_sigma);
}

bool bob::math::LPInteriorPointLongstep::operator!=(
  const bob::math::LPInteriorPointLongstep& other) const
{
  return !(this->operator==(other));
}

bool bob::math::LPInteriorPointLongstep::isInV(const blitz::Array<double,1>& x,
  const blitz::Array<double,1>& mu, const double gamma) const
{
  bob::core::array::assertSameDimensionLength(x.extent(0), mu.extent(0));

  // Check using Linf norm
  // 1) compute nu
  double nu = bob::math::dot(x, mu) / x.extent(0);

  // 2) check
  return (!blitz::any(x*mu < gamma*nu));
}

void bob::math::LPInteriorPointLongstep::solve(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda, 
  const blitz::Array<double,1>& mu)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(lambda.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(mu.extent(0), m_N);

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize variables
  double nu;

  initializeLargeSystem(A);
  m_lambda = lambda;
  m_mu = mu;

  int k=0;  
  while(true)
  {
    // 1) nu = 1/n <x,mu>
    nu = bob::math::dot(x, m_mu) / n;
    // Stopping criterion
    if (nu < m_epsilon)
      break;

    // 2) Update the big system and solve it
    updateLargeSystem(x, m_sigma, m);
    bob::math::linsolve(m_cache_A_large, m_cache_x_large, m_cache_b_large);

    // 3) alpha=1
    double alpha = 1.;

    // 4) Find alpha and update x, lamda and mu
    do {
      m_cache_lambda = m_lambda + alpha * m_cache_x_large(r_m+n);
      m_cache_x = x + alpha * m_cache_x_large(r_n);
      m_cache_mu = m_mu + alpha * m_cache_x_large(r_n+m+n);
      alpha /= 2.;
      if (alpha < 2*std::numeric_limits<double>::epsilon())
        throw bob::math::Exception();
    } while (!isInVS(A, b, c, m_cache_x, m_cache_lambda, m_cache_mu, m_gamma));
    // Move content back
    m_lambda = m_cache_lambda;
    x = m_cache_x;
    m_mu = m_cache_mu;

    // 5) k = k + 1.
    ++k;
  }
}


void bob::math::LPInteriorPointLongstep::solve(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  blitz::Array<double,1>& x)
{
  // Check
  bob::core::array::assertSameDimensionLength(A.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(A.extent(1), m_N);
  bob::core::array::assertSameDimensionLength(b.extent(0), m_M);
  bob::core::array::assertSameDimensionLength(c.extent(0), m_N);
  bob::core::array::assertSameDimensionLength(x.extent(0), m_N);

  // Initialize dual variables
  m_lambda = 0.;
  m_mu = 0.;
  initializeDualLambdaMu(A, c);

  // Find an initial solution
  centeringV(A, m_gamma, x);

  // Launch the long step algorithm
  solve(A, b, c, x, m_lambda, m_mu);
}

