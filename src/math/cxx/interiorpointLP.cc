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

#include "bob/math/linear.h"
#include "bob/math/interiorpointLP.h"
#include "bob/math/linsolve.h"
#include "bob/math/Exception.h"


namespace math = bob::math;
namespace mathdetail = bob::math::detail;
namespace ca = bob::core::array;

bool mathdetail::isPositive(const blitz::Array<double,1>& x)
{
  for( int i=x.lbound(0); i<=x.ubound(0); ++i)
    if( x(i) < 0. )
      return false;
  return true;
}

bool mathdetail::isFeasible(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon)
{
  // x >= 0
  if( !mathdetail::isPositive(x) )
    return false;

  // mu >= 0
  if( !mathdetail::isPositive(mu) )
    return false;

  // A*x = b (abs(A*x-b)<=epsilon)
  blitz::Array<double,1> Ax(A.extent(0));
  math::prod(A,x,Ax);
  Ax -= b;
  for( int i=Ax.lbound(0); i<=Ax.ubound(0); ++i)
    if( fabs(Ax(i)) > epsilon )
      return false;

  // A'*lambda + mu = c (abs(A'*lambda+mu-c)<=epsilon)
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0); 
  blitz::Array<double,1> A_t_lambda(A_t.extent(0));
  math::prod(A_t,lambda,A_t_lambda);
  A_t_lambda += mu - c;
  for( int i=A_t_lambda.lbound(0); i<=A_t_lambda.ubound(0); ++i)
    if( fabs(A_t_lambda(i)) > epsilon )
      return false;

  return true;
}

bool mathdetail::isInV2(const blitz::Array<double,1>& x,
  const blitz::Array<double,1>& mu, const double theta)
{
  // Check that the L2 norm is smaller than theta
  // 1) compute nu
  double nu = math::dot(x, mu) / x.extent(0);

  // 2) compute the L2 norm
  blitz::firstIndex i;
  blitz::Array<double,1> xmu_nu( blitz::Range(x.lbound(0),x.ubound(0)) );
  xmu_nu = x(i)*mu(i) - nu;
  double norm = sqrt( sum( xmu_nu*xmu_nu ) ) / nu;

  // 3) check
  return norm<=theta;
}

bool mathdetail::isInVinf(const blitz::Array<double,1>& x,
  const blitz::Array<double,1>& mu, const double gamma)
{
  // Check using Linf norm
  // 1) compute nu
  double nu = math::dot(x, mu) / x.extent(0);

  // 2) check
  blitz::Array<double,1> xmu( blitz::Range(x.lbound(0),x.ubound(0)) );
  xmu = x*mu; 
  for( int i=xmu.lbound(0); i<=xmu.ubound(0); ++i)
    if(  xmu(i) < gamma*nu )
      return false;

  return true;
}

bool mathdetail::isInV2S(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon, const double theta)
{
  // Check if the tested 'solution' belongs to the set of feasible solutions
  if( !mathdetail::isFeasible(A,b,c,x,lambda,mu,epsilon) )
    return false;

  // Check if the point belongs to V2(theta)
  return mathdetail::isInV2(x, mu, theta);
}

bool mathdetail::isInVinfS(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon, const double gamma)
{
  // Check if the tested 'solution' belongs to the set of feasible solutions
  if( !mathdetail::isFeasible(A,b,c,x,lambda,mu,epsilon) )
    return false;

  // Check if the point belongs to V-inf(gamma)
  return mathdetail::isInVinf(x, mu, gamma);
}

double mathdetail::logBarrierLP(const blitz::Array<double,2>& A_t,
  const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda)
{
  blitz::Array<double,1> A_t_lambda(A_t.extent(0));
  math::prod( A_t, lambda, A_t_lambda);
  double res = 0.;
  for( int i=c.lbound(0); i<=c.ubound(0); ++i) {
    if( c(i) - A_t_lambda(i) <= 0 )
      return std::numeric_limits<double>::infinity();
    else
      res += -log( c(i) - A_t_lambda(i) );
  }
      
  return res;
}

void mathdetail::gradientLogBarrierLP(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda, 
  blitz::Array<double,1>& working_array, blitz::Array<double,1>& gradient)
{
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);
  math::prod( A_t, lambda, working_array);
  blitz::firstIndex ind;
  working_array = c(ind) - working_array(ind); // c-transpose(A)*lambda
  for( int i=working_array.lbound(0); 
      i<=working_array.ubound(0); ++i)
  {
    if( working_array(i) < std::numeric_limits<double>::epsilon() )
      working_array(i) = std::numeric_limits<double>::epsilon();
  }
  blitz::firstIndex i1;
  blitz::secondIndex i2;
  gradient = -sum( A(i1,i2) / working_array(i2), i2);
}


void mathdetail::initializeDualLambdaMuLP(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda, 
  blitz::Array<double,1>& mu)
{
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);

  // Initialize working array and d (gradient)
  blitz::Range r_c( c.lbound(0), c.ubound(0));
  blitz::Range r_lambda( lambda.lbound(0), lambda.ubound(0));
  blitz::Array<double,1> working_array( r_c), d( r_lambda);

  // Loop until we find a tuple (lambda,mu) which satisfies the constraint:
  //   transpose(A)*lambda + mu = c, with mu>=0
  while( true)
  {
    double alpha = std::numeric_limits<double>::epsilon();
    // Compute the value of the logarithm barrier function
    double f_old = mathdetail::logBarrierLP( A_t, c, lambda);

    // Find an alpha/lambda which decreases the barrier function
    while( alpha!= std::numeric_limits<double>::infinity())
    {
      // Compute the gradient vector/direction d
      mathdetail::gradientLogBarrierLP( A, c, lambda, working_array, d);
      
      // Move lambda towards the d direction
      lambda += alpha * d;
      
      // Compute the new value of the barrier
      double f_new = mathdetail::logBarrierLP( A_t, c, lambda);
    
      // Break if the value of the barrier decreases
      if( f_new < f_old)
        break;

      // Increase alpha
      alpha *= 2.;
    }

    // Update mu (= c - transpose(A)*lambda )
    math::prod( A_t, lambda, working_array);
    mu = c - working_array; // c-transpose(A)*lambda
    
    // break if all the mu_i are positive
    bool mu_positive = true;
    for( int i=mu.lbound(0); i<=mu.ubound(0); ++i)
      if( mu(i) < 0 ) {
        mu_positive = false;
        break;
      }
    if( mu_positive) break;
  }
}


void mathdetail::centeringV2(const blitz::Array<double,2>& A, 
  const double theta, blitz::Array<double,1>& x, 
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu)
{
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize arrays for the large linear system
  blitz::Array<double,2> A_large;
  blitz::Array<double,1> x_large; // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_large;

  mathdetail::initializeLargeSystem( A, A_large, b_large, x_large);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  
  while(true)
  {
    // 1) Stopping criterion
    if( mathdetail::isInV2(x, mu, theta) )
      break;

    // 2) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, 1., m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 4) Find alpha and update x, lamda and mu
    double alpha=1.;
    do {
      lambda_alpha(r_m) = lambda(r_m+lambda.lbound(0)) + alpha * x_large(r_m+n);
      x_alpha(r_n) = x(r_n+x.lbound(0)) + alpha * x_large(r_n);
      mu_alpha(r_n) = mu(r_n+mu.lbound(0)) + alpha * x_large(r_n+m+n);
      alpha /= 2.;
      if( alpha<2*std::numeric_limits<double>::epsilon())
        throw math::Exception();
    } while( !(mathdetail::isPositive(x_alpha) && mathdetail::isPositive(mu_alpha)) );
    // Move content back
    lambda( r_m+lambda.lbound(0)) = lambda_alpha(r_m);
    x( r_n+x.lbound(0)) = x_alpha(r_n);
    mu( r_n+mu.lbound(0)) = mu_alpha(r_n);

    // 4) k = k + 1.
    ++k;
  }
}

void mathdetail::centeringVinf(const blitz::Array<double,2>& A, 
  const double gamma, blitz::Array<double,1>& x,
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu)
{
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize arrays for the large linear system
  blitz::Array<double,2> A_large;
  blitz::Array<double,1> x_large; // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_large;

  mathdetail::initializeLargeSystem( A, A_large, b_large, x_large);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  
  while(true)
  {
    // 1) Stopping criterion
    if( mathdetail::isInVinf(x, mu, gamma) )
      break;

    // 2) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, 1., m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 4) Find alpha and update x, lamda and mu
    double alpha=1.;
    do {
      lambda_alpha(r_m) = lambda(r_m+lambda.lbound(0)) + alpha * x_large(r_m+n);
      x_alpha(r_n) = x(r_n+x.lbound(0)) + alpha * x_large(r_n);
      mu_alpha(r_n) = mu(r_n+mu.lbound(0)) + alpha * x_large(r_n+m+n);
      alpha /= 2.;
      if( alpha<2*std::numeric_limits<double>::epsilon())
        throw math::Exception();
    } while( !(mathdetail::isPositive(x_alpha) && mathdetail::isPositive(mu_alpha)) );
    // Move content back
    lambda( r_m+lambda.lbound(0)) = lambda_alpha(r_m);
    x( r_n+x.lbound(0)) = x_alpha(r_n);
    mu( r_n+mu.lbound(0)) = mu_alpha(r_n);

    // 4) k = k + 1.
    ++k;
  }
}

void mathdetail::initializeLargeSystem(const blitz::Array<double,2>& A,
  blitz::Array<double,2>& A_large, blitz::Array<double,1>& b_large,
  blitz::Array<double,1>& x_large)
{
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);

  // Reindex and resize A_large, b_large and x_large
  bob::core::array::reindexAndResize( A_large, 0, 0, m+2*n, m+2*n);
  bob::core::array::reindexAndResize( b_large, 0, m+2*n);
  bob::core::array::reindexAndResize( x_large, 0, m+2*n);

  // 'Compute' transpose(A)
  const blitz::Array<double,2> A_t = 
    const_cast<blitz::Array<double,2>&>(A).transpose(1,0);

  // Initialize
  A_large = 0.;

  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // block 0x0: A
  A_large( r_m, r_n) = A(r_m+A.lbound(0), r_n+A.lbound(1));
  // block 1x1: transpose(A)
  A_large( r_n+m, r_m+n) = A_t(r_n+A_t.lbound(0), r_m+A_t.lbound(1));
  // block 1x2: I
  for( int i=0; i<n; ++i)
    A_large(m+i, m+n+i) = 1.;

  b_large = 0.;
}

void mathdetail::updateLargeSystem(const blitz::Array<double,1>& x, 
  const blitz::Array<double,1>& mu, const double sigma, const int m,
  blitz::Array<double,2>& A_large, blitz::Array<double,1>& b_large)
{
  // Get dimensions from the A matrix
  const int n = x.extent(0);
 
  // Compute nu*sigma
  double nu_sigma = sigma * math::dot(x, mu) / n;

  // block 2x0: S_k
  for( int i=0; i<n; ++i)
    A_large( m+n+i, i) = mu(i+mu.lbound(0));
  // block 2x2: X_k
  for( int i=0; i<n; ++i)
    A_large( m+n+i, m+n+i) = x(i+x.lbound(0));
  // block 2: -X S e + nu sigma e
  blitz::Range r_n(0,n-1);
  b_large( r_n+m+n ) = -x(r_n+x.lbound(0))*mu(r_n+mu.lbound(0)) + nu_sigma;
}

void math::interiorpointShortstepNoInitLP(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta, blitz::Array<double,1>& x, 
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize variables
  const double sigma = 1. - theta / sqrt(n);
  double nu;

  // Declare and initialize arrays for the large linear system
  blitz::Array<double,2> A_large;
  blitz::Array<double,1> x_large; // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_large;

  mathdetail::initializeLargeSystem( A, A_large, b_large, x_large);

  int k=0;  
  while(true)
  {
    // 1) nu = 1/n <x,mu>
    nu = math::dot(x, mu) / n;
    // Stopping criterion
    if( nu < epsilon )
      break;

    // 2) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, sigma, m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 3) Update x, lamda and mu
    lambda( r_m+lambda.lbound(0)) += x_large( r_m+n);
    x( r_n+x.lbound(0)) += x_large( r_n);
    mu( r_n+mu.lbound(0)) += x_large( r_n+m+n);

    // 4) k = k + 1.
    ++k;
  }
}

void math::interiorpointPredictorCorrectorNoInitLP(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta_pred, blitz::Array<double,1>& x,
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare variable
  double nu;

  // Declare and initialize arrays for the large linear system
  blitz::Array<double,2> A_large;
  blitz::Array<double,1> x_large; // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_large;

  mathdetail::initializeLargeSystem( A, A_large, b_large, x_large);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  
  while(true)
  {
    /////////////////////////////
    // PREDICTION
    // 1) nu = 1/n <x,mu>
    nu = math::dot(x, mu) / n;
    // Stopping criterion
    if( nu < epsilon )
      break;

    // 2) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, 0., m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 3) alpha=1
    double alpha = 1.;
 
    // 4) Find alpha and update x, lamda and mu
    do {
      lambda_alpha(r_m) = lambda(r_m+lambda.lbound(0)) + alpha * x_large(r_m+n);
      x_alpha(r_n) = x(r_n+x.lbound(0)) + alpha * x_large(r_n);
      mu_alpha(r_n) = mu(r_n+mu.lbound(0)) + alpha * x_large(r_n+m+n);
      alpha /= 2.;
      if( alpha<2*std::numeric_limits<double>::epsilon())
        throw math::Exception();
    } while( !mathdetail::isInV2S(A,b,c,x_alpha,lambda_alpha,mu_alpha,epsilon,theta_pred));
    // Move content back
    lambda( r_m+lambda.lbound(0)) = lambda_alpha(r_m);
    x( r_n+x.lbound(0)) = x_alpha(r_n);
    mu( r_n+mu.lbound(0)) = mu_alpha(r_n);

    // 5) k = k + 1.
    ++k;


    /////////////////////////////
    // CORRECTION
    // 6) Update nu
    nu = math::dot(x, mu) / n;
    // Stopping criterion
    if( nu < epsilon )
      break;

    // 7) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, 1., m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 8) Update x
    lambda( r_m+lambda.lbound(0)) += x_large( r_m+n);
    x( r_n+x.lbound(0)) += x_large( r_n);
    mu( r_n+mu.lbound(0)) += x_large( r_n+m+n);

    // 9) k = k + 1
    ++k;
  } 
}

void math::interiorpointLongstepNoInitLP(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double gamma, const double sigma, blitz::Array<double,1>& x, 
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions
  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);
  blitz::Range r_m(0,m-1);
  blitz::Range r_n(0,n-1);

  // Declare and initialize variables
  double nu;

  // Declare and initialize arrays for the large linear system
  blitz::Array<double,2> A_large;
  blitz::Array<double,1> x_large; // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_large;

  mathdetail::initializeLargeSystem( A, A_large, b_large, x_large);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  
  while(true)
  {
    // 1) nu = 1/n <x,mu>
    nu = math::dot(x, mu) / n;
    // Stopping criterion
    if( nu < epsilon )
      break;

    // 2) Update the big system and solve it
    mathdetail::updateLargeSystem( x, mu, sigma, m, A_large, b_large);
    math::linsolve( A_large, x_large, b_large);

    // 3) alpha=1
    double alpha = 1.;

    // 4) Find alpha and update x, lamda and mu
    do {
      lambda_alpha(r_m) = lambda(r_m+lambda.lbound(0)) + alpha * x_large(r_m+n);
      x_alpha(r_n) = x(r_n+x.lbound(0)) + alpha * x_large(r_n);
      mu_alpha(r_n) = mu(r_n+mu.lbound(0)) + alpha * x_large(r_n+m+n);
      alpha /= 2.;
      if( alpha<2*std::numeric_limits<double>::epsilon())
        throw math::Exception();
    } while( !mathdetail::isInVinfS(A,b,c,x_alpha,lambda_alpha,mu_alpha,epsilon,gamma));
    // Move content back
    lambda( r_m+lambda.lbound(0)) = lambda_alpha(r_m);
    x( r_n+x.lbound(0)) = x_alpha(r_n);
    mu( r_n+mu.lbound(0)) = mu_alpha(r_n);

    // 5) k = k + 1.
    ++k;
  }
}

void math::interiorpointShortstepLP(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta, blitz::Array<double,1>& x, const double epsilon)
{
  // Initialize dual variables
  const int m=A.extent(0);
  const int n=A.extent(1);
  blitz::Array<double,1> lambda(m);
  lambda = 0.;
  blitz::Array<double,1> mu(n);
  mu = 0.;
  mathdetail::initializeDualLambdaMuLP( A, c, lambda, mu);

  // Find an initial solution
  mathdetail::centeringV2( A, theta, x, lambda, mu);

  // Launch the short step algorithm
  math::interiorpointShortstepNoInitLP( A, b, c, theta, x, lambda, mu, epsilon);
}

void math::interiorpointPredictorCorrectorLP(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta_pred, const double theta_corr, blitz::Array<double,1>& x, 
  const double epsilon)
{
  // Initialize dual variables
  const int m=A.extent(0);
  const int n=A.extent(1);
  blitz::Array<double,1> lambda(m);
  lambda = 0.;
  blitz::Array<double,1> mu(n);
  mu = 0.;
  mathdetail::initializeDualLambdaMuLP( A, c, lambda, mu);

  // Find an initial solution
  mathdetail::centeringV2( A, theta_corr, x, lambda, mu);

  // Launch the predictor-corrector algorithm
  math::interiorpointPredictorCorrectorNoInitLP( A, b, c, theta_pred, x, lambda, mu, epsilon);
}

void math::interiorpointLongstepLP(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double gamma, const double sigma, blitz::Array<double,1>& x, 
  const double epsilon)
{
  // Initialize dual variables
  const int m=A.extent(0);
  const int n=A.extent(1);
  blitz::Array<double,1> lambda(m);
  lambda = 0.;
  blitz::Array<double,1> mu(n);
  mu = 0.;
  mathdetail::initializeDualLambdaMuLP( A, c, lambda, mu);

  // Find an initial solution
  mathdetail::centeringVinf( A, gamma, x, lambda, mu);

  // Launch the long step algorithm
  math::interiorpointLongstepNoInitLP( A, b, c, gamma, sigma, x, lambda, mu, epsilon);
}
