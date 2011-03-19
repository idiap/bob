#include "math/interiorpoint.h"
#include "math/linsolve.h"

namespace math = Torch::math;

bool math::isFeasible(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon)
{
  //TODO: check dimensions

  // x >= 0
  for( int i=0; i<x.extent(0); ++i)
    if( x(i+x.lbound(i)) < 0. )
      return false;
  // mu >= 0
  for( int i=0; i<mu.extent(0); ++i)
    if( mu(i+mu.lbound(i)) < 0. )
      return false;

  // A*x = b (abs(A*x-b)<=epsilon)
  for( int i=0; i<A.extent(0); ++i) {
    double cond = 0.;
    for( int j=0; j<A.extent(1); ++j)
      cond += A(i+A.lbound(0),j+A.lbound(1))*x(j+x.lbound(0));
    cond -= b(i+b.lbound(0));
    if( fabs(cond) > epsilon )
      return false;
  }

  // A'*lambda + mu = c (abs(A'*lambda+mu-c)<=epsilon)
  for( int i=0; i<A.extent(1); ++i) {
    double cond = 0.;
    for( int j=0; j<A.extent(0); ++j)
      cond += A(j+A.lbound(0),i+A.lbound(1))*lambda(j+lambda.lbound(0));
    cond += mu(i+mu.lbound(0)) - c(i+c.lbound(0));
    if( fabs(cond) > epsilon )
      return false;
  }

  return true;
}

bool math::isInV2(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon, const double theta)
{
  //TODO: check dimensions

  // Check if the tested 'solution' belongs to the set of feasible solutions
  if( !math::isFeasible(A,b,c,x,lambda,mu,epsilon) )
    return false;

  // Check that the L2 norm is smaller than theta
  // 1) compute nu
  double nu = 0.;
  for( int i=0; i<x.extent(0); ++i)
    nu += x(i+x.lbound(0))*mu(i+mu.lbound(0));
  nu /= x.extent(0);

  // 2) compute the L2 norm
  double norm = 0.;
  for( int i=0; i<x.extent(0); ++i) {
    double val_i = x(i+x.lbound(0))*mu(i+mu.lbound(0))-nu;
    norm += val_i*val_i;
  }
  norm = sqrt(norm) / nu;
  // 3) check
  if( norm > theta )
    return false;

  return true;
}

bool math::isInVinf(const blitz::Array<double,2>& A,
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
  const blitz::Array<double,1>& mu, const double epsilon, const double gamma)
{
  //TODO: check dimensions

  // Check if the tested 'solution' belongs to the set of feasible solutions
  if( !math::isFeasible(A,b,c,x,lambda,mu,epsilon) )
    return false;

  // Check using Linf norm
  // 1) compute nu
  double nu = 0.;
  for( int i=0; i<x.extent(0); ++i)
    nu += x(i+x.lbound(0))*mu(i+mu.lbound(0));
  nu /= x.extent(0);

  // 2) check
  for( int i=0; i<x.extent(0); ++i)
    if(  x(i+x.lbound(0))*mu(i+mu.lbound(0)) < gamma*nu )
      return false;

  return true;
}

void math::interiorpointShortstep(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta, blitz::Array<double,1>& x, 
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);

  // Declare and initialize variables
  const double sigma = 1. - theta / sqrt(n);
  double nu;

  // Declare and allocate arrays for the large linear system
  blitz::Array<double,2> A_bigsystem(m+2*n,m+2*n);
  blitz::Array<double,1> x_bigsystem(m+2*n); // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_bigsystem(m+2*n);

  int k=0;  

  do {
    // 1) nu = 1/n <x,mu>
    nu = 0.;
    for( int i=0; i<n; ++i)
      nu += x( i+x.lbound(0))*mu( i+mu.lbound(0));
    nu /= n;


    // 2) Update the big system and solve it
    if( k==0 ) {
      A_bigsystem = 0.;
      // block 0x0: A
      for( int i=0; i<m; ++i)
        for( int j=0; j<n; ++j)
          A_bigsystem(i,j) = A(i+A.lbound(0), j+A.lbound(1));
      // block 1x1: A'
      for( int i=0; i<n; ++i)
        for( int j=0; j<m; ++j)
          A_bigsystem(m+i,n+j) = A(j+A.lbound(0), i+A.lbound(1));
      // block 1x2: I
      for( int i=0; i<n; ++i)
        A_bigsystem(m+i, m+n+i) = 1.;

      b_bigsystem = 0.;
    }    
    // block 2x0: S_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, i) = mu(i+mu.lbound(0));
    // block 2x2: X_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, m+n+i) = x(i+x.lbound(0));
    // block 2: -X S e + nu sigma e
    for( int i=0; i<n; ++i)
      b_bigsystem( m+n+i) = -x(i+x.lbound(0))*mu(i+mu.lbound(0)) + nu*sigma;

    // Solve the big linear system
    linsolve( A_bigsystem, x_bigsystem, b_bigsystem);


    // 3) Update x, lamda and mu
    for( int i=0; i<m; ++i)
      lambda(i+lambda.lbound(0)) += x_bigsystem(i+n);
    for( int i=0; i<n; ++i) {
      x(i+x.lbound(0)) += x_bigsystem(i);
      mu(i+mu.lbound(0)) += x_bigsystem(i+m+n);
    }


    // 4) k = k + 1.
    ++k;

  // Stopping criterion
  } while( nu > epsilon);
}

void math::interiorpointPredictorCorrector(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double theta_pred, const double theta_corr, blitz::Array<double,1>& x,
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);

  // Declare variable
  double nu;

  // Declare and allocate arrays for the large linear system
  blitz::Array<double,2> A_bigsystem(m+2*n,m+2*n);
  blitz::Array<double,1> x_bigsystem(m+2*n); // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_bigsystem(m+2*n);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  

  do {
    /////////////////////////////
    // PREDICTION

    // 1) nu = 1/n <x,mu>
    nu = 0.;
    for( int i=0; i<n; ++i)
      nu += x( i+x.lbound(0))*mu( i+mu.lbound(0));
    nu /= n;


    // 2) Update the big system and solve it
    if( k==0 ) {
      A_bigsystem = 0.;
      // block 0x0: A
      for( int i=0; i<m; ++i)
        for( int j=0; j<n; ++j)
          A_bigsystem(i,j) = A(i+A.lbound(0), j+A.lbound(1));
      // block 1x1: A'
      for( int i=0; i<n; ++i)
        for( int j=0; j<m; ++j)
          A_bigsystem(m+i,n+j) = A(j+A.lbound(0), i+A.lbound(1));
      // block 1x2: I
      for( int i=0; i<n; ++i)
        A_bigsystem(m+i, m+n+i) = 1.;

      b_bigsystem = 0.;
    }    
    // block 2x0: S_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, i) = mu(i+mu.lbound(0));
    // block 2x2: X_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, m+n+i) = x(i+x.lbound(0));
    // block 2: -X S e + nu sigma e
    for( int i=0; i<n; ++i)
      b_bigsystem( m+n+i) = -x(i+x.lbound(0))*mu(i+mu.lbound(0));

    // Solve the big linear system
    linsolve( A_bigsystem, x_bigsystem, b_bigsystem);


    // 3) alpha=1
    double alpha = 1.;

 
    // 4) Find alpha and update x, lamda and mu
    // TODO: check if alpha close to zero and throw exception if required
    do {
      for( int i=0; i<m; ++i)
        lambda_alpha(i) = lambda(i+lambda.lbound(0)) + alpha*x_bigsystem(i+n);
      for( int i=0; i<n; ++i) {
        x_alpha(i) = x(i+x.lbound(0)) + alpha*x_bigsystem(i);
        mu_alpha(i) = mu(i+mu.lbound(0)) + alpha*x_bigsystem(i+m+n);
      }
    } while( !isInV2(A,b,c,x_alpha,lambda_alpha,mu_alpha,epsilon,theta_pred));
    // Move content back
    for( int i=0; i<m; ++i)
      lambda(i+lambda.lbound(0)) = lambda_alpha(i);
    for( int i=0; i<n; ++i) {
      x(i+x.lbound(0)) = x_alpha(i);
      mu(i+mu.lbound(0)) = mu_alpha(i);
    }


    // 5) k = k + 1.
    ++k;

    // CHECK if nu < epsilon TODO: is it really required
    if( nu < epsilon )
      break;


    /////////////////////////////
    // CORRECTION
    // 6) Update nu

    // 7) Update the big system and solve it
    // block 2x0: S_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, i) = mu(i+mu.lbound(0));
    // block 2x2: X_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, m+n+i) = x(i+x.lbound(0));
    // block 2: -X S e + nu sigma e
    for( int i=0; i<n; ++i)
      b_bigsystem( m+n+i) = -x(i+x.lbound(0))*mu(i+mu.lbound(0)) + nu;

    // Solve the big linear system
    linsolve( A_bigsystem, x_bigsystem, b_bigsystem);


    // 8) Update x
    for( int i=0; i<m; ++i)
      lambda(i+lambda.lbound(0)) += x_bigsystem(i+n);
    for( int i=0; i<n; ++i) {
      x(i+x.lbound(0)) += x_bigsystem(i);
      mu(i+mu.lbound(0)) += x_bigsystem(i+m+n);
    }


    // 9) k = k + 1
    ++k;

  // Stopping criterion
  } while( nu > epsilon);
}

void math::interiorpointLongstep(const blitz::Array<double,2>& A, 
  const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
  const double gamma, const double sigma, blitz::Array<double,1>& x, 
  blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
  const double epsilon)
{
  //TODO: check dimensions

  // Get dimensions from the A matrix
  const int m = A.extent(0);
  const int n = A.extent(1);

  // Declare and initialize variables
  double nu;

  // Declare and allocate arrays for the large linear system
  blitz::Array<double,2> A_bigsystem(m+2*n,m+2*n);
  blitz::Array<double,1> x_bigsystem(m+2*n); // i.e. (d_x,d_lambda,d_mu)
  blitz::Array<double,1> b_bigsystem(m+2*n);

  blitz::Array<double,1> x_alpha( x.extent(0));
  blitz::Array<double,1> lambda_alpha( lambda.extent(0));
  blitz::Array<double,1> mu_alpha( mu.extent(0));

  int k=0;  

  do {
    // 1) nu = 1/n <x,mu>
    nu = 0.;
    for( int i=0; i<n; ++i)
      nu += x( i+x.lbound(0))*mu( i+mu.lbound(0));
    nu /= n;


    // 2) Update the big system and solve it
    if( k==0 ) {
      A_bigsystem = 0.;
      // block 0x0: A
      for( int i=0; i<m; ++i)
        for( int j=0; j<n; ++j)
          A_bigsystem(i,j) = A(i+A.lbound(0), j+A.lbound(1));
      // block 1x1: A'
      for( int i=0; i<n; ++i)
        for( int j=0; j<m; ++j)
          A_bigsystem(m+i,n+j) = A(j+A.lbound(0), i+A.lbound(1));
      // block 1x2: I
      for( int i=0; i<n; ++i)
        A_bigsystem(m+i, m+n+i) = 1.;

      b_bigsystem = 0.;
    }    
    // block 2x0: S_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, i) = mu(i+mu.lbound(0));
    // block 2x2: X_k
    for( int i=0; i<n; ++i)
      A_bigsystem( m+n+i, m+n+i) = x(i+x.lbound(0));
    // block 2: -X S e + nu sigma e
    for( int i=0; i<n; ++i)
      b_bigsystem( m+n+i) = -x(i+x.lbound(0))*mu(i+mu.lbound(0)) + nu*sigma;

    // Solve the big linear system
    linsolve( A_bigsystem, x_bigsystem, b_bigsystem);


    // 3) alpha=1
    double alpha = 1.;

 
    // 4) Find alpha and update x, lamda and mu
    // TODO: check if alpha close to zero and throw exception if required
    do {
      for( int i=0; i<m; ++i)
        lambda_alpha(i) = lambda(i+lambda.lbound(0)) + alpha*x_bigsystem(i+n);
      for( int i=0; i<n; ++i) {
        x_alpha(i) = x(i+x.lbound(0)) + alpha*x_bigsystem(i);
        mu_alpha(i) = mu(i+mu.lbound(0)) + alpha*x_bigsystem(i+m+n);
      }
    } while( !isInVinf(A,b,c,x_alpha,lambda_alpha,mu_alpha,epsilon,gamma));
    // Move content back
    for( int i=0; i<m; ++i)
      lambda(i+lambda.lbound(0)) = lambda_alpha(i);
    for( int i=0; i<n; ++i) {
      x(i+x.lbound(0)) = x_alpha(i);
      mu(i+mu.lbound(0)) = mu_alpha(i);
    }


    // 5) k = k + 1.
    ++k;

  // Stopping criterion
  } while( nu > epsilon);
}

