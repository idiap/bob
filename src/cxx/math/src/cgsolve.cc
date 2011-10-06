#include "math/cgsolve.h"
#include "math/linear.h"
#include "core/array_assert.h"

namespace math = Torch::math;

void math::cgsolveSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b, const double acc, const int max_iter)
{
  // Dimensionality of the problem
  int N = b.extent(0);

  // Check x and b
  Torch::core::array::assertZeroBase(x);
  Torch::core::array::assertZeroBase(b);
  Torch::core::array::assertSameDimensionLength(x.extent(0), N);
  
  // Check A
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertSameDimensionLength(A.extent(0), N);
  Torch::core::array::assertSameDimensionLength(A.extent(1), N);

  math::cgsolveSympos_(A, x, b, acc, max_iter);
}

void math::cgsolveSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b, const double acc, const int max_iter)
{
  // Dimensionality of the problem
  int N = b.extent(0);

  blitz::Array<double,1> r(N), d(N), best_x(N), q(N), tmp(N);
  x = 0.;
  r = b;
  d = b;

  double delta = math::dot(r,r);
  double delta0 = math::dot(b,b);

  int n_iter = 0;
  best_x = x;
  double best_res = sqrt(delta / delta0);

  while( n_iter < max_iter && delta > acc*acc*delta0)
  {
    // q = A*d
    math::prod_(A, d, q);

    // alpha = delta/(d'*q);
    double alpha = delta / math::dot(d,q);
    x = x + alpha * d;

    if( n_iter+1 % 50 == 0)
    {
      math::prod(A,x,tmp);
      r = b - tmp;
    }
    else
      r = r - alpha * q;
      
    double delta_old = delta;
    delta = math::dot(r,r);
    double beta = delta / delta_old;
    d = r + beta * d;
    ++n_iter;

    if( sqrt(delta/delta0) < best_res)
    {
      best_x = x;
      best_res = sqrt(delta/delta0);
    }
  }

  x = best_x;

  // TODO return best_res and number of iterations?
  //double res = best_res;
}

