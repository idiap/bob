/**
 * @file cxx/math/src/cgsolve.cc
 * @date Mon Jun 27 21:14:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "math/cgsolve.h"
#include "math/linear.h"
//#include "math/Exception.h"
//#include "core/array_old.h"
//#include "core/array_assert.h"

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

