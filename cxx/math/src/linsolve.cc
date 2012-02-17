/**
 * @file cxx/math/src/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include "math/linsolve.h"
#include "math/Exception.h"
#include "math/linear.h"
#include "core/array_assert.h"
#include "core/array_check.h"

namespace math = bob::math;
namespace ca = bob::core::array;

// Declaration of the external LAPACK function (Linear system solvers)
extern "C" void dgesv_( int *N, int *NRHS, double *A, int *lda, int *ipiv, 
  double *B, int *ldb, int *info);
extern "C" void dposv_( char* uplo, int *N, int *NRHS, double *A, int *lda,
  double *B, int *ldb, int *info);

void math::linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  // Check x and b
  ca::assertZeroBase(x);
  ca::assertZeroBase(b);
  ca::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  ca::assertZeroBase(A);
  ca::assertSameDimensionLength(A.extent(0), A.extent(1));
  ca::assertSameDimensionLength(A.extent(1), b.extent(0));

  math::linsolve_(A, x, b);
}

void math::linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK arrays
  int* ipiv = new int[b.extent(0)];
  double* A_lapack = new double[A.extent(0)*A.extent(1)];
  for(int i=0; i<A.extent(0)*A.extent(1); ++i)
    A_lapack[i] = 
      A( (i%A.extent(1)), (i/A.extent(1)) );
  double* x_lapack;
  bool x_direct_use = ca::isCZeroBaseContiguous(x);
  if( !x_direct_use )
    x_lapack = new double[b.extent(0)];
  else
    x_lapack = x.data();
  for(int i=0; i<b.extent(0); ++i)
    x_lapack[i] = b(i); 

  // Remaining variables
  int info = 0;  
  int N =  A.extent(0);
  int lda = N;
  int ldb = N;
  int NRHS = 1;
 
  // Call the LAPACK function 
  dgesv_( &N, &NRHS, A_lapack, &lda, ipiv, x_lapack, &ldb, &info );
 
  // Check info variable
  if( info != 0)
    throw math::LapackError("The LAPACK dgesv function returned a non-zero value.");

  // Copy result back to x if required
  if( !x_direct_use )
    for(int i=0; i<x.extent(0); ++i)
      x(i) = x_lapack[i];

  // Free memory
  if( !x_direct_use )
    delete [] x_lapack;
  delete [] A_lapack;
  delete [] ipiv;
}


void math::linsolveSympos(const blitz::Array<double,2>& A, 
  blitz::Array<double,1>& x, const blitz::Array<double,1>& b)
{
  // Check x and b
  ca::assertZeroBase(x);
  ca::assertZeroBase(b);
  ca::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  ca::assertZeroBase(A);
  ca::assertSameDimensionLength(A.extent(0), A.extent(1));
  ca::assertSameDimensionLength(A.extent(1), b.extent(0));

  math::linsolveSympos_(A, x, b);
}

void math::linsolveSympos_(const blitz::Array<double,2>& A, 
  blitz::Array<double,1>& x, const blitz::Array<double,1>& b)
{
  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK arrays
  double* A_lapack = new double[A.extent(0)*A.extent(1)];
  for(int i=0; i<A.extent(0)*A.extent(1); ++i)
    A_lapack[i] = 
      A( (i%A.extent(1)), (i/A.extent(1)));
  double* x_lapack;
  bool x_direct_use = ca::isCZeroBaseContiguous(x);
  if( !x_direct_use )
    x_lapack = new double[b.extent(0)];
  else
    x_lapack = x.data();
  for(int i=0; i<b.extent(0); ++i)
    x_lapack[i] = b(i); 

  // Remaining variables
  char uplo = 'U';
  int info = 0;  
  int N =  A.extent(0);
  int lda = N;
  int ldb = N;
  int NRHS = 1;
 
  // Call the LAPACK function 
  dposv_( &uplo, &N, &NRHS, A_lapack, &lda, x_lapack, &ldb, &info );
 
  // Check info variable
  if( info != 0)
    throw math::LapackError("The LAPACK dposv function returned a \
      non-zero value. This might be caused by a non-symmetric definite \
      positive matrix.");

  // Copy result back to x if required
  if( !x_direct_use )
    for(int i=0; i<x.extent(0); ++i)
      x(i) = x_lapack[i];

  // Free memory
  if( !x_direct_use )
    delete [] x_lapack;
  delete [] A_lapack;
}

void math::linsolveCGSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b, const double acc, const int max_iter)
{
  // Dimensionality of the problem
  int N = b.extent(0);

  // Check x and b
  ca::assertZeroBase(x);
  ca::assertZeroBase(b);
  ca::assertSameDimensionLength(x.extent(0), N);
  
  // Check A
  ca::assertZeroBase(A);
  ca::assertSameDimensionLength(A.extent(0), N);
  ca::assertSameDimensionLength(A.extent(1), N);

  math::linsolveCGSympos_(A, x, b, acc, max_iter);
}

void math::linsolveCGSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
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

