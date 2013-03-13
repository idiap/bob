/**
 * @file math/cxx/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/math/linsolve.h>
#include <bob/math/Exception.h>
#include <bob/math/linear.h>
#include <bob/core/assert.h>
#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <boost/shared_array.hpp>

// Declaration of the external LAPACK function (Linear system solvers)
extern "C" void dgesv_( const int *N, const int *NRHS, double *A, 
  const int *lda, int *ipiv, double *B, const int *ldb, int *info);
extern "C" void dposv_( const char* uplo, const int *N, const int *NRHS, 
  double *A, const int *lda, double *B, const int *ldb, int *info);

void bob::math::linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  // Check x and b
  bob::core::array::assertZeroBase(x);
  bob::core::array::assertZeroBase(b);
  bob::core::array::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  bob::core::array::assertSameDimensionLength(A.extent(1), b.extent(0));

  bob::math::linsolve_(A, x, b);
}

void bob::math::linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  // Defines dimensionality variables
  const int N = A.extent(0);

  // Prepares to call LAPACK function
  // Initialises LAPACK arrays
  boost::shared_array<int> ipiv(new int[N]);
  // Transpose (C: row major order, Fortran: column major)
  // Ugly fix for old blitz version support
  blitz::Array<double,2> A_blitz_lapack( 
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use X directly
  bool x_direct_use = bob::core::array::isCZeroBaseContiguous(x);
  blitz::Array<double,1> x_blitz_lapack;
  if (x_direct_use) 
  {
    x_blitz_lapack.reference(x);
    x_blitz_lapack = b;
  }
  else
    x_blitz_lapack.reference(bob::core::array::ccopy(b));
  double *x_lapack = x_blitz_lapack.data();
  // Remaining variables
  int info = 0;  
  const int lda = N;
  const int ldb = N;
  const int NRHS = 1;
 
  // Calls the LAPACK function (dgesv(
  dgesv_( &N, &NRHS, A_lapack, &lda, ipiv.get(), x_lapack, &ldb, &info );
 
  // Check info variable
  if (info != 0)
    throw bob::math::LapackError("The LAPACK dgesv function returned a non-zero value.");

  // Copy result back to x if required
  if (!x_direct_use)
    x = x_blitz_lapack;
}


void bob::math::linsolve(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
  const blitz::Array<double,2>& B)
{
  // Checks dimensionality and zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(X);
  bob::core::array::assertZeroBase(B);
  bob::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  bob::core::array::assertSameDimensionLength(A.extent(1), X.extent(0));
  bob::core::array::assertSameDimensionLength(A.extent(0), B.extent(0));
  bob::core::array::assertSameDimensionLength(X.extent(1), B.extent(1));

  bob::math::linsolve_(A, X, B);
}

void bob::math::linsolve_(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
  const blitz::Array<double,2>& B)
{
  // Defines dimensionality variables
  const int N = A.extent(0); 
  const int P = X.extent(1);

  // Prepares to call LAPACK function (dgesv)
  // Initialises LAPACK arrays
  boost::shared_array<int> ipiv(new int[N]);
  // Transpose (C: row major order, Fortran: column major)
  // Ugly fix for old blitz version support
  blitz::Array<double,2> A_blitz_lapack( 
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use X directly
  blitz::Array<double,2> Xt = X.transpose(1,0);
  bool X_direct_use = bob::core::array::isCZeroBaseContiguous(Xt);
  blitz::Array<double,2> X_blitz_lapack;
  if (X_direct_use) 
  {
    X_blitz_lapack.reference(Xt);
    // Ugly fix for old blitz version support
    X_blitz_lapack = const_cast<blitz::Array<double,2>&>(B).transpose(1,0);
  }
  else
    X_blitz_lapack.reference(
      bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(B).transpose(1,0)));
  double *X_lapack = X_blitz_lapack.data();
  // Remaining variables
  int info = 0;  
  const int lda = N;
  const int ldb = N;
  const int NRHS = P;
 
  // Calls the LAPACK function (dgesv)
  dgesv_( &N, &NRHS, A_lapack, &lda, ipiv.get(), X_lapack, &ldb, &info );
 
  // Checks info variable
  if (info != 0)
    throw bob::math::LapackError("The LAPACK dgesv function returned a non-zero value.");

  // Copy result back to X if required
  if (!X_direct_use )
    X = X_blitz_lapack.transpose(1,0);
}




void bob::math::linsolveSympos(const blitz::Array<double,2>& A, 
  blitz::Array<double,1>& x, const blitz::Array<double,1>& b)
{
  // Check x and b
  bob::core::array::assertZeroBase(x);
  bob::core::array::assertZeroBase(b);
  bob::core::array::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  bob::core::array::assertSameDimensionLength(A.extent(1), b.extent(0));

  bob::math::linsolveSympos_(A, x, b);
}

void bob::math::linsolveSympos_(const blitz::Array<double,2>& A, 
  blitz::Array<double,1>& x, const blitz::Array<double,1>& b)
{
  // Defines dimensionality variables
  const int N = A.extent(0);

  // Prepares to call LAPACK function
  // Initialises LAPACK arrays
  // Transpose (C: row major order, Fortran: column major)
  // Ugly fix for old blitz version support
  blitz::Array<double,2> A_blitz_lapack( 
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use X directly
  bool x_direct_use = bob::core::array::isCZeroBaseContiguous(x);
  blitz::Array<double,1> x_blitz_lapack;
  if (x_direct_use) 
  {
    x_blitz_lapack.reference(x);
    x_blitz_lapack = b;
  }
  else
    x_blitz_lapack.reference(bob::core::array::ccopy(b));
  double *x_lapack = x_blitz_lapack.data();
  // Remaining variables
  int info = 0;  
  const char uplo = 'U';
  const int lda = N;
  const int ldb = N;
  const int NRHS = 1;
 
  // Calls the LAPACK function (dposv)
  dposv_( &uplo, &N, &NRHS, A_lapack, &lda, x_lapack, &ldb, &info );
 
  // Check info variable
  if (info != 0)
    throw bob::math::LapackError("The LAPACK dposv function returned a \
      non-zero value. This might be caused by a non-symmetric definite \
      positive matrix.");

  // Copy result back to x if required
  if (!x_direct_use )
    x = x_blitz_lapack;
}

void bob::math::linsolveSympos(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
  const blitz::Array<double,2>& B)
{
  // Checks dimensionality and zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(X);
  bob::core::array::assertZeroBase(B);
  bob::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  bob::core::array::assertSameDimensionLength(A.extent(1), X.extent(0));
  bob::core::array::assertSameDimensionLength(A.extent(0), B.extent(0));
  bob::core::array::assertSameDimensionLength(X.extent(1), B.extent(1));

  bob::math::linsolveSympos_(A, X, B);
}

void bob::math::linsolveSympos_(const blitz::Array<double,2>& A, blitz::Array<double,2>& X,
  const blitz::Array<double,2>& B)
{
  // Defines dimensionality variables
  const int N = A.extent(0); 
  const int P = X.extent(1);

  // Prepares to call LAPACK function (dposv)
  // Initialises LAPACK arrays
  // Transpose (C: row major order, Fortran: column major)
  // Ugly fix for old blitz version support
  blitz::Array<double,2> A_blitz_lapack( 
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double* A_lapack = A_blitz_lapack.data();
  // Tries to use X directly
  blitz::Array<double,2> Xt = X.transpose(1,0);
  bool X_direct_use = bob::core::array::isCZeroBaseContiguous(Xt);
  blitz::Array<double,2> X_blitz_lapack;
  if (X_direct_use) 
  {
    X_blitz_lapack.reference(Xt);
    // Ugly fix for old blitz version support
    X_blitz_lapack = const_cast<blitz::Array<double,2>&>(B).transpose(1,0);
  }
  else
    X_blitz_lapack.reference(
      bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(B).transpose(1,0)));
  double *X_lapack = X_blitz_lapack.data();
  // Remaining variables
  int info = 0;  
  const char uplo = 'U';
  const int lda = N;
  const int ldb = N;
  const int NRHS = P;
 
  // Calls the LAPACK function (dposv)
  dposv_( &uplo, &N, &NRHS, A_lapack, &lda, X_lapack, &ldb, &info );
 
  // Check info variable
  if (info != 0)
    throw bob::math::LapackError("The LAPACK dposv function returned a \
      non-zero value. This might be caused by a non-symmetric definite \
      positive matrix.");

  // Copy result back to X if required
  if (!X_direct_use)
    X = X_blitz_lapack.transpose(1,0);
}




void bob::math::linsolveCGSympos(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b, const double acc, const int max_iter)
{
  // Dimensionality of the problem
  const int N = b.extent(0);

  // Check x and b
  bob::core::array::assertZeroBase(x);
  bob::core::array::assertZeroBase(b);
  bob::core::array::assertSameDimensionLength(x.extent(0), N);
  
  // Check A
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertSameDimensionLength(A.extent(0), N);
  bob::core::array::assertSameDimensionLength(A.extent(1), N);

  bob::math::linsolveCGSympos_(A, x, b, acc, max_iter);
}

void bob::math::linsolveCGSympos_(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b, const double acc, const int max_iter)
{
  // Dimensionality of the problem
  const int N = b.extent(0);

  blitz::Array<double,1> r(N), d(N), best_x(N), q(N), tmp(N);
  x = 0.;
  r = b;
  d = b;

  double delta = bob::math::dot(r,r);
  double delta0 = bob::math::dot(b,b);

  int n_iter = 0;
  best_x = x;
  double best_res = sqrt(delta / delta0);

  while (n_iter < max_iter && delta > acc*acc*delta0)
  {
    // q = A*d
    bob::math::prod_(A, d, q);

    // alpha = delta/(d'*q);
    double alpha = delta / bob::math::dot(d,q);
    x = x + alpha * d;

    if (n_iter+1 % 50 == 0)
    {
      bob::math::prod(A,x,tmp);
      r = b - tmp;
    }
    else
      r = r - alpha * q;
      
    double delta_old = delta;
    delta = bob::math::dot(r,r);
    double beta = delta / delta_old;
    d = r + beta * d;
    ++n_iter;

    if (sqrt(delta/delta0) < best_res)
    {
      best_x = x;
      best_res = sqrt(delta/delta0);
    }
  }

  x = best_x;

  // TODO return best_res and number of iterations?
  //double res = best_res;
}

