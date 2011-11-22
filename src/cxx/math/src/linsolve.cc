/**
 * @file cxx/math/src/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
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
#include "math/linsolve.h"
#include "math/Exception.h"
#include "core/array_old.h"
#include "core/array_assert.h"

namespace math = Torch::math;

// Declaration of the external LAPACK function (Linear system solvers)
extern "C" void dgesv_( int *N, int *NRHS, double *A, int *lda, int *ipiv, 
  double *B, int *ldb, int *info);
extern "C" void dposv_( char* uplo, int *N, int *NRHS, double *A, int *lda,
  double *B, int *ldb, int *info);

void math::linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  // Check x and b
  Torch::core::array::assertZeroBase(x);
  Torch::core::array::assertZeroBase(b);
  Torch::core::array::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  Torch::core::array::assertSameDimensionLength(A.extent(1), b.extent(0));

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
  bool x_direct_use = checkSafedata(x);
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
    throw Torch::math::LapackError("The LAPACK dgesv function returned a non-zero value.");

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
  Torch::core::array::assertZeroBase(x);
  Torch::core::array::assertZeroBase(b);
  Torch::core::array::assertSameDimensionLength(x.extent(0), b.extent(0));
  
  // Check A
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertSameDimensionLength(A.extent(0), A.extent(1));
  Torch::core::array::assertSameDimensionLength(A.extent(1), b.extent(0));

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
  bool x_direct_use = checkSafedata(x);
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
    throw Torch::math::LapackError("The LAPACK dposv function returned a \
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
