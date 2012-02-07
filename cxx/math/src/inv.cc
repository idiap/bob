/**
 * @file cxx/math/src/inv.cc
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "math/inv.h"
#include "math/Exception.h"
#include "core/array_assert.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif

namespace math = bob::math;
namespace ca = bob::core::array;

// Declaration of the external LAPACK functions
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( int *M, int *N, double *A, int *lda, int *ipiv, 
  int *info);
// Inverse of a general matrix (dgetri)
extern "C" void dgetri_( int *N, double *A, int *lda, int *ipiv, double *work,
  int *lwork, int *info);

void math::inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,2> shapeA(N,N);
  ca::assertZeroBase(A);
  ca::assertZeroBase(B);

  ca::assertSameShape(A,shapeA);
  ca::assertSameShape(B,shapeA);

  math::inv_(A, B);
}

void math::inv_(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  int info = 0;  
  int lda = N;
  int lwork = N;

  // Initialize LAPACK arrays
  double *A_lapack = new double[N*N];
  int *ipiv = new int[N];
  double *work = new double[N];
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*N] = A(j,i);
 
  // Call the LAPACK functions
  // 1/ Compute LU decomposition
  dgetrf_( &N, &N, A_lapack, &lda, ipiv, &info);
 
  // Check info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetrf function returned a \
      non-zero value.");

  // 2/ Compute the inverse
  dgetri_( &N, A_lapack, &lda, ipiv, work, &lwork, &info);
 
  // Check info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetri function returned a \
      non-zero value. The matrix might not be invertible.");

  // Copy result back to B
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
      B(j,i) = A_lapack[j+i*N];

  // Free memory
  delete [] A_lapack;
  delete [] ipiv;
  delete [] work;
}

