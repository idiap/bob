/**
 * @file math/cxx/inv.cc
 * @date Fri Jan 27 14:10:23 2012 +0100
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
#include "bob/math/inv.h"
#include "bob/math/linear.h"
#include "bob/math/Exception.h"
#include "bob/core/array_assert.h"
#include "bob/core/array_check.h"
#include "bob/core/array_copy.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif

namespace math = bob::math;
namespace ca = bob::core::array;

// Declaration of the external LAPACK function
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( const int *M, const int *N, double *A, const int *lda, 
  int *ipiv, int *info);
// Inverse of a general matrix (dgetri)
extern "C" void dgetri_( const int *N, double *A, const int *lda, 
  const int *ipiv, double *work, const int *lwork, int *info);

void math::inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  const int N = A.extent(0);
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
  const int N = A.extent(0);

  //////////////////////////////////////
  // Prepares to call LAPACK functions
  // Initializes LAPACK variables
  int info = 0;  
  const int lda = N;

  // Initializes LAPACK arrays
  int *ipiv = new int[N];

  // Tries to use B directly if possible 
  //   Input and output arrays are both column-major order.
  //   Hence, we can ignore the problem of column- and row-major order
  //   conversions.
  bool B_direct_use = ca::isCZeroBaseContiguous(B);
  blitz::Array<double,2> A_blitz_lapack;
  if(B_direct_use) 
  {
    A_blitz_lapack.reference(B);
    A_blitz_lapack = A;
  }
  else
    A_blitz_lapack.reference(ca::ccopy(A));
  double *A_lapack = A_blitz_lapack.data();


  // Calls the LAPACK functions
  // 1/ Computes the LU decomposition
  dgetrf_( &N, &N, A_lapack, &lda, ipiv, &info); 
  // Checks the info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetrf function returned a \
      non-zero value.");

  // TODO: We might consider adding a real invertibility test as described in
  // this thread (Btw, this is what matlab does): 
  // http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00778.html

  // 2/ Computes the inverse matrix
  // 2/A/ Queries the optimal size of the working array
  const int lwork_query = -1;
  double work_query;
  dgetri_( &N, A_lapack, &lda, ipiv, &work_query, &lwork_query, &info);
  // 2/B/ Computes the inverse
  const int lwork = static_cast<int>(work_query);
  double *work = new double[lwork];
  dgetri_( &N, A_lapack, &lda, ipiv, work, &lwork, &info);
  // Checks info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetri function returned a \
      non-zero value. The matrix might not be invertible.");

  // Copy back content to B if required
  if(!B_direct_use)
    B = A_blitz_lapack;

  // Free memory
  delete [] work;
  delete [] ipiv;
}

