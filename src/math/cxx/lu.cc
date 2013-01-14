/**
 * @file math/cxx/lu.cc
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
#include "bob/math/lu.h"
#include "bob/math/Exception.h"
#include "bob/core/array_assert.h"
#include "bob/core/array_copy.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <algorithm>

namespace math = bob::math;
namespace ca = bob::core::array;

// Declaration of the external LAPACK functions
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( const int *M, const int *N, double *A, 
  const int *lda, int *ipiv, int *info);


void math::lu(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
  blitz::Array<double,2>& U, blitz::Array<double,2>& P)
{
  // Size variable
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int minMN = std::min(M,N);

  // Check
  const blitz::TinyVector<int,2> shapeL(M,minMN);
  const blitz::TinyVector<int,2> shapeU(minMN,N);
  const blitz::TinyVector<int,2> shapeP(minMN,minMN);

  ca::assertZeroBase(A);
  ca::assertZeroBase(L);
  ca::assertZeroBase(U);
  ca::assertZeroBase(P);

  ca::assertSameShape(L,shapeL);
  ca::assertSameShape(U,shapeU);
  ca::assertSameShape(P,shapeP);

  math::lu_(A, L, U, P);
}

void math::lu_(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
  blitz::Array<double,2>& U, blitz::Array<double,2>& P)
{
  // Size variable
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int minMN = std::min(M,N);

  // Prepares to call LAPACK function

  // Initialises LAPACK variables
  int info = 0;  
  const int lda = M;

  // Initialises LAPACK arrays
  blitz::Array<double,2> A_blitz_lapack(
    ca::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double *A_lapack = A_blitz_lapack.data();
  int *ipiv = new int[minMN];

  // Calls the LAPACK function 
  dgetrf_( &M, &N, A_lapack, &lda, ipiv, &info);
 
  // Checks info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetrf function returned a \
      non-zero value.");

  // Copy result back to L and U
  blitz::firstIndex bi;
  blitz::secondIndex bj;
  blitz::Array<double,2> A_blitz_lapack_t = A_blitz_lapack.transpose(1,0);
  blitz::Range rall = blitz::Range::all();
  L = blitz::where(bi>bj, A_blitz_lapack_t(rall,blitz::Range(0,minMN-1)), 0.);
  L = blitz::where(bi==bj, 1., L);
  U = blitz::where(bi<=bj, A_blitz_lapack_t(blitz::Range(0,minMN-1),rall), 0.);

  // Converts weird permutation format returned by LAPACK into a permutation 
  // function
  blitz::Array<int,1> Pp(minMN);
  Pp = bi;
  int temp;
  for( int i=0; i<minMN-1; ++i)
  {
    temp = Pp(ipiv[i]-1);
    Pp(ipiv[i]-1) = Pp(i);
    Pp(i) = temp;
  }
  // Updates P
  P = 0.;
  for(int j = 0; j<minMN; ++j)
    P(j,Pp(j)) = 1.;
  
  // Free memory
  delete [] ipiv;
}

