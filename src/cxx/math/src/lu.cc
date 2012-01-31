/**
 * @file cxx/math/src/lu.cc
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
#include "math/lu.h"
#include "math/Exception.h"
#include "core/array_assert.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <algorithm>

namespace math = bob::math;
namespace ca = bob::core::array;

// Declaration of the external LAPACK functions
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( int *M, int *N, double *A, int *lda, int *ipiv, 
  int *info);


void math::lu(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
  blitz::Array<double,2>& U, blitz::Array<double,2>& P)
{
  // Size variable
  int M = A.extent(0);
  int N = A.extent(1);
  int minMN = std::min(M,N);
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
  int M = A.extent(0);
  int N = A.extent(1);
  int minMN = std::min(M,N);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  int info = 0;  
  int lda = M;

  // Initialize LAPACK arrays
  double *A_lapack = new double[M*N];
  int *ipiv = new int[minMN];
  for(int j=0; j<M; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*M] = A(j,i);
 
  // Call the LAPACK function 
  dgetrf_( &M, &N, A_lapack, &lda, ipiv, &info);
 
  // Check info variable
  if( info != 0)
    throw bob::math::LapackError("The LAPACK dgetrf function returned a non-zero value.");

  // Copy result back to L
  L = 0.;
  for(int j=0; j<minMN; ++j)
    L(j,j) = 1.;;
  for(int j=0; j<M; ++j)
    for(int i=0; i<std::min(j,minMN); ++i)
      L(j,i) = A_lapack[j+i*M];

  // Copy result back to U
  U = 0.;
  for(int j=0; j<minMN; ++j)
    for(int i=j; i<N; ++i)
      U(j,i) = A_lapack[j+i*M];

  // Convert weird permutation format returned by LAPACK into a permutation function
  blitz::Array<int,1> Pp(minMN);
  blitz::firstIndex ind;
  Pp = ind;
  int temp;
  for( int i=0; i<minMN-1; ++i)
  {
    temp = Pp(ipiv[i]-1);
    Pp(ipiv[i]-1) = Pp(i);
    Pp(i) = temp;
  }
  // Update P
  P = 0.;
  //std::cout << ipiv << std::endl;
  for(int j = 0; j<minMN; ++j)
    P(j,Pp(j)) = 1.;
  
  // Free memory
  delete [] A_lapack;
  delete [] ipiv;
}

