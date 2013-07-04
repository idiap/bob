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

#include <stdexcept>
#include <bob/math/lu.h>
#include <bob/core/assert.h>
#include <bob/core/array_copy.h>
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <algorithm>
#include <boost/shared_array.hpp>

// Declaration of the external LAPACK functions
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( const int *M, const int *N, double *A,
  const int *lda, int *ipiv, int *info);
// Cholesky decomposition of a real symmetric definite-positive matrix (dpotrf)
extern "C" void dpotrf_( const char *uplo, const int *N, double *A,
  const int *lda, int *info);


void bob::math::lu(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
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

  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(L);
  bob::core::array::assertZeroBase(U);
  bob::core::array::assertZeroBase(P);

  bob::core::array::assertSameShape(L,shapeL);
  bob::core::array::assertSameShape(U,shapeU);
  bob::core::array::assertSameShape(P,shapeP);

  bob::math::lu_(A, L, U, P);
}

void bob::math::lu_(const blitz::Array<double,2>& A, blitz::Array<double,2>& L,
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
    bob::core::array::ccopy(const_cast<blitz::Array<double,2>&>(A).transpose(1,0)));
  double *A_lapack = A_blitz_lapack.data();
  boost::shared_array<int> ipiv(new int[minMN]);

  // Calls the LAPACK function
  dgetrf_( &M, &N, A_lapack, &lda, ipiv.get(), &info);

  // Checks info variable
  if (info != 0)
    throw std::runtime_error("The LAPACK dgetrf function returned a non-zero value.");

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
  for (int i=0; i<minMN-1; ++i)
  {
    temp = Pp(ipiv[i]-1);
    Pp(ipiv[i]-1) = Pp(i);
    Pp(i) = temp;
  }
  // Updates P
  P = 0.;
  for (int j = 0; j<minMN; ++j)
    P(j,Pp(j)) = 1.;
}


void bob::math::chol(const blitz::Array<double,2>& A,
  blitz::Array<double,2>& L)
{
  // Size variable
  const int M = A.extent(0);
  const int N = A.extent(1);

  // Check
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(L);
  bob::core::array::assertSameDimensionLength(M,N);
  bob::core::array::assertSameShape(A,L);

  bob::math::chol_(A, L);
}

void bob::math::chol_(const blitz::Array<double,2>& A,
  blitz::Array<double,2>& L)
{
  // Size variable
  const int N = A.extent(0);

  // Prepares to call LAPACK function
  // Initialises LAPACK variables
  int info = 0;
  const int lda = N;
  const char uplo = 'L';

  // Initialises LAPACK arrays
  blitz::Array<double,2> A_blitz_lapack;
  // Tries to use V directly
  blitz::Array<double,2> Lt = L.transpose(1,0);
  const bool Lt_direct_use = bob::core::array::isCZeroBaseContiguous(Lt);
  if (Lt_direct_use)
  {
    A_blitz_lapack.reference(Lt);
    A_blitz_lapack = A;
  }
  else
    A_blitz_lapack.reference(bob::core::array::ccopy(A));
  double *A_lapack = A_blitz_lapack.data();

  // Calls the LAPACK function
  dpotrf_( &uplo, &N, A_lapack, &lda, &info);

  // Checks info variable
  if (info != 0)
    throw std::runtime_error("The LAPACK dpotrf function returned a non-zero value.");

  // Copy result back to L if required
  if (!Lt_direct_use)
    Lt = A_blitz_lapack;

  // Sets strictly upper triangular part to 0
  blitz::firstIndex i;
  blitz::secondIndex j;
  L = blitz::where(i < j, 0, L);
}

