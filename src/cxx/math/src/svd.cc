/**
 * @file cxx/math/src/svd.cc
 * @date Sat Mar 19 22:14:10 2011 +0100
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
#include "math/svd.h"
#include "core/array_assert.h"
#include "core/array_old.h"

namespace math = Torch::math;

// Declaration of the external LAPACK function (Generic linear system solver)
extern "C" void dgesvd_( char *jobu, char *jobvt, int *M, int *N, double *A, 
  int *lda, double *S, double *U, int* ldu, double *VT, int *ldvt, 
  double *work, int *lwork, int *info);


void math::svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& V)
{
  // Size variables
  int M = A.extent(0);
  int N = A.extent(1);
  int nb_singular = std::min(M,N);

  // Checks zero base
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(U);
  Torch::core::array::assertZeroBase(sigma);
  Torch::core::array::assertZeroBase(V);
  // Checks and resizes if required
  Torch::core::array::assertSameDimensionLength(U.extent(0), M);
  Torch::core::array::assertSameDimensionLength(U.extent(1), M);
  Torch::core::array::assertSameDimensionLength(sigma.extent(0), nb_singular);
  Torch::core::array::assertSameDimensionLength(V.extent(0), N);
  Torch::core::array::assertSameDimensionLength(V.extent(1), N);

  math::svd_(A, U, sigma, V);
}

void math::svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma, blitz::Array<double,2>& V)
{
  // Size variables
  int M = A.extent(0);
  int N = A.extent(1);
  int nb_singular = std::min(M,N);

  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  char jobu = 'A'; // Get All left singular vectors
  char jobvt = 'A'; // Get All right singular vectors
  int info = 0;  
  int lda = M;
  int ldu = M;
  int ldvt = N;
  int lwork = std::max(3*std::min(M,N)+std::max(M,N),5*std::min(M,N));

  // Initialize LAPACK arrays
  double *work = new double[lwork];
  double *U_lapack = new double[M*M];
  double *VT_lapack = new double[N*N];
  double* A_lapack = new double[M*N];
  for(int j=0; j<M; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*M] = A(j,i);
  double *S_lapack;
  bool sigma_direct_use = checkSafedata(sigma);
  if( !sigma_direct_use )
    S_lapack = new double[nb_singular];
  else
    S_lapack = sigma.data();
 
  // Call the LAPACK function 
  dgesvd_( &jobu, &jobvt, &M, &N, A_lapack, &lda, S_lapack, U_lapack, &ldu, 
    VT_lapack, &ldvt, work, &lwork, &info );
 
  // Copy singular vectors back to U and V (column-major order)
  for(int j=0; j<M; ++j)
    for(int i=0; i<M; ++i)
      U(j,i) = U_lapack[j+i*M];
  // Please not that LAPACK returns transpose(V). We return V!
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
      V(j,i) = VT_lapack[j*N+i];

  // Copy result back to sigma if required
  if( !sigma_direct_use )
    for(int i=0; i<nb_singular; ++i)
      sigma(i) = S_lapack[i];

  // Free memory
  if( !sigma_direct_use )
    delete [] S_lapack;
  delete [] A_lapack;
  delete [] U_lapack;
  delete [] VT_lapack;
  delete [] work;
}


void math::svd(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma)
{
  // Size variables
  int M = A.extent(0);
  int N = A.extent(1);
  int nb_singular = std::min(M,N);

  // Checks zero base
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(U);
  Torch::core::array::assertZeroBase(sigma);
  // Checks and resizes if required
  Torch::core::array::assertSameDimensionLength(U.extent(0), M);
  Torch::core::array::assertSameDimensionLength(U.extent(1), nb_singular);
  Torch::core::array::assertSameDimensionLength(sigma.extent(0), nb_singular);
 
  math::svd_(A, U, sigma);
}

void math::svd_(const blitz::Array<double,2>& A, blitz::Array<double,2>& U,
  blitz::Array<double,1>& sigma)
{
  // Size variables
  int M = A.extent(0);
  int N = A.extent(1);
  int nb_singular = std::min(M,N);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  char jobu = 'S'; // Get All left singular vectors
  char jobvt = 'N'; // Get All right singular vectors
  int info = 0;  
  int lda = M;
  int ldu = M;
  int ldvt = N;
  int lwork = std::max(3*std::min(M,N)+std::max(M,N),5*std::min(M,N));

  // Initialize LAPACK arrays
  double *work = new double[lwork];
  double *U_lapack = new double[M*nb_singular];
  double *VT_lapack = 0;
  double* A_lapack = new double[M*N];
  for(int j=0; j<M; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*M] = A(j,i);
  double *S_lapack;
  bool sigma_direct_use = checkSafedata(sigma);
  if( !sigma_direct_use )
    S_lapack = new double[nb_singular];
  else
    S_lapack = sigma.data();
 
  // Call the LAPACK function 
  dgesvd_( &jobu, &jobvt, &M, &N, A_lapack, &lda, S_lapack, U_lapack, &ldu, 
    VT_lapack, &ldvt, work, &lwork, &info );
 
  // Copy singular vectors back to U and V (column-major order)
  for(int j=0; j<M; ++j)
    for(int i=0; i<nb_singular; ++i)
      U(j,i) = U_lapack[j+i*M];

  // Copy result back to sigma if required
  if( !sigma_direct_use )
    for(int i=0; i<nb_singular; ++i)
      sigma(i) = S_lapack[i];

  // Free memory
  if( !sigma_direct_use )
    delete [] S_lapack;
  delete [] A_lapack;
  delete [] U_lapack;
  delete [] work;
}
