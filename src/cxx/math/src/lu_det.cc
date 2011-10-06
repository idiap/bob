#include "math/linear.h"
#include "math/lu_det.h"
#include "math/Exception.h"
#include "core/array_assert.h"
#include "core/array_old.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <algorithm>

namespace math = Torch::math;

// Declaration of the external LAPACK functions
// LU decomposition of a general matrix (dgetrf)
extern "C" void dgetrf_( int *M, int *N, double *A, int *lda, int *ipiv, 
  int *info);
// Inverse of a general matrix (dgetri)
extern "C" void dgetri_( int *N, double *A, int *lda, int *ipiv, double *work,
  int *lwork, int *info);


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
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(L);
  Torch::core::array::assertZeroBase(U);
  Torch::core::array::assertZeroBase(P);

  Torch::core::array::assertSameShape(L,shapeL);
  Torch::core::array::assertSameShape(U,shapeU);
  Torch::core::array::assertSameShape(P,shapeP);

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
    throw Torch::math::LapackError("The LAPACK dgetrf function returned a non-zero value.");

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

double math::det(const blitz::Array<double,2>& A)
{
  // Size variable
  int N = A.extent(0);
  Torch::core::array::assertSameDimensionLength(A.extent(0),A.extent(1));

  math::det_(A);
}

double math::det_(const blitz::Array<double,2>& A)
{
  // Size variable
  int N = A.extent(0);

  // Perform an LU decomposition
  blitz::Array<double,2> L(N,N);
  blitz::Array<double,2> U(N,N);
  blitz::Array<double,2> P(N,N);
  math::lu(A, L, U, P);

  // Compute the determinant of A = det(P*L)*PI(diag(U))
  //  where det(P*L) = +- 1 (Number of permutation in P)
  //  and PI(diag(U)) is the product of the diagonal elements of U
  blitz::Array<double,2> Lperm(N,N);
  math::prod(P,L,Lperm);
  int s = 1;
  double Udiag=1.;
  for( int i=0; i<N; ++i) 
  {
    for(int j=i+1; j<N; ++j)
      if( P(i,j) > 0)
      {
        s = -s; 
        break;
      }
    Udiag *= U(i,i);
  }

  return s*Udiag;
}


void math::inv(const blitz::Array<double,2>& A, blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,2> shapeA(N,N);
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(B);

  Torch::core::array::assertSameShape(A,shapeA);
  Torch::core::array::assertSameShape(B,shapeA);

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
    throw Torch::math::LapackError("The LAPACK dgetrf function returned a \
      non-zero value.");

  // 2/ Compute the inverse
  dgetri_( &N, A_lapack, &lda, ipiv, work, &lwork, &info);
 
  // Check info variable
  if( info != 0)
    throw Torch::math::LapackError("The LAPACK dgetri function returned a \
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

