#include "math/eig.h"
#include "core/array_assert.h"
#include "core/array_old.h"
#include <blitz/tinyvec-et.h>

namespace math = Torch::math;

// Declaration of the external LAPACK function (Eigenvalue decomposition of 
// real symmetric matrix)
extern "C" void dsyev_( char *jobu, char *uplo, int *N, double *A, 
  int *lda, double *W, double *work, int *lwork, int *info);


void math::eigSymReal(const blitz::Array<double,2>& A, 
  blitz::Array<double,2>& V, blitz::Array<double,1>& D)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,1> shape1(N);
  const blitz::TinyVector<int,2> shape2(N,N);
  // TODO: assert square matrix A.extent(0) == A.extent(1)
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(V);
  Torch::core::array::assertZeroBase(D);

  Torch::core::array::assertSameShape(A,shape2);
  Torch::core::array::assertSameShape(A,V);
  Torch::core::array::assertSameShape(D,shape1);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  char jobu = 'V'; // Get both the eigenvalues and the eigenvectors
  char uplo = 'U';
  int info = 0;  
  int lda = N;
  int lwork = 2*std::max(1,3*N-1);

  // Initialize LAPACK arrays
  double *work = new double[lwork];
  double* A_lapack = new double[N*N];
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*N] = A(j,i);
  double *D_lapack;
  bool D_direct_use = checkSafedata(D);
  if( !D_direct_use )
    D_lapack = new double[N];
  else
    D_lapack = D.data();
 
  // Call the LAPACK function 
  dsyev_( &jobu, &uplo, &N, A_lapack, &lda, D_lapack, work, &lwork, &info);
 
  // Copy singular vectors back to V (column-major order)
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
      V(j,i) = A_lapack[j+i*N];

  // Copy result back to sigma if required
  if( !D_direct_use )
    for(int i=0; i<N; ++i)
      D(i) = D_lapack[i];

  // Free memory
  if( !D_direct_use )
    delete [] D_lapack;
  delete [] A_lapack;
  delete [] work;
}
