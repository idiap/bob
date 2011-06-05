#include "math/eig.h"
#include "core/array_assert.h"
#include "core/array_old.h"
#include <blitz/tinyvec-et.h>

namespace math = Torch::math;

// Declaration of the external LAPACK functions
// Eigenvalue decomposition of real symmetric matrix (dsyev)
extern "C" void dsyev_( char *jobz, char *uplo, int *N, double *A, 
  int *lda, double *W, double *work, int *lwork, int *info);
// Generalized eigenvalue decomposition of real symmetric matrices (dsygv)
extern "C" void dsygv_( int *itype, char *jobz, char *uplo, int *N, double *A, 
  int *lda, double *B, int *ldb, double *W, double *work, int *lwork, int *info);


void math::eigSymReal(const blitz::Array<double,2>& A, 
  blitz::Array<double,2>& V, blitz::Array<double,1>& D)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,1> shape1(N);
  const blitz::TinyVector<int,2> shape2(N,N);
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(V);
  Torch::core::array::assertZeroBase(D);

  Torch::core::array::assertSameShape(A,shape2);
  Torch::core::array::assertSameShape(A,V);
  Torch::core::array::assertSameShape(D,shape1);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  char jobz = 'V'; // Get both the eigenvalues and the eigenvectors
  char uplo = 'U';
  int info = 0;  
  int lda = N;
  int lwork = 2*std::max(1,3*N-1);

  // Initialize LAPACK arrays
  double *work = new double[lwork];
  double *A_lapack = new double[N*N];
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
  dsyev_( &jobz, &uplo, &N, A_lapack, &lda, D_lapack, work, &lwork, &info);
 
  // TODO: check info variable

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


void math::eig(const blitz::Array<double,2>& A, const blitz::Array<double,2>& B,
  blitz::Array<double,2>& V, blitz::Array<double,1>& D)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,1> shape1(N);
  const blitz::TinyVector<int,2> shape2(N,N);
  Torch::core::array::assertZeroBase(A);
  Torch::core::array::assertZeroBase(B);
  Torch::core::array::assertZeroBase(V);
  Torch::core::array::assertZeroBase(D);

  Torch::core::array::assertSameShape(A,shape2);
  Torch::core::array::assertSameShape(B,shape2);
  Torch::core::array::assertSameShape(A,V);
  Torch::core::array::assertSameShape(D,shape1);


  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK variables
  int itype = 1;
  char jobz = 'V'; // Get both the eigenvalues and the eigenvectors
  char uplo = 'U';
  int info = 0;  
  int lda = N;
  int ldb = N;
  int lwork = 2*std::max(1,3*N-1);

  // Initialize LAPACK arrays
  double *work = new double[lwork];
  double *A_lapack = new double[N*N];
  double *B_lapack = new double[N*N];
  for(int j=0; j<N; ++j)
    for(int i=0; i<N; ++i)
    {
      A_lapack[j+i*N] = A(j,i);
      B_lapack[j+i*N] = B(j,i);
    }
  double *D_lapack;
  bool D_direct_use = checkSafedata(D);
  if( !D_direct_use )
    D_lapack = new double[N];
  else
    D_lapack = D.data();
 
  // Call the LAPACK function 
  dsygv_( &itype, &jobz, &uplo, &N, A_lapack, &lda, B_lapack, &ldb, D_lapack, 
    work, &lwork, &info);

  // TODO: check info variable
 
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
  delete [] B_lapack;
  delete [] work;
}
