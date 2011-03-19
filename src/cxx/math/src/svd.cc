#include "math/svd.h"

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
  int nb_singular = std::min(A.extent(0),A.extent(1));

  // Check and reindex if required
  if( U.base(0) != 0 || U.base(1) != 0) {
    const blitz::TinyVector<int,2> zero_base = 0;
    U.reindexSelf( zero_base );
  }
  if( sigma.base(0) != 0 ) {
    const blitz::TinyVector<int,1> zero_base = 0;
    sigma.reindexSelf( zero_base );
  }
  if( V.base(0) != 0 || V.base(1) != 0) {
    const blitz::TinyVector<int,2> zero_base = 0;
    V.reindexSelf( zero_base );
  }
  // Check and resize if required
  if( U.extent(0) != M || U.extent(1) != M)
    U.resize( M, M);
  if( sigma.extent(0) != nb_singular)
    sigma.resize( nb_singular );
  if( V.extent(0) != N || V.extent(1) != N)
    V.resize( N, N);


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
  double* A_lapack = new double[A.extent(0)*A.extent(1)];
  for(int j=0; j<M; ++j)
    for(int i=0; i<N; ++i)
      A_lapack[j+i*M] = A(j,i);
  double *S_lapack;
  bool sigma_direct_use = checkSafedata(sigma);
  if( !sigma_direct_use )
    S_lapack = new double[sigma.extent(0)];
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
    for(int i=0; i<sigma.extent(0); ++i)
      sigma(i+sigma.lbound(0)) = S_lapack[i];

  // Free memory
  if( !sigma_direct_use )
    delete [] S_lapack;
  delete [] A_lapack;
  delete [] U_lapack;
  delete [] VT_lapack;
  delete [] work;
}
