#include "math/linsolve.h"
#include "core/array_old.h"

namespace math = Torch::math;

// Declaration of the external LAPACK function (Generic linear system solver)
extern "C" void dgesv_( int *N, int *NRHS, double *A, int *lda, int *ipiv, 
  double *B, int *ldb, int *info);


void math::linsolve(const blitz::Array<double,2>& A, blitz::Array<double,1>& x,
  const blitz::Array<double,1>& b)
{
  // Check and reindex if required
  if( x.base(0) != 0 ) {
    const blitz::TinyVector<int,1> zero_base = 0;
    x.reindexSelf( zero_base );
  }
  // Check and resize x if required
  if( x.extent(0) != b.extent(0) )
    x.resize( b.extent(0) );

  // Check dimensions of the inputs and throw exception if required
  if( A.extent(1) != b.extent(0) || A.extent(1) != A.extent(0)) 
  {
    // TODO: define exceptions
    throw Torch::core::Exception();
  }

  ///////////////////////////////////
  // Prepare to call LAPACK function

  // Initialize LAPACK arrays
  int* ipiv = new int[b.extent(0)];
  double* A_lapack = new double[A.extent(0)*A.extent(1)];
  for(int i=0; i<A.extent(0)*A.extent(1); ++i)
    A_lapack[i] = 
      A( (i%A.extent(1)) + A.lbound(0), (i/A.extent(1)) + A.lbound(1) );
  double* x_lapack;
  bool x_direct_use = checkSafedata(x);
  if( !x_direct_use )
    x_lapack = new double[b.extent(0)];
  else
    x_lapack = x.data();
  for(int i=0; i<b.extent(0); ++i)
    x_lapack[i] = b(i+b.lbound(0)); 

  // Remaining variables
  int info = 0;  
  int N =  A.extent(0);
  int lda = N;
  int ldb = N;
  int NRHS = 1;
 
  // Call the LAPACK function 
  dgesv_( &N, &NRHS, A_lapack, &lda, ipiv, x_lapack, &ldb, &info );
 
  // Copy result back to x if required
  if( !x_direct_use )
    for(int i=0; i<x.extent(0); ++i)
      x(i+x.lbound(0)) = x_lapack[i];

  // Free memory
  if( !x_direct_use )
    delete [] x_lapack;
  delete [] A_lapack;
  delete [] ipiv;
}
