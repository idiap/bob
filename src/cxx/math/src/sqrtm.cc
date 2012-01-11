#include "math/sqrtm.h"

#include "core/array_assert.h"
#include "math/eig.h"
#include "math/linear.h"


namespace tca = bob::core::array;
namespace math = bob::math;


void math::sqrtSymReal(const blitz::Array<double,2>& A, 
  blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,2> shape(N,N);
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(B);

  bob::core::array::assertSameShape(A,shape);
  bob::core::array::assertSameShape(B,shape);

  math::sqrtSymReal_(A, B);
}

void math::sqrtSymReal_(const blitz::Array<double,2>& A, 
  blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);

  // 1/ Perform the Eigenvalue decomposition of the symmetric matrix
  //    A = V.D.V^T, and V^-1=V^T
  blitz::Array<double,2> V(N,N);
  blitz::Array<double,2> Vt = V.transpose(1,0);
  blitz::Array<double,1> D(N);
  blitz::Array<double,2> tmp(N,N); // Cache for multiplication
  math::eigSymReal_(A,V,D);

  // 2/ Updates the diagonal matrix D, such that D=sqrt(|D|)
  //    |.| is used to deal with values close to zero (-epsilon)
  // TODO: check positiveness of the eigenvalues (with an epsilon tolerance)?
  D = blitz::sqrt(blitz::abs(D));

  // 3/ Compute the square root matrix B = V.sqrt(D).V^T
  //    B.B = V.sqrt(D).V^T.V.sqrt(D).V^T = V.sqrt(D).sqrt(D).V^T = A
  blitz::firstIndex i;
  blitz::secondIndex j;
  tmp = V(i,j) * D(j); // tmp = V.sqrt(D)
  math::prod_(tmp, Vt, B); // B = V.sqrt(D).V^T  
}
