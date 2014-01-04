/**
 * @file math/cxx/det.cc
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/math/det.h>
#include <bob/math/linear.h>
#include <bob/math/lu.h>
#include <bob/core/assert.h>
#include <limits>

double bob::math::det(const blitz::Array<double,2>& A)
{
  bob::core::array::assertSameDimensionLength(A.extent(0),A.extent(1));
  return bob::math::det_(A);
}

double bob::math::det_(const blitz::Array<double,2>& A)
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
  for (int i=0; i<N; ++i) 
  {
    for (int j=i+1; j<N; ++j)
      if (P(i,j) > 0)
      {
        s = -s; 
        break;
      }
    Udiag *= U(i,i);
  }

  return s*Udiag;
}

double bob::math::slogdet(const blitz::Array<double,2>& A, int& sign)
{
  bob::core::array::assertSameDimensionLength(A.extent(0),A.extent(1));
  return bob::math::slogdet_(A, sign);
}

double bob::math::slogdet_(const blitz::Array<double,2>& A, int& sign)
{
  // Size variable
  int N = A.extent(0);

  // Perform an LU decomposition
  blitz::Array<double,2> L(N,N);
  blitz::Array<double,2> U(N,N);
  blitz::Array<double,2> P(N,N);
  math::lu(A, L, U, P);

  // Compute the determinant of A = det(P*L)*SI(diag(U))
  //  where det(P*L) = +- 1 (Number of permutation in P)
  //  and SI(diag(log|U|)) is the sum of the logarithm of the 
  //  diagonal elements of U
  blitz::Array<double,2> Lperm(N,N);
  math::prod(P,L,Lperm);
  sign = 1;
  double Udiag=0.;
  for (int i=0; i<N; ++i) 
  {
    for (int j=i+1; j<N; ++j)
      if (P(i,j) > 0)
      {
        sign = -sign; 
        break;
      }
    Udiag += log(fabs(U(i,i)));
  }
  // Check for infinity
  if ((Udiag*-1) == std::numeric_limits<double>::infinity()) 
    sign = 0;

  return Udiag;
}

