/**
 * @file math/cxx/det.cc
 * @date Fri Jan 27 14:10:23 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
#include "bob/math/linear.h"
#include "bob/math/det.h"
#include "bob/math/lu.h"
#include "bob/math/Exception.h"
#include "bob/core/array_assert.h"
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <algorithm>

namespace math = bob::math;
namespace ca = bob::core::array;

double math::det(const blitz::Array<double,2>& A)
{
  ca::assertSameDimensionLength(A.extent(0),A.extent(1));
  return math::det_(A);
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

