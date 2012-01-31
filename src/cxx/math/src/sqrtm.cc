/**
 * @file cxx/math/src/sqrtm.cc
 * @date Wed Oct 12 18:02:26 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the (unique) square root of
 * a real symmetric definite-positive matrix.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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


#include "math/sqrtm.h"

#include "core/array_assert.h"
#include "math/eig.h"
#include "math/linear.h"


namespace ca = bob::core::array;
namespace math = bob::math;


void math::sqrtSymReal(const blitz::Array<double,2>& A, 
  blitz::Array<double,2>& B)
{
  // Size variable
  int N = A.extent(0);
  const blitz::TinyVector<int,2> shape(N,N);
  ca::assertZeroBase(A);
  ca::assertZeroBase(B);

  ca::assertSameShape(A,shape);
  ca::assertSameShape(B,shape);

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
