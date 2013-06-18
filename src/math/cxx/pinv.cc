/**
 * @file math/cxx/pinv.cc
 * @date Tue Jun 18 18:31:12 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <bob/math/pinv.h>
#include <bob/math/svd.h>
#include <bob/math/linear.h>
#include <bob/core/assert.h>

void bob::math::pinv(const blitz::Array<double,2>& A,
  blitz::Array<double,2>& B, const double rcond)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);

  // Checks zero base
  bob::core::array::assertZeroBase(A);
  bob::core::array::assertZeroBase(B);
  // Checks size
  bob::core::array::assertSameDimensionLength(B.extent(0), N);
  bob::core::array::assertSameDimensionLength(B.extent(1), M);

  // Calls pinv_()
  bob::math::pinv_(A, B, rcond);
}

void bob::math::pinv_(const blitz::Array<double,2>& A,
  blitz::Array<double,2>& B, const double rcond)
{
  // Size variables
  const int M = A.extent(0);
  const int N = A.extent(1);
  const int nb_singular = std::min(M,N);

  // Allocates arrays for the SVD
  blitz::Array<double,2> U(M,M);
  blitz::Array<double,1> sigma(nb_singular);
  blitz::Array<double,1> sigma_full(N);
  blitz::Array<double,2> Vt(N,N);
  blitz::Array<double,2> sigmai_Ut(N, M);
  blitz::Array<double,1> Ut_sum(M);
  // Computes the SVD
  bob::math::svd_(A, U, sigma, Vt);

  // Cuts off small sigmas for the inverse similar to Numpy:
  // cf. https://github.com/numpy/numpy/blob/maintenance/1.7.x/numpy/linalg/linalg.py#L1577
  const double cutoff = rcond*blitz::max(sigma);
  sigma = blitz::where(sigma > cutoff, 1./sigma, 0.);
  sigma_full(blitz::Range(0,nb_singular-1)) = sigma(blitz::Range(0,nb_singular-1));

  // Computes the pseudo inverse: B = V.'sigma^-1'*U^T
  // a. Because of potential boundaries
  sigmai_Ut = 0.;
  // b. 'sigma^-1'*U^T except for the boundaries
  //    Slice sigma
  blitz::Array<double,2> sigmai_Ut_s = sigmai_Ut(blitz::Range(0,nb_singular-1),blitz::Range::all());
  blitz::Array<double,2> Ut = U.transpose(1,0);
  blitz::firstIndex bi;
  blitz::secondIndex bj;
  sigmai_Ut_s = sigma_full(bi) * Ut(bi,bj);
  // Final computation: B = V.'sigma^-1'*U^T
  blitz::Array<double,2> V = Vt.transpose(1,0);
  bob::math::prod(V, sigmai_Ut, B);
}

