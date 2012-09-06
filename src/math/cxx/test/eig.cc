/**
 * @file math/cxx/test/eig.cc
 * @date Fri Jan 27 17:30:18 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Eigenvalue Decomposition
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE math-eig Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "bob/math/eig.h"
#include "bob/math/inv.h"
#include "bob/math/linear.h"
#include <algorithm>

struct T {
  blitz::Array<double,2> A33_1, B33_1;
  blitz::Array<double,1> S3_1, S3_2;
  double eps;

  T(): A33_1(3,3), B33_1(3,3), S3_1(3), S3_2(3),
    eps(2e-4)
  {
    A33_1 = 1., 2., 3., 2., 4., 5., 3., 5., 6.;
    B33_1 = 2., -1., 0., -1., 2., -1., 0., -1., 2.;
    S3_1 = -0.5157, 0.1709, 11.3448;
    S3_2 = -0.2728, 0.0510, 17.9718;
  }

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions( blitz::Array<T,d>& t1, blitz::Array<U,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,1>& t1, blitz::Array<T,1>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs( t2(i)-t1(i) ), eps);
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs( t2(i,j)-t1(i,j) ), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_eigSym_3x3 )
{
  blitz::Array<double,2> V(3,3);
  blitz::Array<double,1> S(3);

  // Calls eigenvalue decomposition function
  bob::math::eigSym(A33_1, V, S);

  // Check eigenvalues
  checkBlitzClose(S3_1, S, eps);

  // Check A = V.S.inv(V)
  blitz::Array<double,2> iV(3,3), dS(3,3), VS(3,3), VSiV(3,3);
  bob::math::inv(V, iV);
  bob::math::diag(S, dS);
  bob::math::prod(V, dS, VS);
  bob::math::prod(VS, iV, VSiV);
  checkBlitzClose(A33_1, VSiV, eps);
}

BOOST_AUTO_TEST_CASE( test_eigSymGen_3x3 )
{
  blitz::Array<double,2> V(3,3);
  blitz::Array<double,1> S(3);

  // Calls eigenvalue decomposition function
  bob::math::eigSym(A33_1, B33_1, V, S);

  // Check eigenvalues
  checkBlitzClose(S3_2, S, eps);
}

BOOST_AUTO_TEST_SUITE_END()

