/**
 * @file cxx/math/test/svd.cc
 * @date Sat Mar 19 22:14:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Singular Value Decomposition
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE math-svd Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "math/linear.h"
#include "math/svd.h"


struct T {
  blitz::Array<double,2> A33_1, A24_1;
  blitz::Array<double,2> U33_1, U22_1;
  blitz::Array<double,1> S3_1, S2_1;
  blitz::Array<double,2> V33_1, V44_1;
  double eps;

  T(): A33_1(3,3), A24_1(2,4), U33_1(3,3), U22_1(2,2), S3_1(3), S2_1(2), 
    V33_1(3,3), V44_1(4,4), eps(2e-4)
  {
    A33_1 = 0.8147, 0.9134, 0.2785, 0.9058, 0.6324, 0.5469, 0.1270, 0.0975, 
      0.9575;
    U33_1 = -0.6612, -0.4121, -0.6269, -0.6742, -0.0400, 0.7374, -0.3290, 
      0.9103, -0.2514;
    S3_1 = 1.8168, 0.8389, 0.1815;
    V33_1 = -0.6557, -0.3056, 0.6904, -0.5848, -0.3730, -0.7204, -0.4777,
      0.8761, -0.0658;

    A24_1 = 0.7922, 0.6557, 0.8491, 0.6787,
      0.9595, 0.0357, 0.9340, 0.7577;
    U22_1 = -0.6961, -0.7179, -0.7179, 0.6961;
    S2_1 = 2.0966, 0.4604;
    V44_1 = -0.591595, 0.215378, 0.606365, 0.485745,
      -0.22993, -0.968629, 0.0650405, 0.0682623,
      -0.601755, 0.0880797, 0.0173668, -0.793619,
      -0.484807, 0.0872455, -0.792332, 0.359945;
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

BOOST_AUTO_TEST_CASE( test_svd_3x3 )
{
  blitz::Array<double,2> U(3,3);
  blitz::Array<double,1> S(3);
  blitz::Array<double,2> V(3,3);

  // Full SVD function
  bob::math::svd(A33_1, U, S, V);
  checkBlitzClose(S3_1, S, eps);

  // Check that A33_1 == U*S*V'
  // Note that the singular vectors are not unique
  blitz::Array<double,2> US(3,3);
  blitz::Array<double,2> USVt(3,3);
  blitz::firstIndex i;
  blitz::secondIndex j;
  US = U(i,j) * S(j);
  blitz::Array<double,2> Vt = V.transpose(1,0);
  bob::math::prod(US, Vt, USVt);
  checkBlitzClose(A33_1, USVt, eps);

  // Economic SVD function 
  bob::math::svd(A33_1, U, S);
  checkBlitzClose(S3_1, S, eps);
}
  
BOOST_AUTO_TEST_CASE( test_svd_2x4 )
{
  blitz::Array<double,2> U(2,2);
  blitz::Array<double,1> S(2);
  blitz::Array<double,2> V(4,4);

  // Full SVD function
  bob::math::svd(A24_1, U, S, V);
  checkBlitzClose(S2_1, S, eps);
  // Check that A24_1 == U*S*V'
  // Note that the singular vectors are not unique
  blitz::Array<double,2> US(2,4);
  blitz::Array<double,2> USVt(2,4);
  blitz::firstIndex i;
  blitz::secondIndex j;
  US = 0.;
  US(blitz::Range::all(), blitz::Range(0,1)) = U(i,j) * S(j);
  blitz::Array<double,2> Vt = V.transpose(1,0);
  bob::math::prod(US, Vt, USVt);
  checkBlitzClose(A24_1, USVt, eps);

  // Economic SVD function 
  bob::math::svd(A24_1, U, S);
  checkBlitzClose(S2_1, S, eps);
}

BOOST_AUTO_TEST_SUITE_END()

