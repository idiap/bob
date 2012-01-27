/**
 * @file cxx/math/test/lu_det_inv.cc
 * @date Tue Jun 7 01:00:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the LU decomposition and the determinant
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
#define BOOST_TEST_MODULE math-lu_det_inv Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/cast.h"
#include "math/lu.h"
#include "math/det.h"
#include "math/inv.h"
#include "math/linear.h"


struct T {
  blitz::Array<double,2> A33_1, A24_1;
  blitz::Array<double,2> L33_1, L24_1;
  blitz::Array<double,2> U33_1, U24_1;
  blitz::Array<double,2> P33_1, P24_1;
  blitz::Array<double,2> A33_1_inv, I33;
  double det_A33_1, eps;

  T(): A33_1(3,3), A24_1(2,4), L33_1(3,3), L24_1(2,2), U33_1(3,3), U24_1(2,4), 
    P33_1(3,3), P24_1(2,2), A33_1_inv(3,3), I33(3,3), det_A33_1(-0.2766), 
    eps(2e-4)
  {
    A33_1 = 0.8147, 0.9134, 0.2785, 0.9058, 0.6324, 0.5469, 0.1270, 0.0975, 
      0.9575;
    L33_1 = 1.0000, 0., 0., 0.8994, 1.0000, 0., 0.1402, 0.0256, 1.0000;
    U33_1 = 0.9058, 0.6324, 0.5469, 0., 0.3446, -0.2134, 0., 0., 0.8863;
    P33_1 = 0, 1, 0, 1, 0, 0, 0, 0, 1;

    A24_1 = 0.7922, 0.6557, 0.8491, 0.6787,
      0.9595, 0.0357, 0.9340, 0.7577;
    L24_1 = 1., 0., 0.8256, 1.;
    U24_1 = 0.9595, 0.0357, 0.9340, 0.7577, 0., 0.6262, 0.0780, 0.0531;
    P24_1 = 0, 1, 1, 0;

    A33_1_inv = -1.9960, 3.0632, -1.1690, 2.8840, -2.6919, 0.6987, -0.0289, 
                -0.1322, 1.1283;
    I33 = 1., 0., 0., 0., 1., 0., 0., 0., 1.;

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

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,1>& t1, blitz::Array<U,1>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_EQUAL(t1(i), bob::core::cast<T>(t2(i)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<U,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), bob::core::cast<T>(t2(i,j,k)));
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

BOOST_AUTO_TEST_CASE( test_lu_3x3 )
{
  blitz::Array<double,2> L(3,3);
  blitz::Array<double,2> U(3,3);
  blitz::Array<double,2> P(3,3);

  bob::math::lu(A33_1, L, U, P);

  checkBlitzClose(L, L33_1, eps);
  checkBlitzClose(U, U33_1, eps);
  checkBlitzClose(P, P33_1, eps);
}
  
BOOST_AUTO_TEST_CASE( test_lu_2x4 )
{
  blitz::Array<double,2> L(2,2);
  blitz::Array<double,2> U(2,4);
  blitz::Array<double,2> P(2,2);

  bob::math::lu(A24_1, L, U, P);

  checkBlitzClose(L, L24_1, eps);
  checkBlitzClose(U, U24_1, eps);
  checkBlitzClose(P, P24_1, eps);
}
  
BOOST_AUTO_TEST_CASE( test_det_3x3 )
{
  blitz::Array<double,2> det(3,3);

  BOOST_CHECK_SMALL( fabs(bob::math::det(A33_1) - det_A33_1), eps);
}

BOOST_AUTO_TEST_CASE( test_inv_3x3 )
{
  blitz::Array<double,2> inv(3,3);

  // Invert a matrix and compare to reference
  bob::math::inv(A33_1, inv);
  checkBlitzClose(inv, A33_1_inv, eps);

  blitz::Array<double,2> I(3,3);
  bob::math::prod(A33_1, inv, I);
  checkBlitzClose(I, I33, eps);
}

BOOST_AUTO_TEST_SUITE_END()

