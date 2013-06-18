/**
 * @file math/cxx/test/pinv.cc
 * @date Tue Jun 18 18:55:23 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the pseudo-inverse
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE math-pinv Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <bob/math/pinv.h>


struct T {
  blitz::Array<double,2> A23_1, A33_1, A32_1;
  blitz::Array<double,2> B32_1, B33_1, B23_1;
  double eps;

  T(): A23_1(2,3), A33_1(3,3), A32_1(3,2),
    B32_1(3,2), B33_1(3,3), B23_1(2,3),
    eps(2e-4)
  {
    A23_1 = 0.8147, 0.1270, 0.6324, 0.9058, 0.9134, 0.0975;
    A33_1 = -0.4336, 2.7694, 0.7254, 0.3426, -1.3499, -0.0631, 3.5784, 3.0349, 0.7147;
    A32_1 = -0.2050, 1.4090, -0.1241, 1.4172, 1.4897, 0.6715;

    B32_1 = 0.5492, 0.2421, -0.6519, 0.9075, 1.0047, -0.4942;
    B33_1 = -0.2350, 0.0675, 0.2445, -0.1430, -0.8831, 0.0672, 1.7842, 3.4120, -0.1105;
    B23_1 = -0.1694, -0.1333, 0.6369, 0.3374, 0.3349, 0.0743;
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

BOOST_AUTO_TEST_CASE( test_pinv_2x3 )
{
  blitz::Array<double,2> B(3,2);
  bob::math::pinv(A23_1, B);
  checkBlitzClose(B, B32_1, eps);
}

BOOST_AUTO_TEST_CASE( test_pinv_3x3 )
{
  blitz::Array<double,2> B(3,3);
  bob::math::pinv(A33_1, B);
  checkBlitzClose(B, B33_1, eps);
}

BOOST_AUTO_TEST_CASE( test_pinv_3x2 )
{
  blitz::Array<double,2> B(2,3);
  bob::math::pinv(A32_1, B);
  checkBlitzClose(B, B23_1, eps);
}

BOOST_AUTO_TEST_SUITE_END()

