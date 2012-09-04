/**
 * @file cxx/math/test/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the linear solvers A*x=b
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
#define BOOST_TEST_MODULE math-linsolve Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "bob/core/cast.h"
#include "bob/math/linsolve.h"


struct T {
  blitz::Array<double,2> A33_1, A33_2, A33_3, B33_1, S33_2, S33_3;
  blitz::Array<double,1> b3_1, b3_2, s3_1, s3_2, s3_3;
  double eps;

  T(): A33_1(3,3), A33_2(3,3), A33_3(3,3), B33_1(3,3), S33_2(3,3), S33_3(3,3),
        b3_1(3), b3_2(3), s3_1(3), s3_2(3), s3_3(3), eps(1e-6)
  {
    A33_1 = 1., 0., 0., 0., 1., 0., 0., 0., 1.;
    b3_1 = 7., 5., 3.;
    s3_1 = b3_1;
    A33_2 = 1., 3., 5., 7., 9., 1., 3., 5., 7.;
    b3_2 = 2., 4., 6.;
    s3_2 = 3., -2., 1.;
    A33_3 = 2., -1., 0., -1, 2., -1., 0., -1., 2.;
    s3_3 = 8.5, 10., 6.5;
    B33_1 = 4., 23., 5., 7., 8., 2., 1., 9., 5.;
    S33_2 = -5.45, -24.7, -2.2, 5.15, 20.4, 1.9, -1.2, -2.7, 0.3;
    S33_3 = 6.75, 23.5, 6., 9.5, 24., 7., 5.25, 16.5, 6.;
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

BOOST_AUTO_TEST_CASE( test_solve_3x3_vector )
{
  blitz::Array<double,1> x(3);

  bob::math::linsolve(A33_1, x, b3_1);
  checkBlitzClose(s3_1, x, eps); 

  bob::math::linsolve(A33_2, x, b3_2);
  checkBlitzClose(s3_2, x, eps); 

  bob::math::linsolve(A33_3, x, b3_1);
  checkBlitzClose(s3_3, x, eps); 
}

BOOST_AUTO_TEST_CASE( test_solve_3x3_matrix )
{
  blitz::Array<double,2> X(3,3);

  bob::math::linsolve(A33_1, X, B33_1);
  checkBlitzClose(B33_1, X, eps); 

  bob::math::linsolve(A33_2, X, B33_1);
  checkBlitzClose(S33_2, X, eps); 

  bob::math::linsolve(A33_3, X, B33_1);
  checkBlitzClose(S33_3, X, eps); 
}

BOOST_AUTO_TEST_CASE( test_solveSympos_3x3_vector )
{
  blitz::Array<double,1> x(3);

  bob::math::linsolveSympos(A33_1, x, b3_1);
  checkBlitzClose(s3_1, x, eps); 

  bob::math::linsolveSympos(A33_3, x, b3_1);
  checkBlitzClose(s3_3, x, eps); 
}

BOOST_AUTO_TEST_CASE( test_solveSympos_3x3_matrix )
{
  blitz::Array<double,2> X(3,3);

  bob::math::linsolveSympos(A33_1, X, B33_1);
  checkBlitzClose(B33_1, X, eps); 

  bob::math::linsolveSympos(A33_3, X, B33_1);
  checkBlitzClose(S33_3, X, eps); 
}

BOOST_AUTO_TEST_CASE( test_solveCGSympos_3x3 )
{
  blitz::Array<double,1> x(3);

  bob::math::linsolveCGSympos(A33_1, x, b3_1, 1e-6, 1000);
  checkBlitzClose(s3_1, x, eps);

  bob::math::linsolveCGSympos(A33_3, x, b3_1, 1e-6, 1000);
  checkBlitzClose(s3_3, x, eps);
}

BOOST_AUTO_TEST_SUITE_END()

