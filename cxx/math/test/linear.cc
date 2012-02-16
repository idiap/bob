/**
 * @file cxx/math/test/linear.cc
 * @date Fri Apr 8 20:57:37 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test basic linear algebra operations
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
#define BOOST_TEST_MODULE math-linear Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "math/linear.h"


struct T {
  blitz::Array<double,2> A_24, A_43, A_23, Asol_44, Asol_eye_44, Asol_diag_44;
  blitz::Array<double,1> b_4, b_2, b_5a, b_5b, n_b_4;
  double b5_dot, tr_Asol_44, ned_b_4;
  double eps;

  T(): A_24(2,4), A_43(4,3), A_23(2,3), Asol_44(4,4), Asol_eye_44(4,4), Asol_diag_44(4,4),
       b_4(4), b_2(2), b_5a(5), b_5b(5), n_b_4(4), 
       b5_dot(99.), ned_b_4(5.4772), eps(1e-3)
  {
    A_24 = 1., 2., 3., 4., 5., 6., 7., 8.;
    A_43 = 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.;
    A_23 = 60., 50., 40., 180., 154., 128.;

    b_4 = 4., 3., 2., 1.;
    b_2 = 20., 60.;

    b_5a = 3., 2., 1., 2., 3.;
    b_5b = 7., 8., 9., 10., 11.;
    n_b_4 = 0.7303, 0.5477, 0.3651, 0.1826;

    Asol_44 = 16., 12., 8., 4., 
              12.,  9., 6., 3.,
               8.,  6., 4., 2.,
               4.,  3., 2., 1.;

    tr_Asol_44 = 30.;

    Asol_eye_44 = 1., 0., 0., 0.,
                  0., 1., 0., 0.,
                  0., 0., 1., 0.,
                  0., 0., 0., 1.; 

    Asol_diag_44 = 4., 0., 0., 0.,
                   0., 3., 0., 0.,
                   0., 0., 2., 0.,
                   0., 0., 0., 1.; 
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

BOOST_AUTO_TEST_CASE( test_matrix_matrix_prod )
{
  blitz::Array<double,2> sol(2,3);

  bob::math::prod( A_24, A_43, sol);
  checkBlitzClose( A_23, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_matrix_vector_prod )
{
  blitz::Array<double,1> sol(2);

  bob::math::prod( A_24, b_4, sol);
  checkBlitzClose( b_2, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_vector_matrix_prod )
{
  blitz::Array<double,1> sol(2);

  bob::math::prod( b_4, A_24.transpose(1,0), sol);
  checkBlitzClose( b_2, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_vector_vector_prod )
{
  blitz::Array<double,2> sol(4,4);

  bob::math::prod( b_4, b_4, sol);
  checkBlitzClose( Asol_44, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_vector_vector_dot )
{
  double sol = bob::math::dot( b_5a, b_5b);
  BOOST_CHECK_SMALL( fabs(b5_dot - sol), eps);
}

BOOST_AUTO_TEST_CASE( test_matrix_trace )
{
  double sol = bob::math::trace( Asol_44);
  BOOST_CHECK_SMALL( fabs(tr_Asol_44 - sol), eps);
}

BOOST_AUTO_TEST_CASE( test_vector_norm )
{
  double sol = bob::math::norm(b_4);
  BOOST_CHECK_SMALL( fabs(ned_b_4 - sol), eps);
}

BOOST_AUTO_TEST_CASE( test_vector_normalized )
{
  blitz::Array<double,1> sol(4);
  bob::math::normalize(b_4, sol);
  checkBlitzClose( n_b_4, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_eye )
{
  blitz::Array<double,2> sol(4,4);
  bob::math::eye(sol);
  checkBlitzClose( Asol_eye_44, sol, eps);
}

BOOST_AUTO_TEST_CASE( test_diag )
{
  blitz::Array<double,2> sol(4,4);
  bob::math::diag(b_4, sol);
  checkBlitzClose( Asol_diag_44, sol, eps);
}

BOOST_AUTO_TEST_SUITE_END()
