/**
 * @file math/cxx/test/sqrtm.cc
 * @date Fri Jan 27 17:37:41 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the computation of the square root of a matrix
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
#define BOOST_TEST_MODULE math-sqrtm Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "bob/math/sqrtm.h"
#include "bob/math/linear.h"

struct T {
  blitz::Array<double,2> A44_1, S44_1;
  double eps;

  T(): A44_1(4,4), S44_1(4,4),
    eps(1e-6)
  {
    A44_1 = 1, -1, 0, 0,  -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 1;
    S44_1 =  0.81549316, -0.54489511, -0.16221167, -0.10838638,
            -0.54489510,  1.19817659, -0.49106981, -0.16221167,
            -0.16221167, -0.49106981,  1.19817659, -0.54489511,
            -0.10838638, -0.16221167, -0.54489511,  0.81549316;
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
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs( t2(i,j)-t1(i,j) ), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_sqrtm_4x4 )
{
  blitz::Array<double,2> S(4,4);

  // Computes the square root
  bob::math::sqrtSymReal(A44_1,S);

  // Compare to reference
  checkBlitzClose(S44_1, S, eps);

  // Check that this is 'really' the square root
  blitz::Array<double,2> SS(4,4);
  bob::math::prod(S, S, SS);
  checkBlitzClose(A44_1, SS, eps);
}

BOOST_AUTO_TEST_SUITE_END()

