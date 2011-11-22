/**
 * @file cxx/ip/test/gammaCorrection.cc
 * @date Thu Mar 17 18:46:09 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the gamma correction function for 2D arrays/images
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
#define BOOST_TEST_MODULE IP-gammaCorrection Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/cast.h"
#include "ip/gammaCorrection.h"


struct T {
  blitz::Array<uint32_t,2> a2;
  blitz::Array<double,2> a2_g01, a2_g11;
  double eps;

  T(): a2(4,4), a2_g01(4,4), a2_g11(4,4), eps(2e-4)
  {
    a2 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15;

    a2_g01 = 0, 1, 1.0718, 1.1161, 1.1487, 1.1746, 1.1962, 1.2148, 
      1.2311, 1.2457, 1.2589, 1.2710, 1.2821, 1.2924, 1.3020, 1.3110;

    a2_g11 = 0, 1, 2.1435, 3.3484, 4.5948, 5.8731, 7.1774, 8.5037,
      9.8492, 11.2116, 12.5893, 13.9808, 15.3851, 16.8011, 18.2281, 19.6653;
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
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<U,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), Torch::core::cast<T>(t2(i,j)));
}

template<typename T>  
void checkBlitzSmall( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, double eps)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t1(i,j)-t2(i,j)), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_gammacorrection_2d_uint32 )
{
  blitz::Array<double,2> b2(a2.shape());
  
  // gamma == 0.1
  Torch::ip::gammaCorrection(a2, b2, 0.1);
  checkBlitzSmall(b2, a2_g01, eps); 

  // gamma == 1.1
  Torch::ip::gammaCorrection(a2, b2, 1.1);
  checkBlitzSmall(b2, a2_g11, eps); 
}
  
BOOST_AUTO_TEST_SUITE_END()
