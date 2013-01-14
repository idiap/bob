/**
 * @file ip/cxx/test/LBP.cc
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the LBP functions for 2D arrays/images
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
#define BOOST_TEST_MODULE IP-Lbp Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "bob/ip/LBP.h"
#include "bob/ip/LBP4R.h"
#include "bob/ip/LBP8R.h"

#include <iostream>

struct T {
  blitz::Array<uint8_t,2> a1, a2;
  uint16_t b1, b2;
  uint16_t c1, c2;

  T(): a1(3,3), a2(3,3)
  {
    a1 = 0, 1, 2,
         3, 4, 5,
         6, 7, 8;

    a2 = 8, 7, 6,
         5, 4, 3,
         2, 1, 0;

    b1 = 6;
    b2 = 9;

    c1 = 30;
    c2 = 225;
  }

  ~T() {}
};

template<typename T, int d>  
void check_dimensions( blitz::Array<T,d>& t1, blitz::Array<T,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

template<typename T>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<T,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), t2(i,j,k));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_lbp41_2d_uint8 )
{
  // LBP 4,1
  bob::ip::LBP4R lbp;

  BOOST_CHECK_EQUAL( b1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( b2, lbp(a2,1,1) );

  bob::ip::LBP4R lbpcirc(1.,true);
  BOOST_CHECK_EQUAL( b1, lbpcirc(a1,1,1) );
  BOOST_CHECK_EQUAL( b2, lbpcirc(a2,1,1) );
}

BOOST_AUTO_TEST_CASE( test_lbp81_2d_uint8 )
{
  // LBP 8,1
  bob::ip::LBP8R lbp;

  BOOST_CHECK_EQUAL( c1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( c2, lbp(a2,1,1) );

  bob::ip::LBP8R lbpcirc(1.,true);
  BOOST_CHECK_EQUAL( c1, lbpcirc(a1,1,1) );
  BOOST_CHECK_EQUAL( c2, lbpcirc(a2,1,1) );
}
  
BOOST_AUTO_TEST_SUITE_END()
