/**
 * @file cxx/ip/test/flipflop.cc
 * @date Mon Mar 14 16:31:07 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the flip/flop functions for 2D and 3D arrays/images
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
#define BOOST_TEST_MODULE IP-flipflop Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "core/logging.h"
#include "ip/flipflop.h"


struct T {
  blitz::Array<uint32_t,2> a2, a2_flip, a2_flop;
  blitz::Array<uint32_t,3> a3, a3_flip, a3_flop;

  T(): a2(4,4), a2_flip(4,4), a2_flop(4,4),
       a3(3,4,4), a3_flip(3,4,4), a3_flop(3,4,4)
  {
    a2 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15;

    a2_flip = 12, 13, 14, 15, 8, 9, 10, 11,
              4, 5, 6, 7, 0, 1, 2, 3;

    a2_flop = 3, 2, 1, 0, 7, 6, 5, 4, 
              11, 10, 9, 8, 15, 14, 13, 12;

    a3 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47;

    a3_flip = 12, 13, 14, 15, 8, 9, 10, 11,
              4, 5, 6, 7, 0, 1, 2, 3,
              28, 29, 30, 31, 24, 25, 26, 27,
              20, 21, 22, 23, 16, 17, 18, 19,
              44, 45, 46, 47, 40, 41, 42, 43,
              36, 37, 38, 39, 32, 33, 34, 35;

    a3_flop = 3, 2, 1, 0, 7, 6, 5, 4,
              11, 10, 9, 8, 15, 14, 13, 12,
              19, 18, 17, 16, 23, 22, 21, 20,
              27, 26, 25, 24, 31, 30, 29, 28,
              35, 34, 33, 32, 39, 38, 37, 36,
              43, 42, 41, 40, 47, 46, 45, 44;
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

BOOST_AUTO_TEST_CASE( test_flipflop_2d_uint8 )
{
  blitz::Array<uint32_t,2> b2(a2.shape());
  // Flip
  bob::ip::flip(a2, b2);
  checkBlitzEqual(a2_flip, b2); 

  // Flop
  bob::ip::flop(a2, b2);
  checkBlitzEqual(a2_flop, b2); 
}
  
BOOST_AUTO_TEST_CASE( test_flipflop_3d_uint8 )
{
  blitz::Array<uint32_t,3> b3(a3.shape());
  // Flip
  bob::ip::flip(a3, b3);
  checkBlitzEqual(a3_flip, b3); 

  // Flop
  bob::ip::flop(a3, b3);
  checkBlitzEqual(a3_flop, b3);
}

BOOST_AUTO_TEST_SUITE_END()
