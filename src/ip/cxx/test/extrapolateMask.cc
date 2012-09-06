/**
 * @file ip/cxx/test/extrapolateMask.cc
 * @date Mon May 9 19:54:44 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the extrapolate functions
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
#define BOOST_TEST_MODULE IP-extrapolateMask Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "bob/core/cast.h"
#include "bob/ip/extrapolateMask.h"


struct T {
  blitz::Array<bool,2> a2_5_1, a2_5_2, a2_5_3, a2_5_4;
  blitz::Array<int32_t,2> i2_5_1, i2_5_2, i2_5_3, i2_5_4;
  blitz::Array<int32_t,2> s2_5_1, s2_5_2, s2_5_3, s2_5_4;

  T(): a2_5_1(5,5), a2_5_2(5,5), a2_5_3(5,5), a2_5_4(5,5), 
       i2_5_1(5,5), i2_5_2(5,5), i2_5_3(5,5), i2_5_4(5,5), 
       s2_5_1(5,5), s2_5_2(5,5), s2_5_3(5,5), s2_5_4(5,5)
  {
    a2_5_1 =  true, true, true, true, true,
              true, true, true, true, false,
              true, true, true, false, false,
              true, true, false, false, false,
              true, false, false, false, false;

    i2_5_1 =   0,  1,  2,  3,  4,
               5,  6,  7,  8,  9, 
              10, 11, 12, 13, 14,
              15, 16, 17, 18, 19,
              20, 21, 22, 23, 24;

    s2_5_1 =   0,  1,  2,  3,  4,
               5,  6,  7,  8,  4, 
              10, 11, 12,  8,  4,
              15, 16, 12,  8,  4,
              20, 16, 12,  8,  4;

    a2_5_2 =  false, false, true, false, false,
              false, true, true, true, false,
              true, true, true, true, true,
              false, true, true, true, false,
              false, false, true, false, false;

    i2_5_2 =   0,  1,  2,  3,  4,
               5,  6,  7,  8,  9, 
              10, 11, 12, 13, 14,
              15, 16, 17, 18, 19,
              20, 21, 22, 23, 24;
    
    s2_5_2 =  10,  6,  2,  8, 14,
              10,  6,  7,  8, 14, 
              10, 11, 12, 13, 14,
              10, 16, 17, 18, 14,
              10, 16, 22, 18, 14; 

    a2_5_3 =  true, true, true, true, true,
              false, true, true, false, false,
              false, true, true, false, false,
              false, true, true, false, false,
              false, true, true, false, false;

    i2_5_3 =   0,  1,  2,  3,  4,
               5,  6,  7,  8,  9, 
              10, 11, 12, 13, 14,
              15, 16, 17, 18, 19,
              20, 21, 22, 23, 24;

    s2_5_3 =   0,  1,  2,  3,  4,
               0,  6,  7,  3,  4, 
               0, 11, 12,  3,  4,
               0, 16, 17,  3,  4,
               0, 21, 22,  3,  4; 
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

template<typename T, int N>  
void checkTinyEqual( blitz::TinyVector<T,N>& t1, blitz::TinyVector<T,N>& t2)
{
  for( int i=0; i<N; ++i)
    BOOST_CHECK_EQUAL(t1(i), t2(i));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_extrapolate )
{
  bob::ip::extrapolateMask( a2_5_1, i2_5_1 );
  checkBlitzEqual( i2_5_1, s2_5_1);

  bob::ip::extrapolateMask( a2_5_2, i2_5_2 );
  checkBlitzEqual( i2_5_2, s2_5_2);

  bob::ip::extrapolateMask( a2_5_3, i2_5_3 );
  checkBlitzEqual( i2_5_3, s2_5_3);
}
  
BOOST_AUTO_TEST_SUITE_END()
