/**
 * @file core/cxx/test/reshape.cc
 * @date Sun Jul 17 13:31:35 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Tests the reshape function for 1D and 2D blitz arrays
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
#define BOOST_TEST_MODULE Core-reshape Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <bob/core/array_reshape.h>

struct T {
  blitz::Array<uint32_t,2> a23, a32_s, a16_s;
  blitz::Array<uint32_t,1> b6;
  blitz::Array<uint32_t,2> c;
  T(): a23(2,3), a32_s(3,2), a16_s(1,6), b6(6) {
    a23 = 1, 2, 3, 4, 5, 6;
    a32_s = 1, 5, 4, 3, 2, 6;
    a16_s = 1, 4, 2, 5, 3, 6;
    b6 = 1, 4, 2, 5, 3, 6;
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
void checkBlitzEqual( blitz::Array<T,1>& t1, blitz::Array<T,1>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_EQUAL( t1(i), t2(i) );
}

template<typename T>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_reshape_2d_2d )
{
  blitz::Array<uint32_t,2> a32(3, 2);
  bob::core::array::reshape(a23, a32);
  checkBlitzEqual(a32, a32_s);
 
  blitz::Array<uint32_t,2> a16(1, 6);
  bob::core::array::reshape(a23, a16);
  checkBlitzEqual(a16, a16_s);
}

BOOST_AUTO_TEST_CASE( test_reshape_2d_1d )
{
  blitz::Array<uint32_t,1> c6(6);
  bob::core::array::reshape(a23, c6);
  checkBlitzEqual(c6, b6);
}

BOOST_AUTO_TEST_CASE( test_reshape )
{
  blitz::Array<uint32_t,2> c23(2,3);
  bob::core::array::reshape(b6, c23);
  checkBlitzEqual(c23, a23);
}

BOOST_AUTO_TEST_SUITE_END()
