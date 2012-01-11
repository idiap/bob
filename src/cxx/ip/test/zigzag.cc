/**
 * @file cxx/ip/test/zigzag.cc
 * @date Tue Apr 5 16:55:23 2011 +0200
 * @author Niklas Johansson <niklas.johansson@idiap.ch>
 *
 * @brief Test the zigzag decomposition function for 2D arrays/images,
 *   which is used by the DCT features
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
#define BOOST_TEST_MODULE IP-zigzag Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include "core/cast.h"
#include "ip/zigzag.h"

struct T {
	blitz::Array<uint32_t,2> src_a, src_b;
	blitz::Array<uint32_t,1>  dst_a3, dst_a6, dst_a10, dst_a21, dst_a26, dst_a36,
                            dst_b2, dst_b3, dst_b6, dst_b8, dst_b8r;

	T(): src_a(6,6), src_b(4,2),
     dst_a3(3), dst_a6(6), dst_a10(10), dst_a21(21), dst_a26(26), dst_a36(36),
     dst_b2(2), dst_b3(3), dst_b6(6), dst_b8(8), dst_b8r(8)
	{
		src_a =  0,  1,  2,  3,  4,  5, 
      			 6,  7,  8,  9, 10, 11, 
			      12, 13, 14, 15, 16, 17, 
      			18, 19, 20, 21, 22, 23, 
			      24, 25, 26, 27, 28, 29,
      			30, 31, 32, 33, 34, 35;

		dst_a3 = 0, 1, 6;
		dst_a6 = 0, 1, 6, 12, 7, 2;
		dst_a10 = 0, 1, 6, 12, 7, 2, 3, 8, 13, 18;
		dst_a21 = 0, 1, 6, 12, 7, 2, 3, 8, 13, 18, 24, 19, 14, 9, 4, 5, 10, 15, 
              20, 25, 30;
		dst_a26 = 0, 1, 6, 12, 7, 2, 3, 8, 13, 18, 24, 19, 14, 9, 4, 5, 10, 15, 
              20, 25, 30, 31, 26, 21, 16, 11;
		dst_a36 = 0, 1, 6, 12, 7, 2, 3, 8, 13, 18, 24, 19, 14, 9, 4, 5, 10, 15, 
              20, 25, 30, 31, 26, 21, 16, 11, 17, 22, 27, 32, 33, 28, 23, 29, 
              34, 35;


    src_b = 0, 1, 
            2, 3, 
            4, 5,
            6, 7;

    dst_b2 = 0, 1;
    dst_b3 = 0, 1, 2;
    dst_b6 = 0, 1, 2, 4, 3, 5;
    dst_b8 = 0, 1, 2, 4, 3, 5, 6, 7;
    dst_b8r = 0, 2, 1, 3, 4, 6, 5, 7;
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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_zigzag_input1 )
{
	blitz::Array<uint32_t,1> dst;

  // Process src_a for various number of DCT coefficients
  dst.resize(3);
	bob::ip::zigzag(src_a, dst, 3);
	checkBlitzEqual(dst, dst_a3);

  dst.resize(6);
	bob::ip::zigzag(src_a, dst, 6);
	checkBlitzEqual(dst, dst_a6);

  dst.resize(10);
	bob::ip::zigzag(src_a, dst, 10);
	checkBlitzEqual(dst, dst_a10);

  dst.resize(21);
	bob::ip::zigzag(src_a, dst, 21);
	checkBlitzEqual(dst, dst_a21);

  dst.resize(26);
	bob::ip::zigzag(src_a, dst, 26);
	checkBlitzEqual(dst, dst_a26);
 
  dst.resize(36);
	bob::ip::zigzag(src_a, dst, 36);
	checkBlitzEqual(dst, dst_a36);
}

BOOST_AUTO_TEST_CASE( test_zigzag_input2 )
{
	blitz::Array<uint32_t,1> dst;

  // Process src_b for various number of DCT coefficients
  dst.resize(2);
	bob::ip::zigzag(src_b, dst, 2);
	checkBlitzEqual(dst, dst_b2);

  dst.resize(3);
	bob::ip::zigzag(src_b, dst, 3);
	checkBlitzEqual(dst, dst_b3);

  dst.resize(6);
	bob::ip::zigzag(src_b, dst, 6);
	checkBlitzEqual(dst, dst_b6);

  dst.resize(8);
	bob::ip::zigzag(src_b, dst, 8);
	checkBlitzEqual(dst, dst_b8);

  // Process fully
	bob::ip::zigzag(src_b, dst);
	checkBlitzEqual(dst, dst_b8);

  // Reverse order
	bob::ip::zigzag(src_b, dst, 8, true);
	checkBlitzEqual(dst, dst_b8r);
}
  
BOOST_AUTO_TEST_SUITE_END()
