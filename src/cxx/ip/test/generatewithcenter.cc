/**
 * @file cxx/ip/test/generatewithcenter.cc
 * @date Sun Apr 17 23:11:51 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the shift function for 2D and 3D arrays/images
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
#define BOOST_TEST_MODULE IP-generatewithcenter Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/cast.h"
#include "ip/generateWithCenter.h"
#include "io/Array.h"

struct T {
	blitz::Array<uint32_t,2> a2, a2g_11, a2g_12;
	blitz::Array<bool,2> a2m, a2m_11, a2m_12;

	T(): a2(4,4), a2g_11(5,5), a2g_12(5,5),
       a2m(4,4), a2m_11(5,5), a2m_12(5,5)
	{
		a2 = 
			0, 1, 2, 3, 
			4, 5, 6, 7,
			8, 9, 10, 11, 
			12, 13, 14, 15;

		a2g_11 = 
      0, 0, 0, 0, 0,
      0, 0, 1, 2, 3,
      0, 4, 5, 6, 7,
      0, 8, 9, 10, 11,
      0, 12, 13, 14, 15;

		a2g_12 = 
      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      4, 5, 6, 7, 0, 
      8, 9, 10, 11, 0, 
      12, 13, 14, 15, 0;
  
    a2m = true;
    
    a2m_11 = false, false, false, false, false,
             false, true, true, true, true,
             false, true, true, true, true,
             false, true, true, true, true,
             false, true, true, true, true;

    a2m_12 = false, false, false, false, false,
             true, true, true, true, false,
             true, true, true, true, false,
             true, true, true, true, false,
             true, true, true, true, false;
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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_generateWithCenter )
{
	blitz::Array<uint32_t,2> b2(bob::ip::getGenerateWithCenterShape(a2,1,1));
  bob::ip::generateWithCenter(a2,b2,1,1);
  checkBlitzEqual(a2g_11, b2);

	b2.resize(bob::ip::getGenerateWithCenterShape(a2,1,2));
  bob::ip::generateWithCenter(a2,b2,1,2);
  checkBlitzEqual(a2g_12, b2);
}

BOOST_AUTO_TEST_CASE( test_generateWithCenter_mask )
{
	blitz::Array<uint32_t,2> b2(bob::ip::getGenerateWithCenterShape(a2,1,1));
  blitz::Array<bool,2> b2_mask(b2.shape());
  bob::ip::generateWithCenter(a2, a2m, b2, b2_mask, 1, 1);
  checkBlitzEqual(a2g_11, b2);
  checkBlitzEqual(a2m_11, b2_mask);

	b2.resize(bob::ip::getGenerateWithCenterShape(a2, 1, 2));
  bob::ip::generateWithCenter(a2, a2m, b2, b2_mask, 1, 2);
  checkBlitzEqual(a2g_12, b2);
  checkBlitzEqual(a2m_12, b2_mask);
}

BOOST_AUTO_TEST_SUITE_END()
