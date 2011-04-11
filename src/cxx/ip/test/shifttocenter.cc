/**
 * @file src/cxx/ip/test/shiftToCenter.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the shift function for 2D and 3D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-ShiftToCenter Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/cast.h"
#include "ip/shiftToCenter.h"
#include "database/Array.h"

struct T {
	blitz::Array<uint32_t,2> a2, a2s_1, a2s_2, a2s_3, a2s_4, a2_centered;

	T(): a2(4,4), a2s_1(4,4), a2s_2(4,4), a2s_3(4,4), a2s_4(4,4), a2_centered(5,5)
	{
		a2 = 
			0, 1, 2, 3, 
			4, 5, 6, 7,
			8, 9, 10, 11, 
			12, 13, 14, 15;

		a2s_1 = 
			5, 6, 7, 7,
			9, 10, 11, 11,
			13, 14, 15, 15,
			13, 14, 15, 15;

		a2s_2 = 
			0, 0, 1, 2,
			0, 0, 1, 2, 
			4, 4, 5, 6,
			8, 8, 9, 10;

		a2s_3 = 
			0, 1, 2, 3, 
			0, 1, 2, 3, 
			4, 5, 6, 7,
			8, 9, 10, 11;

		a2s_4 = 
			0, 0, 1, 2,
			4, 4, 5, 6,
			8, 8, 9, 10, 
			12, 12, 13, 14;

		a2_centered = 
      0, 0, 0, 0, 0,
			0, 0, 1, 2, 3, 
			0, 4, 5, 6, 7,
			0, 8, 9, 10, 11, 
			0, 12, 13, 14, 15;
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

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
	check_dimensions( t1, t2);
	for( int i=0; i<t1.extent(0); ++i)
		for( int j=0; j<t1.extent(1); ++j)
			for( int k=0; k<t1.extent(2); ++k)
				BOOST_CHECK_EQUAL(t1(i,j,k), Torch::core::cast<T>(t2(i,j,k)));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_shift_down_right )
{
	blitz::Array<uint32_t,2> b2;
	// "No" shift +0 +0
	Torch::ip::shiftToCenter(a2, b2, 3, 3);

	checkBlitzEqual(a2s_1, b2); 
}

BOOST_AUTO_TEST_CASE( test_shift_up_left )
{
	blitz::Array<uint32_t,2> b2;
	Torch::ip::shiftToCenter(a2, b2, 1, 1);

	checkBlitzEqual(a2s_2, b2); 
}

BOOST_AUTO_TEST_CASE( test_shift_down )
{
	blitz::Array<uint32_t,2> b2;
	Torch::ip::shiftToCenter(a2, b2, 1, 2);

	checkBlitzEqual(a2s_3, b2); 
}

BOOST_AUTO_TEST_CASE( test_shift_right )
{
	blitz::Array<uint32_t,2> b2;
	Torch::ip::shiftToCenter(a2, b2, 2, 1);

	checkBlitzEqual(a2s_4, b2); 
}

BOOST_AUTO_TEST_CASE( test_shift_to_center_of_down )
{
	blitz::Array<uint32_t,2> b2;
	Torch::ip::shiftToCenterOfPoints(a2, b2, 0, 2, 2, 2);

	checkBlitzEqual(a2s_3, b2); 
}

BOOST_AUTO_TEST_CASE( test_shift_to_center_of_right )
{
	blitz::Array<uint32_t,2> b2;
	Torch::ip::shiftToCenterOfPoints(a2, b2, 2, 2, 2, 1);

	checkBlitzEqual(a2s_4, b2); 
}

BOOST_AUTO_TEST_SUITE_END()
