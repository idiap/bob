/**
 * @file src/cxx/ip/test/block.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the block decomposition function for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-ZigZag Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <vector>
#include "core/logging.h"
#include "ip/zigzag.h"

struct T {
	blitz::Array<uint32_t,2> src;
	blitz::Array<uint32_t,1> dst3, dst6, dst10;

	T(): src(6,6), dst3(3), dst6(6), dst10(10)
	{
		src = 0, 1, 2, 3, 4, 5, 
			6, 7, 8, 9, 10, 11, 
			12, 13, 14, 15, 16, 17, 
			18, 19, 20, 21, 22, 23, 
			24, 25, 26, 27, 28, 29,
			30, 31, 32, 33, 34, 35;

		dst3 = 0, 1, 6;
		
		dst6 = 0, 1, 6, 12, 7, 2;
		
		dst10 = 0, 1, 6, 12, 7, 2, 3, 8, 13, 18;
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
      BOOST_CHECK_EQUAL(t1(i), Torch::core::cast<T>(t2(i)));
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

BOOST_AUTO_TEST_CASE( test_zigzag )
{
	blitz::Array<uint32_t,1> dst;

	Torch::ip::zigzag(src, dst, 3);
	checkBlitzEqual(dst, dst3);

	Torch::ip::zigzag(src, dst, 6);
	checkBlitzEqual(dst, dst6);

	Torch::ip::zigzag(src, dst, 10);
	checkBlitzEqual(dst, dst10);
}
  
BOOST_AUTO_TEST_SUITE_END()
