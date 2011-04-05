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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_zigzag )
{
	blitz::Array<uint32_t,1> dst;

	std::cout << "SRC: " << src << std::endl;

	Torch::ip::zigzagt(src, dst, 3);
	std::cout << "3 dct kept: " << dst << std::endl;
	std::cout << "3 dct kept: " << dst3 << std::endl;

	Torch::ip::zigzagt(src, dst, 6);
	std::cout << "6 dct kept: " << dst << std::endl;
	std::cout << "6 dct kept: " << dst6 << std::endl;

	Torch::ip::zigzagt(src, dst, 10);
	std::cout << "10 dct kept: " << dst << std::endl;
	std::cout << "10 dct kept: " << dst10 << std::endl;
}
  
BOOST_AUTO_TEST_SUITE_END()
