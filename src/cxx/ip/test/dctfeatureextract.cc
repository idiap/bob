/**
 * @file src/cxx/ip/test/block.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the block decomposition function for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-Block Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <vector>
#include "core/logging.h"
#include "ip/DCTfeatureExtract.h"

struct T {
	blitz::Array<uint32_t,2> src;

	T(): src(6,8)
	{
		src = 0, 1, 2, 3, 4, 5, 6, 7,
			8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23,
			24, 25, 26, 27, 28, 29, 30, 31,
			32, 33, 34, 35, 36, 37, 38, 39,
			40, 41, 42, 43, 44, 45, 46, 47;
	}
	
	~T() {}
};

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_dct_feature_extract )
{
	std::vector<blitz::Array<double, 2> > dst;
	Torch::ip::dctFeatureExtract(src, dst, 3, 3, 0, 0);
}
  
BOOST_AUTO_TEST_SUITE_END()
