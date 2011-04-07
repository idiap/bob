/**
 * @file src/cxx/ip/test/dctfeatures.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the DCT feature extractor for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-DCTFeatures Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "core/logging.h"
#include "ip/DCTFeatures.h"
#include <vector>

struct T {
	blitz::Array<uint32_t,2> src;
	blitz::Array<double,1> dst1, dst2, dst3, dst4;
  std::vector<blitz::Array<double,1> > dst_mat;
  double eps;

	T(): src(6,8), dst1(6), dst2(6), dst3(6), dst4(6), dst_mat(0), eps(1e-3)
	{
		src = 0, 1, 2, 3, 4, 5, 6, 7,
			8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23,
			24, 25, 26, 27, 28, 29, 30, 31,
			32, 33, 34, 35, 36, 37, 38, 39,
			40, 41, 42, 43, 44, 45, 46, 47;

    dst1 = 32.9090, -3.8632, -22.6274, 0., 0., 0.;
    dst2 = 46.7654, -3.8632, -22.6274, 0., 0., 0.;
    dst3 = 116.0474, -3.8632, -22.6274, 0., 0., 0.;
    dst4 = 129.9038, -3.8632, -22.6274, 0., 0., 0.;

    dst_mat.push_back(dst1);
    dst_mat.push_back(dst2);
    dst_mat.push_back(dst3);
    dst_mat.push_back(dst4);
	}
	
	~T() {}
};

template<typename T>  
void checkBlitzClose( const blitz::Array<T,1>& t1, const blitz::Array<T,1>& t2,
  const double eps )
{
  BOOST_CHECK_EQUAL( t1.extent(0), t2.extent(0) );
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t1(i)-t2(i)), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_dct_feature_extract )
{
	std::vector<blitz::Array<double,1> > dst;
	Torch::ip::DCTFeatures dctfeatures( 3, 4, 0, 0, 6);

  dctfeatures( src, dst);
  // Iterate over the blocks and compare the vector of DCT coefficients with 
  // the one obtained using matlab
  int i=0;
  for( std::vector<blitz::Array<double,1> >::const_iterator it = dst.begin();
    it != dst.end(); ++it)
  {
    checkBlitzClose( *it, dst_mat[i], eps);
    ++i;
  }
}
  
BOOST_AUTO_TEST_SUITE_END()
