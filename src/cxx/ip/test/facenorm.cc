/**
 * @file src/cxx/ip/test/facenorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the geometric normalization function for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-FaceNorm Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/cast.h"
#include "core/convert.h"
#include "ip/FaceEyesNorm.h"
#include "database/Array.h"


struct T {
  double eps;

	T(): eps(1e-5) {}

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

template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
	check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t2.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t1(i,j)-t2(i,j)), eps );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_facenorm )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_IP_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_IP_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  // Load original image
  boost::filesystem::path testdata_path_img( testdata_cpath);
  testdata_path_img /= "image_r10.pgm";
  Torch::database::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,2> img = ar_img.get<uint8_t,2>();
  blitz::Array<double,2> img_processed_d(40,40);
  

  Torch::ip::FaceEyesNorm facenorm(20,40,40,5/19.*40,20);

  // Process giving the coordinates of the eyes
  facenorm(img,img_processed_d,67,47,62,71);
  blitz::Array<uint8_t,2> img_processed = Torch::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10_facenorm.pgm";
  Torch::database::Array ar_img_facenorm(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_facenorm = ar_img_facenorm.get<uint8_t,2>();
  checkBlitzClose( img_ref_facenorm, img_processed, eps);
}

BOOST_AUTO_TEST_SUITE_END()
