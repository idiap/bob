/**
 * @file src/cxx/ip/test/geomnorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the geometric normalization function for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-GeomNorm Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/cast.h"
#include "core/convert.h"
#include "ip/GeomNorm.h"
#include "io/Array.h"


struct T {
	blitz::Array<uint32_t,2> a2, a2_centered;
  double eps;

	T(): a2(4,4), a2_centered(5,5), eps(1e-5)
	{
		a2 = 
			0, 1, 2, 3, 
			4, 5, 6, 7,
			8, 9, 10, 11, 
			12, 13, 14, 15;

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

BOOST_AUTO_TEST_CASE( test_generateWithCenter )
{
	blitz::Array<uint32_t,2> b2(Torch::ip::getGenerateWithCenterShape(a2,1,1));
  BOOST_CHECK_EQUAL( b2.extent(0), a2_centered.extent(0));
  BOOST_CHECK_EQUAL( b2.extent(1), a2_centered.extent(1));
	Torch::ip::generateWithCenter(a2, b2, 1, 1);
	checkBlitzEqual(a2_centered, b2); 
}

BOOST_AUTO_TEST_CASE( test_geomnorm )
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
  Torch::io::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,2> img = ar_img.get<uint8_t,2>();
  blitz::Array<double,2> img_processed_d(40,40);
  

  Torch::ip::GeomNorm geomnorm(20,40,40,0,0);

  // Process giving the coordinates of the eyes
  geomnorm(img,img_processed_d,67,47,62,71);
  blitz::Array<uint8_t,2> img_processed = Torch::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10_geomnorm.pgm";
  Torch::io::Array ar_img_geomnorm(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_geomnorm = ar_img_geomnorm.get<uint8_t,2>();
  checkBlitzClose( img_ref_geomnorm, img_processed, eps);

  Torch::ip::GeomNorm geomnorm2(20,40,40,0,0);
  img_processed_d.resize(40,40);
  geomnorm2(img,img_processed_d,67,47,62,71);
  blitz::Array<uint8_t,2> img_processed2 = Torch::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10_geomnorm_BORDER25x0.pgm";
  Torch::io::Array ar_img_geomnorm2(img_processed2);
  ar_img_geomnorm2.save(testdata_path_img.string());
}

BOOST_AUTO_TEST_SUITE_END()
