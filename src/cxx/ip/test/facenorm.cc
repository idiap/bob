/**
 * @file cxx/ip/test/facenorm.cc
 * @date Thu Apr 14 21:03:45 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the geometric normalization function for 2D arrays/images
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE IP-FaceNorm Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/logging.h"
#include "core/cast.h"
#include "core/convert.h"
#include "ip/color.h"
#include "ip/FaceEyesNorm.h"
#include "io/Array.h"

#include <iostream>

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
  char *testdata_cpath = getenv("BOB_IP_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_IP_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  // Load original image
  boost::filesystem::path testdata_path_img( testdata_cpath);
  testdata_path_img /= "image_r10.pgm";
  bob::io::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,2> img = ar_img.get<uint8_t,2>();
  blitz::Array<double,2> img_processed_d(40,40);
  

  bob::ip::FaceEyesNorm facenorm(20,40,40,5/19.*40,20);

  // Process giving the coordinates of the eyes
  facenorm(img,img_processed_d,67,47,62,71);
  blitz::Array<uint8_t,2> img_processed = bob::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10_facenorm.pgm";
  bob::io::Array ar_img_facenorm(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_facenorm = ar_img_facenorm.get<uint8_t,2>();
  checkBlitzClose( img_ref_facenorm, img_processed, eps);
}

BOOST_AUTO_TEST_CASE( test_facenorm2 )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("BOB_IP_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_IP_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  // Load original image
  boost::filesystem::path testdata_path_img( testdata_cpath);
  testdata_path_img /= "9003_m_wm_s01_9004_en_1.jpg";
  bob::io::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,3> img_rgb = ar_img.get<uint8_t,3>();
  blitz::Array<uint8_t,2> img(img_rgb.extent(1),img_rgb.extent(2));
  bob::ip::rgb_to_gray(img_rgb, img);
  blitz::Array<double,2> img_processed_d(80,64);
  

  bob::ip::FaceEyesNorm facenorm(33,80,64,16,32);

  // Process giving the coordinates of the eyes
  facenorm(img,img_processed_d,276,276,276,375);
  blitz::Array<uint8_t,2> img_processed = bob::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "9003_m_wm_s01_9004_en_1_facenorm.pgm";
  bob::io::Array ar_img_facenorm(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_facenorm = ar_img_facenorm.get<uint8_t,2>();
  checkBlitzClose( img_ref_facenorm, img_processed, eps);
}
 
  
BOOST_AUTO_TEST_SUITE_END()
