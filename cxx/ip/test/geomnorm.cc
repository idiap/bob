/**
 * @file cxx/ip/test/geomnorm.cc
 * @date Mon Apr 11 22:17:04 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the geometric normalization function for 2D arrays/images
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE IP-GeomNorm Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <boost/filesystem.hpp>
#include "core/cast.h"
#include "core/convert.h"
#include "core/logging.h"
#include "ip/generateWithCenter.h"
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
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
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
  blitz::Array<uint32_t,2> b2(bob::ip::getGenerateWithCenterShape(a2,1,1));
  BOOST_CHECK_EQUAL( b2.extent(0), a2_centered.extent(0));
  BOOST_CHECK_EQUAL( b2.extent(1), a2_centered.extent(1));
  bob::ip::generateWithCenter(a2, b2, 1, 1);
  checkBlitzEqual(a2_centered, b2); 
}

BOOST_AUTO_TEST_CASE( test_geomnorm )
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
  
  // Define a Geometric normalizer 
  // * rotation angle: 10 degrees
  // * scaling factor: 0.65
  // * Cropping area: 40x40
  // * No final cropping offset (i.e. used the provided upper left corner when calling 
  //   the operator() method)
  bob::ip::GeomNorm geomnorm(-10., 0.65, 40, 40, 0, 0);

  // Process giving the upper left corner as the rotation center (and the offset of the cropping area)
  geomnorm(img, img_processed_d, 54, 27);
  blitz::Array<uint8_t,2> img_processed = bob::core::convertFromRange<uint8_t>( img_processed_d, 0., 255.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10_geomnorm.pgm";
//  bob::io::Array(img_processed).save(testdata_path_img.string()); // Re-generate reference data
  bob::io::Array ar_img_geomnorm(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_geomnorm = ar_img_geomnorm.get<uint8_t,2>();
  checkBlitzClose( img_ref_geomnorm, img_processed, eps);
}

BOOST_AUTO_TEST_CASE( test_geomnorm_with_mask )
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
  boost::filesystem::path testdata_path(testdata_cpath);
  testdata_path /= "image_r70.pgm";
  blitz::Array<uint8_t,2> input_image = bob::io::Array(testdata_path.string()).get<uint8_t,2>();
  blitz::Array<double,2> output_image(160,160);
  
  blitz::Array<bool,2> input_mask(input_image.shape()[0],input_image.shape()[1]);
  // estimate the input mask from the black pixels of the input image
  input_mask = input_image != 0;
  blitz::Array<bool,2> output_mask(output_image.shape()[0], output_image.shape()[1]);
  
  // Define a Geometric normalizer 
  // * rotation angle: 70 degrees
  // * scaling factor: 1.2
  // * Cropping area: 160x160
  // * Final cropping offset: 80x80
  bob::ip::GeomNorm geomnorm(-70., 1.2, 160, 160, 80, 80);

  // Process giving the masks and the center of the eye positions
  geomnorm(input_image, input_mask, output_image, output_mask, 64, 69);
  
  // check that the image is close to the reference image
  blitz::Array<uint8_t,2> output_image_uint8 = bob::core::convertFromRange<uint8_t>(output_image, 0., 255.);
  testdata_path = testdata_cpath;
  testdata_path /= "image_r70_geomnorm.pgm";
//  bob::io::Array(output_image_uint8).save(testdata_path.string()); // Re-generate reference data
  blitz::Array<uint8_t,2> output_reference = bob::io::Array(testdata_path.string()).get<uint8_t,2>();
  checkBlitzClose( output_image_uint8, output_reference, eps);
  
  // check that the mask is identical to the reference mask
  blitz::Array<uint8_t,2> output_mask_uint8 = bob::core::convertFromRange<uint8_t>(output_mask, false, true);
  testdata_path = testdata_cpath;
  testdata_path /= "image_r70_mask.pgm";
//  bob::io::Array(output_mask_uint8).save(testdata_path.string()); // Re-generate reference data
  output_reference = bob::io::Array(testdata_path.string()).get<uint8_t,2>();
  checkBlitzEqual(output_mask_uint8, output_reference);
}


BOOST_AUTO_TEST_CASE( test_geomnorm_with_points )
{
  // generate geometric normalizer that rotates by 45 degrees, scales by 2 and moves to new center (40,80)
  // (the image resolution aka cropping area 160x160 is not required for this test...)
  bob::ip::GeomNorm geom_norm(45., 2., 160, 160, 40, 80);

  // define positions to be rotated
  blitz::TinyVector<double,2> position(15,25);

  // we take an offset of 20,20 to rotate the point
  // the centered point is hence (-5,5)
  blitz::TinyVector<double,2> rotated = geom_norm(position,20,20);

  // check the new position
  // new y-value should be 0 plus offset
  BOOST_CHECK_CLOSE(rotated(0), 40., 1e-10);
  // new x value is the length of the centered vector (i.e. 5*sqrt(2)) times the scaling factor 2 plus offset
  BOOST_CHECK_CLOSE(rotated(1), 80. + 5. * std::sqrt(2.) * 2, 1e-10);

}

BOOST_AUTO_TEST_SUITE_END()
