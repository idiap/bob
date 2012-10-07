/**
 * @file ip/cxx/test/geomnorm.cc
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
#include "bob/core/cast.h"
#include "bob/core/convert.h"
#include "bob/core/logging.h"
#include "bob/ip/GeomNorm.h"
#include "bob/io/utils.h"


struct T {
  double eps;

  T(): eps(1e-5)
  {
  }

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions( blitz::Array<T,d> t1, blitz::Array<U,d> t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,2> t1, blitz::Array<U,2> t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,2> t1, blitz::Array<T,2> t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t2.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t1(i,j)-t2(i,j)), eps );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_geomnorm )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  // Load original image
  boost::filesystem::path testdata_path_img( testdata_cpath);
  testdata_path_img /= "image_r10.pgm";
  blitz::Array<uint8_t,2> img = bob::io::load<uint8_t,2>(testdata_path_img.string());
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
//  bob::io::save(testdata_path_img.string(), img_processed); // Re-generate reference data
  blitz::Array<uint8_t,2> img_ref_geomnorm = bob::io::load<uint8_t,2>(testdata_path_img.string());
  checkBlitzClose( img_ref_geomnorm, img_processed, eps);
}

BOOST_AUTO_TEST_CASE( test_geomnorm_with_mask )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    bob::core::error << "Environment variable $BOB_TESTDATA_DIR " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw bob::core::Exception();
  }
  // Load original image
  boost::filesystem::path testdata_path(testdata_cpath);
  testdata_path /= "image_r70.pgm";
  blitz::Array<uint8_t,2> input_image = bob::io::load<uint8_t,2>(testdata_path.string());
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
//  bob::io::open(testdata_path.string(), 'w')->write(output_image_uint8); // Re-generate reference data
  checkBlitzClose( output_image_uint8, bob::io::load<uint8_t,2>(testdata_path.string()), eps);
  
  // check that the mask is identical to the reference mask
  blitz::Array<uint8_t,2> output_mask_uint8 = bob::core::convertFromRange<uint8_t>(output_mask, false, true);
  testdata_path = testdata_cpath;
  testdata_path /= "image_r70_mask.pgm";
//  bob::io::open(testdata_path.string(), 'w')->write(output_mask_uint8); // Re-generate reference data
  blitz::Array<uint8_t,2> output_reference = bob::io::load<uint8_t,2>(testdata_path.string());
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
