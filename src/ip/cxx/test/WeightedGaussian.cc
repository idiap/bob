/**
 * @file ip/cxx/test/WeightedGaussian.cc
 * @date Fri Jul 20 14:54:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Weighted Gaussian smoothing on 2D images
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE IP-WeightedGaussian Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "bob/core/logging.h"
#include "bob/ip/WeightedGaussian.h"

#include "bob/io/utils.h"
#include <algorithm>

#include <random/discrete-uniform.h>
#include <random/uniform.h>

#include <boost/filesystem.hpp>

struct T {
  double eps;
  blitz::Array<uint8_t,2> a;
  blitz::Array<double,2> a_ref;

  T(): eps(1e-4), a(3,4), a_ref(3,4)
  {
    a = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    a_ref = 1.21194, 2, 3, 3.78806, 3.79444, 7.45636, 
            8.45636, 9.20556, 9.21194, 10, 11, 11.7881;
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

template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t2.extent(1); ++j) 
      BOOST_CHECK_SMALL( fabs(t1(i,j)-t2(i,j)), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_gaussianSmoothing_2d_array )
{ 
  bob::ip::WeightedGaussian g_filter(1,1,0.5,0.5);
  blitz::Array<double,2> a_out(3,4);
  g_filter(a,a_out);
  checkBlitzClose( a_out, a_ref, eps);
}

BOOST_AUTO_TEST_CASE( test_gaussianSmoothing_2d_image )
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
  testdata_path_img /= "image.pgm";
  blitz::Array<uint8_t,2> img = bob::io::load<uint8_t,2>(testdata_path_img.string());
  blitz::Array<double,2> img_processed(img.shape());
  bob::ip::WeightedGaussian g_filter(1,1,0.5,0.5);
  g_filter(img,img_processed);

  // Compare to reference image
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_WeightedGaussian.hdf5";
  blitz::Array<double,2> img_ref = bob::io::load<double,2>(testdata_path_img.string());
  checkBlitzClose( img_processed, img_ref, eps);
}

BOOST_AUTO_TEST_SUITE_END()
