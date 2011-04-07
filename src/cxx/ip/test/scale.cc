/**
 * @file src/cxx/ip/test/scale.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the rescaling function for 2D and 3D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-Scale Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/logging.h"
#include "ip/scale.h"

#include "database/Array.h"
#include <algorithm>

#include <random/discrete-uniform.h>
#include <random/uniform.h>

#include <boost/filesystem.hpp>

struct T {
  double eps;

  T(): eps(0.01) {}

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

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), Torch::core::cast<T>(t2(i,j,k)));
}


template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  int y_min = std::min( t1.extent(0), t2.extent(0));
  int x_min = std::min( t1.extent(1), t2.extent(1));
  BOOST_CHECK_LE( abs(t1.extent(0)-t2.extent(0)), 1);
  BOOST_CHECK_LE( abs(t1.extent(1)-t2.extent(1)), 1);
  double diff = 0.;
  for( int i=0; i<y_min; ++i)
    for( int j=0; j<x_min; ++j)
      diff += abs( t1(i,j) - t2(i,j) );
  diff = (diff/(y_min*x_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  std::cout << diff << std::endl;
  BOOST_CHECK_SMALL( diff, eps );
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,3>& t1, blitz::Array<T,3>& t2, 
  const double eps )
{
  int p_min = std::min( t1.extent(0), t2.extent(0));
  int y_min = std::min( t1.extent(1), t2.extent(1));
  int x_min = std::min( t1.extent(2), t2.extent(2));
  BOOST_CHECK_EQUAL( t1.extent(0), t2.extent(0) );
  BOOST_CHECK_LE( abs(t1.extent(1)-t2.extent(1)), 1);
  BOOST_CHECK_LE( abs(t1.extent(2)-t2.extent(2)), 1);
  double diff = 0.;
  for( int i=0; i<p_min; ++i)
    for( int j=0; j<y_min; ++j)
      for( int k=0; k<x_min; ++k)
        diff += abs( t1(i,j,k) - t2(i,j,k) );
  diff = (diff/(y_min*x_min*p_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  BOOST_CHECK_SMALL( diff, eps );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_scale_2d_generic_uint8 )
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
  testdata_path_img /= "image.pgm";
  Torch::database::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,2> img = ar_img.get<uint8_t,2>();
  blitz::Array<uint8_t,2> img_processed;


  // Scale original image and compare with ImageMagick reference image

  // 137x137 
  Torch::ip::scale( img, img_processed, 137, 137, 
    Torch::ip::Rescale::BilinearInterp);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_s137x137.pgm";
  Torch::database::Array ar_img_s137(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_s137 = ar_img_s137.get<uint8_t,2>();
  checkBlitzClose( img_ref_s137, img_processed, eps);

  // 77x77
  Torch::ip::scale( img, img_processed, 77, 77, 
    Torch::ip::Rescale::BilinearInterp);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_s77x77.pgm";
  Torch::database::Array ar_img_s77(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_s77 = ar_img_s77.get<uint8_t,2>();
  checkBlitzClose( img_ref_s77, img_processed, eps);

  // 125x75
  Torch::ip::scale( img, img_processed, 125, 75,
    Torch::ip::Rescale::BilinearInterp);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_s125x75.pgm";
  Torch::database::Array ar_img_s125x75(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_s125x75 = ar_img_s125x75.get<uint8_t,2>();
  checkBlitzClose( img_ref_s125x75, img_processed, eps);
}

BOOST_AUTO_TEST_SUITE_END()
