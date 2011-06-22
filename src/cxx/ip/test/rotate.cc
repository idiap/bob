/**
 * @file src/cxx/ip/test/rotate.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the rotation function for 2D and 3D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-Rotate Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/logging.h"
#include "core/convert.h"
#include "ip/Rotate.h"
#include "ip/shear.h"

#include "io/Array.h"
#include <algorithm>

#include <random/discrete-uniform.h>
#include <random/uniform.h>

#include <boost/filesystem.hpp>

struct T {
  blitz::Array<uint32_t,2> a2, a2r_90, a2r_180, a2r_270, a8;
  blitz::Array<uint32_t,2> a5, a55, a33;
  blitz::Array<bool,2> a8m, a8m_45;
  double eps;

  T(): a2(3,4), a2r_90(4,3), a2r_180(3,4), a2r_270(4,3), a8(8,8),
       a5(3,5), a55(5,5), a33(3,3), a8m(8,8), a8m_45(11,11), eps(0.03)
  {
    a2 = 0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11;

    a2r_90 = 3, 7, 11,
        2, 6, 10,
        1, 5, 9,
        0, 4, 8;

    a2r_180 = 11, 10, 9, 8,
        7, 6, 5, 4,
        3, 2, 1, 0;
  
    a2r_270 = 8, 4, 0,
        9, 5, 1,
        10, 6, 2,
        11, 7, 3;

    a8 = 1;
    a8m = true;
    // The following might likely be improved. Do not consider that as a pure 
    // reference mask. Bilinear interpolation might be better with masks, as
    // the mask keep decreasing in size while performing successive shearing 
    // operations.
    a8m_45 = 
      false, false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, true, false, false, false, false, false, false,
      false, false, false, true, true, true, false, false, false, false, false,
      false, false, false, true, true, true, true, true, false, false, false,
      false, true, true, true, true, true, true, true, false, false, false,
      true, true, true, true, true, true, true, true, true, true, false,
      false, false, true, true, true, true, true, true, false, false, false,
      false, false, false, true, true, true, true, false, false, false, false,
      false, false, false, false, true, true, true, false, false, false, false,
      false, false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, false, false, false, false, false, false, false;
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

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), Torch::core::cast<T>(t2(i,j,k)));
}


template<typename T, typename U>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<U,2>& t2,
  const double eps )
{
  int y_min = std::min( t1.extent(0), t2.extent(0));
  int x_min = std::min( t1.extent(1), t2.extent(1));
  BOOST_CHECK_LE( abs(t1.extent(0)-t2.extent(0)), 1);
  BOOST_CHECK_LE( abs(t1.extent(1)-t2.extent(1)), 1);
  double diff = 0.;
  for( int i=0; i<y_min; ++i)
    for( int j=0; j<x_min; ++j)
      diff += abs( t1(i,j) - Torch::core::cast<T>(t2(i,j)) );
  diff = (diff/(y_min*x_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  BOOST_CHECK_SMALL( diff, eps );
}

template<typename T,typename U>  
void checkBlitzClose( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2,
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
        diff += abs( t1(i,j,k) - Torch::core::cast<T>(t2(i,j,k)) );
  diff = (diff/(y_min*x_min*p_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  BOOST_CHECK_SMALL( diff, eps );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_rotate_2d_mod90_uint32 )
{
  blitz::Array<double,2> b2;

  // Rotation of 0
  Torch::ip::Rotate rotate(0.);
  b2.resize( rotate.getOutputShape(a2,0.) );
  rotate(a2, b2, 0.);
  checkBlitzEqual(a2, b2); 

  // Rotation of 90
  b2.resize( rotate.getOutputShape(a2,90.) );
  rotate(a2, b2, 90.);
  checkBlitzEqual(a2r_90, b2); 

  // Rotation of 180
  b2.resize( rotate.getOutputShape(a2,180.) );
  rotate(a2, b2, 180.);
  checkBlitzEqual(a2r_180, b2); 

  // Rotation of 270
  b2.resize( rotate.getOutputShape(a2,270.) );
  rotate(a2, b2, 270.);
  checkBlitzEqual(a2r_270, b2); 
}


BOOST_AUTO_TEST_CASE( test_rotate_2d_generic_uint32 )
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
  Torch::io::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,2> img = ar_img.get<uint8_t,2>();
  blitz::Array<double,2> img_processed;

  // Rotate original image and compare with ImageMagick reference image
  // Warning: ImageMagick considers opposite angles wrt. to us
  Torch::ip::Rotate rotate(5., Torch::ip::Rotate::Shearing);

  // 5 degrees
  img_processed.resize(rotate.getOutputShape(img,5.) ); 
  rotate( img, img_processed, 5.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r5.pgm";
  Torch::io::Array ar_img_r5(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r5 = ar_img_r5.get<uint8_t,2>();
  checkBlitzClose( img_ref_r5, img_processed, eps);

  // 10 degrees 
  img_processed.resize(rotate.getOutputShape(img,10.) ); 
  rotate( img, img_processed, 10.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r10.pgm";
  Torch::io::Array ar_img_r10(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r10 = ar_img_r10.get<uint8_t,2>();
  checkBlitzClose( img_ref_r10, img_processed, eps);

  // 15 degrees 
  img_processed.resize(rotate.getOutputShape(img,15.) ); 
  rotate( img, img_processed, 15.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r15.pgm";
  Torch::io::Array ar_img_r15(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r15 = ar_img_r15.get<uint8_t,2>();
  checkBlitzClose( img_ref_r15, img_processed, eps);

  // 30 degrees 
  img_processed.resize(rotate.getOutputShape(img,30.) ); 
  rotate( img, img_processed, 30.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r30.pgm";
  Torch::io::Array ar_img_r30(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r30 = ar_img_r30.get<uint8_t,2>();
  checkBlitzClose( img_ref_r30, img_processed, eps);

  // 45 degrees 
  img_processed.resize(rotate.getOutputShape(img,45.) ); 
  rotate( img, img_processed, 45.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r45.pgm";
  Torch::io::Array ar_img_r45(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r45 = ar_img_r45.get<uint8_t,2>();
  checkBlitzClose( img_ref_r45, img_processed, eps);

  // 70 degrees 
  img_processed.resize(rotate.getOutputShape(img,70.) ); 
  rotate( img, img_processed, 70.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r70.pgm";
  Torch::io::Array ar_img_r70(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r70 = ar_img_r70.get<uint8_t,2>();
  checkBlitzClose( img_ref_r70, img_processed, eps);

  // 237 degrees 
  img_processed.resize(rotate.getOutputShape(img,237.) ); 
  rotate( img, img_processed, 237.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_r237.pgm";
  Torch::io::Array ar_img_r237(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_r237 = ar_img_r237.get<uint8_t,2>();
  checkBlitzClose( img_ref_r237, img_processed, eps);

  // -25 degrees 
  img_processed.resize(rotate.getOutputShape(img,-25.) ); 
  rotate( img, img_processed, -25.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "image_rn25.pgm";
  Torch::io::Array ar_img_rn25(testdata_path_img.string());
  blitz::Array<uint8_t,2> img_ref_rn25 = ar_img_rn25.get<uint8_t,2>();
  checkBlitzClose( img_ref_rn25, img_processed, eps);
}

BOOST_AUTO_TEST_CASE( test_rotate_3d_generic_uint32 )
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
  testdata_path_img /= "imageColor.ppm";
  Torch::io::Array ar_img(testdata_path_img.string());
  blitz::Array<uint8_t,3> img = ar_img.get<uint8_t,3>();
  blitz::Array<double,3> img_processed;

  // 5 degrees 
  Torch::ip::Rotate rotate(5., Torch::ip::Rotate::Shearing);
  img_processed.resize(rotate.getOutputShape(img,5.) ); 
  rotate( img, img_processed, 5.);
  testdata_path_img = testdata_cpath;
  testdata_path_img /= "imageColor_r5.ppm";
  Torch::io::Array ar_img_r5(testdata_path_img.string());
  blitz::Array<uint8_t,3> img_ref_r5 = ar_img_r5.get<uint8_t,3>();
  checkBlitzClose( img_ref_r5, img_processed, eps);
}

BOOST_AUTO_TEST_CASE( test_rotate_2d_mask )
{
  blitz::Array<double,2> b8;
  blitz::Array<bool,2> b8_mask;

  // Rotation of 45
  Torch::ip::Rotate rotate(45.);
  b8.resize( rotate.getOutputShape(a8, 45.) );
  b8_mask.resize( b8.shape() );
  rotate(a8, a8m, b8, b8_mask, 45.);
  checkBlitzEqual(a8m_45, b8_mask); 
}

BOOST_AUTO_TEST_SUITE_END()
