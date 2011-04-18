/**
 * @file src/cxx/ip/test/Gabor.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the Gabor filtering function for 2D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-Gabor Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/convert.h"
#include "ip/GaborSpatial.h"
//#include "ip/GaborBankSpatial.h"
#include "ip/GaborFrequency.h"
//#include "ip/GaborBankFrequency.h"

#include "database/Array.h"
#include <algorithm>
#include <boost/filesystem.hpp>

struct T {
  double eps1, eps2;
  int eps3;

  T(): eps1(0.03), eps2(0.22), eps3(10) { }

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions( const blitz::Array<T,d>& t1, const blitz::Array<U,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T>  
void checkBlitzClose( const blitz::Array<T,2>& t1, const blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( (double)abs( t1(i,j) - t2(i,j) ), eps);
}

template<typename T>  
void checkBlitzMeanClose( const blitz::Array<T,2>& t1, const blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  double diff = 0.;
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      diff += abs( t1(i,j) - t2(i,j) );
  diff /= (t1.extent(0)*t1.extent(1));
  BOOST_CHECK_SMALL( diff, eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_Gabor_2d_spatial )
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
  blitz::Array<std::complex<double>,2> img_processed(img.shape());
  blitz::Array<std::complex<double>,2> img_src = 
    Torch::core::cast<std::complex<double> >(img);
  Torch::ip::GaborSpatial GS_filter;

  // Check the kernel
  boost::filesystem::path path_kernel_ref( testdata_cpath);
  path_kernel_ref /= "Gabor/gabor_spatial_filter.hdf5";
  Torch::database::Array ar_spatial_kernel(path_kernel_ref.string());
  blitz::Array<std::complex<double>,2> ref_kernel = ar_spatial_kernel.get<std::complex<double>,2>();
  checkBlitzClose( GS_filter.getKernel(), ref_kernel, eps1);

  // Check the filtered image
  GS_filter(img_src,img_processed);
  boost::filesystem::path path_img_ref( testdata_cpath);
  path_img_ref /= "Gabor/gabor_spatial_filtered.hdf5";
  Torch::database::Array ar_spatial_image(path_img_ref.string());
  blitz::Array<std::complex<double>,2> ref_image = ar_spatial_image.get<std::complex<double>,2>();
  checkBlitzClose( img_processed, ref_image, eps1);
}

BOOST_AUTO_TEST_CASE( test_Gabor_2d_frequency )
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
  blitz::Array<std::complex<double>,2> img_processed( img.shape());
  blitz::Array<std::complex<double>,2> img_src = 
    Torch::core::cast<std::complex<double> >(img);
  Torch::ip::GaborFrequency GF_filter(img.extent(0),img.extent(1));

  // Check the kernel
  boost::filesystem::path path_kernel_ref( testdata_cpath);
  path_kernel_ref /= "Gabor/gabor_frequency_filter.hdf5";
  Torch::database::Array ar_frequency_kernel(path_kernel_ref.string());
  blitz::Array<std::complex<double>,2> ref_kernel = ar_frequency_kernel.get<std::complex<double>,2>();
  checkBlitzClose( GF_filter.getKernelShifted(), ref_kernel, eps1);

  // Check the filtered image
  GF_filter(img_src,img_processed);
  boost::filesystem::path path_img_ref( testdata_cpath);
  path_img_ref /= "Gabor/gabor_frequency_filtered.hdf5";
  Torch::database::Array ar_frequency_image(path_img_ref.string());
  blitz::Array<std::complex<double>,2> ref_image = ar_frequency_image.get<std::complex<double>,2>();
  checkBlitzClose( img_processed, ref_image, eps2);

  // Check with the uint8_t magnitude version
  blitz::Array<double,2> img_mag(img_processed.extent(0), img_processed.extent(1));
  blitz::Range  i0(img_processed.lbound(0), img_processed.ubound(0)),
                i1(img_processed.lbound(1), img_processed.ubound(1));
  img_mag(i0,i1) = abs(img_processed(i0,i1));
  blitz::Array<uint8_t,2> img_mag_uint = Torch::core::convertFromRange<uint8_t>(
    img_mag, min(img_mag), max(img_mag) );
  boost::filesystem::path path_img_ref_pgm( testdata_cpath);
  path_img_ref_pgm /= "Gabor/gabor_frequency_filtered.pgm";
  Torch::database::Array ar_frequency_image_pgm(path_img_ref_pgm.string());
  blitz::Array<uint8_t,2> img_ref_pgm = ar_frequency_image_pgm.get<uint8_t,2>();
  checkBlitzMeanClose( img_mag_uint, img_ref_pgm, eps3);
}

BOOST_AUTO_TEST_CASE( test_Gabor_2d_frequency_envelope )
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
  blitz::Array<std::complex<double>,2> img_processed(img.shape());
  blitz::Array<std::complex<double>,2> img_src = 
    Torch::core::cast<std::complex<double> >(img);
  Torch::ip::GaborFrequency GF_filter(img.extent(0),img.extent(1), 0.25, 0., 1., 1., 0.99, false, true);

  // Check the kernel
  boost::filesystem::path path_kernel_ref( testdata_cpath);
  path_kernel_ref /= "Gabor/gabor_frequency_filter.hdf5";
  Torch::database::Array ar_frequency_kernel(path_kernel_ref.string());
  blitz::Array<std::complex<double>,2> ref_kernel = ar_frequency_kernel.get<std::complex<double>,2>();
  checkBlitzClose( GF_filter.getKernelShifted(), ref_kernel, eps1);

  // Check the filtered image
  GF_filter(img_src,img_processed);
  boost::filesystem::path path_img_ref( testdata_cpath);
  path_img_ref /= "Gabor/gabor_frequency_filtered.hdf5";
  Torch::database::Array ar_frequency_image(path_img_ref.string());
  blitz::Array<std::complex<double>,2> ref_image = ar_frequency_image.get<std::complex<double>,2>();
  checkBlitzClose( img_processed, ref_image, eps2);

  // Check with the uint8_t magnitude version
  blitz::Array<double,2> img_mag(img_processed.extent(0), img_processed.extent(1));
  blitz::Range  i0(img_processed.lbound(0), img_processed.ubound(0)),
                i1(img_processed.lbound(1), img_processed.ubound(1));
  img_mag(i0,i1) = abs(img_processed(i0,i1));
  blitz::Array<uint8_t,2> img_mag_uint = Torch::core::convertFromRange<uint8_t>(
    img_mag, min(img_mag), max(img_mag) );
  boost::filesystem::path path_img_ref_pgm( testdata_cpath);
  path_img_ref_pgm /= "Gabor/gabor_frequency_filtered.pgm";
  Torch::database::Array ar_frequency_image_pgm(path_img_ref_pgm.string());
  blitz::Array<uint8_t,2> img_ref_pgm = ar_frequency_image_pgm.get<uint8_t,2>();
  checkBlitzMeanClose( img_mag_uint, img_ref_pgm, eps3);
}

BOOST_AUTO_TEST_SUITE_END()
