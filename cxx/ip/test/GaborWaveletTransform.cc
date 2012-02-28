/**
 * @file cxx/ip/test/GaborWaveletTransform.cc
 * @date 2012-02-27
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Test the Gabor wavelet transform and performs some sanity checks
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
#define BOOST_TEST_MODULE IP-GWT Tests
#define BOOST_TEST_MAIN

#include <cmath>
#include <fstream>
#include <sstream>

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <core/logging.h>
#include <core/convert.h>
#include <core/cast.h>
#include <io/Array.h>
#include <ip/GaborWaveletTransform.h>



struct T {
  const double epsilon;
  T(): epsilon(1e-4) { }
};


template <int D>
void test_identical(const blitz::TinyVector<int,D>& shape, const blitz::TinyVector<int,D>& reference){
  for (int i = D; i--;)
    BOOST_CHECK_EQUAL(shape[i], reference[i]);

}

void test_close(const blitz::Array<double, 3>& image, const blitz::Array<double, 3>& reference, const double epsilon){
  for (int x = image.extent(0); x--;)
    for (int y = image.extent(1); y--;)
      for (int z = image.extent(2); z--;)
        BOOST_CHECK_SMALL(image(x,y,z) - reference(x,y,z), epsilon);

}

void test_close(const blitz::Array<double, 4>& image, const blitz::Array<double, 4>& reference, const double epsilon){
  for (int w = image.extent(0); w--;)
    for (int x = image.extent(1); x--;)
      for (int y = image.extent(2); y--;)
        for (int z = image.extent(3); z--;)
          BOOST_CHECK_SMALL(image(w,x,y,z) - reference(w,x,y,z), epsilon);

}

void test_close(const blitz::Array<std::complex<double>, 3>& image, const blitz::Array<std::complex<double>, 3>& reference, const double epsilon){
  test_identical(image.shape(), reference.shape());
  for (int x = image.extent(0); x--;)
    for (int y = image.extent(1); y--;)
      for (int z = image.extent(2); z--;){
        BOOST_CHECK_SMALL(std::abs(image(x,y,z).real() - reference(x,y,z).real()), epsilon);
        BOOST_CHECK_SMALL(std::abs(image(x,y,z).imag() - reference(x,y,z).imag()), epsilon);
  }
}



BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_gwt_kernel_sanity )
{
  int v_res = 152, h_res = 152;
  blitz::TinyVector<int,6> horizontal_pairs(1,7,2,6,3,5);
  blitz::TinyVector<int,4> diagonal_pairs(0,4,1,3);
  for (int test_new_resolution = 2; test_new_resolution--;){
    bob::ip::GaborWaveletTransform gwt;
    gwt.generateKernels(blitz::TinyVector<int,2>(v_res, h_res));

    // This test will only work with 8 directions
    BOOST_CHECK_EQUAL(gwt.m_number_of_directions, 8);

    // get the kernel images
    blitz::Array<double,3> kernels = gwt.kernelImages();

    // check that the
    for (int scale = 0; scale < gwt.m_number_of_scales; ++scale){
      int scale_offset = gwt.m_number_of_directions * scale;
      // test the horizontal pairs
      for (int pair = 0; pair < 6; pair += 2){
        // get the two kernels to be checked
        blitz::Array<double,2>
            kernel_1 = kernels(scale_offset + horizontal_pairs[pair], blitz::Range::all(), blitz::Range::all()),
            kernel_2 = kernels(scale_offset + horizontal_pairs[pair+1], blitz::Range::all(), blitz::Range::all());

        for (int y = 0; y < v_res; ++y){
          // do not test the zero'th index since this is unique
          for (int x = 1; x < h_res; ++x){
            BOOST_CHECK_SMALL(std::abs(kernel_1(y,x) - kernel_2(y,h_res-x)), epsilon);
          }
        }

      }

      // test the diagonal pairs
      // these tests will work only with square kernels
      BOOST_CHECK_EQUAL(v_res, h_res);
      for (int pair = 0; pair < 4; pair += 2){
        // get the two kernels to be checked
        blitz::Array<double,2>
            kernel_1 = kernels(scale_offset + diagonal_pairs[pair], blitz::Range::all(), blitz::Range::all()),
            kernel_2 = kernels(scale_offset + diagonal_pairs[pair+1], blitz::Range::all(), blitz::Range::all());

        // do not test the zero'th index since this is unique
        for (int y = 1; y < v_res; ++y){
          for (int x = 1; x < h_res; ++x){
            BOOST_CHECK_SMALL(std::abs(kernel_1(y,x) - kernel_2(x,y)), epsilon);
          }
        }
      }
    }
    // do the tests again, this time with another resolution
    v_res = 183; h_res = 183;
  } // test_new_resolution
}

// #define GENERATE_NEW_REFERENCE_FILES

BOOST_AUTO_TEST_CASE( test_GWT_output )
{
  // Get path to the XML Schema definition
  char* data = getenv("BOB_IP_TESTDATA_DIR");
  if (!data){
    bob::core::error << "Environment variable $BOB_IP_TESTDATA_DIR "
        "is not set. Have you setup your working environment correctly?" << std::endl;
    throw bob::core::Exception();
  }
  std::string data_dir(data);

  // Load original image
  boost::filesystem::path image_file = boost::filesystem::path(data_dir) / "image.pgm";
  bob::io::Array io_image(image_file.string());
  blitz::Array<uint8_t,2> uint8_image = io_image.get<uint8_t,2>();
  blitz::Array<std::complex<double>,2> image = bob::core::cast<std::complex<double> >(uint8_image);

  // transform the image
  bob::ip::GaborWaveletTransform gwt(5, 8, std::sqrt(2.) * M_PI, M_PI/2, 1./1.414);
  blitz::Array<std::complex<double>, 3> gwt_image(gwt.numberOfKernels(), image.extent(0), image.extent(1));
  gwt.performGWT(image, gwt_image);

  // compute jet image
  blitz::Array<double,4> jet_image(image.extent(0), image.extent(1), 2, gwt.numberOfKernels());
  gwt.computeJetImage(image, jet_image, false);

  // Check the kernels
  boost::filesystem::path reference_kernel_file = boost::filesystem::path(data_dir) / "Gabor" / "gabor_filter_bank.hdf5";
  blitz::Array<double,3> kernels = gwt.kernelImages();
#ifdef GENERATE_NEW_REFERENCE_FILES
  bob::io::Array(kernels).save(reference_kernel_file.string());
#else
  bob::io::Array io_reference_kernel(reference_kernel_file.string());
  blitz::Array<double,3> ref_kernel = io_reference_kernel.get<double,3>();
  blitz::Array<double,3> gwt_kernel = gwt.kernelImages();
  test_close(gwt_kernel, ref_kernel, epsilon);
#endif


  // Check the transformed image
  boost::filesystem::path ref_image_file = boost::filesystem::path(data_dir) / "Gabor" / "gabor_filtered_complex.hdf5";
#ifdef GENERATE_NEW_REFERENCE_FILES
  bob::io::Array(gwt_image).save(ref_image_file.string());
#else
  bob::io::Array io_reference_image(ref_image_file.string());
  blitz::Array<std::complex<double>,3> reference_image = io_reference_image.get<std::complex<double>,3>();
  test_close(gwt_image, reference_image, epsilon);
#endif

  // Check the gabor jet image
  boost::filesystem::path ref_jet_image_file = boost::filesystem::path(data_dir) / "Gabor" / "gabor_jet_image.hdf5";
#ifdef GENERATE_NEW_REFERENCE_FILES
  bob::io::Array(jet_image).save(ref_jet_image_file.string());
#else
  bob::io::Array io_reference_jet_image(ref_jet_image_file.string());
  blitz::Array<double,4> reference_jet_image = io_reference_jet_image.get<double,4>();
  test_close(jet_image, reference_jet_image, epsilon);
#endif

}

BOOST_AUTO_TEST_SUITE_END()
