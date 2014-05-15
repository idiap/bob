/**
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief ImageArrayCodec tests
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TensorArrayCodec Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <blitz/array.h>
#include "bob/core/logging.h"
#include "bob/io/utils.h"

struct T {
  blitz::Array<int8_t,2> a, b;

  T() {
    a.resize(6,4);
    a = 1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24;
    b.resize(3,4);
    b = 0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11;
  }

  ~T() { }

};

template<typename T, typename U>
void check_equal(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b)
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), bob::core::cast<T>(b(i,j)));
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( tensor_2d )
{
  std::string filename = bob::core::tmpfile(".tensor");
  bob::io::save(filename.c_str(), a);
  check_equal( bob::io::load<int8_t,2>(filename.c_str()), a );
  boost::filesystem::remove(filename);
}

BOOST_AUTO_TEST_CASE( tensor_2d_read_T5alpha )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("BOB_TESTDATA_DIR");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    throw std::runtime_error("Environment variable $BOB_TESTDATA_DIR is not set. Have you setup your working environment correctly?");
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "tensor_char.tensor";

  check_equal( bob::io::load<int8_t,2>(testdata_path.string().c_str()), b );
}

BOOST_AUTO_TEST_SUITE_END()
