/**
 * @file src/cxx/database/test/arrayset.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Database Arrayset tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DbArrayset Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <vector>

#include <blitz/array.h>
#include "core/cast.h"
#include "database/BinFile.h"
#include "database/Dataset.h"

#include <unistd.h>
#include <iostream>

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;

  blitz::Array<float,2> d;
  blitz::Array<float,2> e;
  blitz::Array<float,2> f;

  blitz::Array<double,4> g;
  blitz::Array<double,4> h;

  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;

    d.resize(2,2);
    d = 1, 2, 3, 4;
    e.resize(2,2);
    e = 5, 6, 7, 8;

    g.resize(2,3,4,5);
    g = 37.;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_core_binformatXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
//  return char_tpl.get();
}

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), Torch::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal_4d(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  BOOST_REQUIRE_EQUAL(a.extent(3), b.extent(3));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          BOOST_CHECK_EQUAL(a(i,j,k,l), Torch::core::cast<T>(b(i,j,k,l)));
        }
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( dbDataset_load_xml )
{
  // Get path to the XML Schema definition
  char *testdata_cpath = getenv("TORCH_TESTDATA_PATH");
  if( !testdata_cpath || !strcmp( testdata_cpath, "") ) {
    Torch::core::error << "Environment variable $TORCH_TESTDATA_PATH " <<
      "is not set. " << "Have you setup your working environment " <<
      "correctly?" << std::endl;
    throw Torch::core::Exception();
  }
  boost::filesystem::path testdata_path( testdata_cpath);
  testdata_path /= "data";
  testdata_path /= "db_inline.xml";

  Torch::database::Dataset d(testdata_path.string());
}

BOOST_AUTO_TEST_SUITE_END()

