/**
 * @file src/core/test/binary_file.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief BinOutputFile tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BinaryFile Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "core/BinOutputFile.h"
#include "core/BinInputFile.h"
#include "core/StaticComplexCast.h"

struct T {

  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<uint32_t,1> c;
  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  std::string tpl = Torch::core::tmpdir();
  tpl += "/torchtest_core_binformatXXXXXX";
  boost::shared_ptr<char> char_tpl(new char[tpl.size()+1]);
  strcpy(char_tpl.get(), tpl.c_str());
  int fd = mkstemp(char_tpl.get());
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  return char_tpl.get();
}

template<typename T, typename U, int d> void check_equal_1d(const blitz::Array<T,d>& a,
  const blitz::Array<U,d>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  T val;
  for (int i=0; i<a.extent(0); ++i) {
    Torch::core::static_complex_cast(b(i), val);
    BOOST_CHECK_EQUAL(a(i), val);
  }
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( blitz1d )
{
  std::string tmp_file = temp_file();
  Torch::core::BinOutputFile out(tmp_file);

  out.write( a);
  out.close();

  Torch::core::BinInputFile in(tmp_file);
  
  in.read( b);
  check_equal_1d( a, b);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz1d_withcast )
{
  std::string tmp_file = temp_file();
  Torch::core::BinOutputFile out(tmp_file);

  out.write( c);
  out.close();

  Torch::core::BinInputFile in(tmp_file);
  
  in.read( b);
  check_equal_1d( c, b);
  in.close();
}

BOOST_AUTO_TEST_SUITE_END()
