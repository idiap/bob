/**
 * @file src/cxx/core/test/binary_file.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief BinOutputFile tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BinaryFile Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

#include "core/BinFile.h"
#include "core/cast.h"

struct T {

  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<uint32_t,1> c;

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

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  T val;
  for (int i=0; i<a.extent(0); ++i) {
    Torch::core::static_complex_cast(b(i), val);
    BOOST_CHECK_EQUAL(a(i), val);
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  T val;
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      Torch::core::static_complex_cast(b(i,j), val);
      BOOST_CHECK_EQUAL(a(i,j), val);
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
  T val;
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          Torch::core::static_complex_cast(b(i,j,k,l), val);
          BOOST_CHECK_EQUAL(a(i,j,k,l), val);
        }
      }
    }
  }
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( blitz1d )
{
  std::string tmp_file = temp_file();
  Torch::core::BinFile out(tmp_file, Torch::core::BinFile::out);

  out.write( a);
  out.close();

  Torch::core::BinFile in(tmp_file, Torch::core::BinFile::in);
  
  blitz::Array<double,1> a_read = in.read<double,1>();

  check_equal_1d( a, a_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz1d_withcast )
{
  std::string tmp_file = temp_file();
  Torch::core::BinFile out(tmp_file, Torch::core::BinFile::out);

  out.write( c);
  out.close();

  Torch::core::BinFile in(tmp_file, Torch::core::BinFile::in);
  
  blitz::Array<double,1> c_read = in.read<double,1>();
  check_equal_1d( c, c_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz2d_directaccess )
{
  std::string tmp_file = temp_file();
  Torch::core::BinFile out(tmp_file, Torch::core::BinFile::out);

  out.write( d);
  out.write( e);
  out.write( d);
  out.close();

  Torch::core::BinFile in(tmp_file, Torch::core::BinFile::in);
  
  blitz::Array<float,2> e_read = in.read<float,2>(1);
  check_equal_2d( e, e_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz1d_inout )
{
  std::string tmp_file = temp_file();
  Torch::core::BinFile out(tmp_file, Torch::core::BinFile::out);

  out.write( a);
  out.close();

  Torch::core::BinFile in(tmp_file, Torch::core::BinFile::in | 
    Torch::core::BinFile::out);
  
  blitz::Array<double,1> a_read = in.read<double,1>();
  check_equal_1d( a, a_read);
  in.close();
}

BOOST_AUTO_TEST_CASE( blitz4d_slice )
{
  std::string tmp_file1 = temp_file();
  Torch::core::BinFile out1(tmp_file1, Torch::core::BinFile::out);
  std::string tmp_file2 = temp_file();
  Torch::core::BinFile out2(tmp_file2, Torch::core::BinFile::out);

  for(int i=0; i<2;++i)
    for(int j=0; j<3;++j)
      for(int k=0; k<4;++k)
        for(int l=0; l<5;++l)
          g(i,j,k,l) = i*3*4*5+j*4*5+k*5+l;

  blitz::Array<double,4> g_sliced1 = g(blitz::Range::all(), blitz::Range(0,0),
    blitz::Range::all(), blitz::Range::all());

  out1.write( g_sliced1);
  out1.close();

  Torch::core::BinFile in1(tmp_file1, Torch::core::BinFile::in);
  
  blitz::Array<double,4> g_sliced1_read = in1.read<double,4>();
  check_equal_4d( g_sliced1, g_sliced1_read);
  in1.close();

  blitz::Array<double,4> g_sliced2 = g(blitz::Range(0,0), blitz::Range::all(),
    blitz::Range::all(), blitz::Range::all());

  out2.write( g_sliced2);
  out2.close();

  Torch::core::BinFile in2(tmp_file2, Torch::core::BinFile::in);
  
  blitz::Array<double,4> g_sliced2_read = in2.read<double,4>();
  check_equal_4d( g_sliced2, g_sliced2_read);
  in1.close();
}

BOOST_AUTO_TEST_SUITE_END()
