/**
 * @file cxx/io/test/binary_file.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binary file tests
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
#define BOOST_TEST_MODULE BinaryFile Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <blitz/array.h>
#include <string>
#include "bob/core/logging.h"
#include "bob/core/cast.h"
#include "bob/io/BinFile.h"


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
    g = 37.;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  boost::filesystem::path tpl = bob::core::tmpdir();
  tpl /= "bobtest_core_binformatXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), bob::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), bob::core::cast<T>(b(i,j)));
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
          BOOST_CHECK_EQUAL(a(i,j,k,l), bob::core::cast<T>(b(i,j,k,l)));
        }
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( blitz1d )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( a);
  out.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  blitz::Array<double,1> a_read = in.read<double,1>();

  check_equal_1d( a, a_read);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz1d_withcast )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( c);
  out.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  
  blitz::Array<double,1> c_read = in.read<double,1>();
  check_equal_1d( c, c_read);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz2d )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( d);
  out.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  blitz::Array<float,2> d_read = in.read<float,2>();

  check_equal_2d( d, d_read);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz1d_inout )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( a);
  out.close();

  bob::io::BinFile inoutap(tmp_file, bob::io::BinFile::in | 
    bob::io::BinFile::out | bob::io::BinFile::append);
  
  inoutap.write( a);
  inoutap.write( a);
  inoutap.write( a);

  blitz::Array<double,1> a_read = inoutap.read<double,1>(0);
  check_equal_1d( a, a_read);
  inoutap.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz1d_append )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( a);
  out.close();

  bob::io::BinFile outap(tmp_file, bob::io::BinFile::out | 
    bob::io::BinFile::append);
  
  outap.write( a);
  outap.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  blitz::Array<double,1> a_read1 = in.read<double,1>(0);
  check_equal_1d( a, a_read1);
  blitz::Array<double,1> a_read2 = in.read<double,1>(1);
  check_equal_1d( a, a_read2);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz2d_withcast )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( d);
  out.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  blitz::Array<uint32_t,2> d_read = in.read<uint32_t,2>();

  check_equal_2d( d, d_read);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz2d_directaccess )
{
  std::string tmp_file = temp_file();
  bob::io::BinFile out(tmp_file, bob::io::BinFile::out);

  out.write( d);
  out.write( e);
  out.write( d);
  out.close();

  bob::io::BinFile in(tmp_file, bob::io::BinFile::in);
  blitz::Array<float,2> e_read = in.read<float,2>(1);
  
  check_equal_2d( e, e_read);
  in.close();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( blitz4d_slice )
{
  std::string tmp_file1 = temp_file();
  bob::io::BinFile out1(tmp_file1, bob::io::BinFile::out);
  std::string tmp_file2 = temp_file();
  bob::io::BinFile out2(tmp_file2, bob::io::BinFile::out);

  for(int i=0; i<2;++i)
    for(int j=0; j<3;++j)
      for(int k=0; k<4;++k)
        for(int l=0; l<5;++l)
          g(i,j,k,l) = i*3*4*5+j*4*5+k*5+l;

  blitz::Array<double,4> g_sliced1 = g(blitz::Range::all(), blitz::Range(0,0),
    blitz::Range::all(), blitz::Range::all());

  out1.write( g_sliced1);
  out1.close();

  bob::io::BinFile in1(tmp_file1, bob::io::BinFile::in);
  
  blitz::Array<double,4> g_sliced1_read = in1.read<double,4>();
  check_equal_4d( g_sliced1, g_sliced1_read);
  in1.close();

  blitz::Array<double,4> g_sliced2 = g(blitz::Range(0,0), blitz::Range::all(),
    blitz::Range::all(), blitz::Range::all());

  out2.write( g_sliced2);
  out2.close();

  bob::io::BinFile in2(tmp_file2, bob::io::BinFile::in);
  
  blitz::Array<double,4> g_sliced2_read = in2.read<double,4>();
  check_equal_4d( g_sliced2, g_sliced2_read);
  in1.close();

  // Clean-up
  boost::filesystem::remove(tmp_file1);
  boost::filesystem::remove(tmp_file2);
}

BOOST_AUTO_TEST_SUITE_END()
