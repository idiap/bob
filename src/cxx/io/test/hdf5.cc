/**
 * @file cxx/io/test/hdf5.cc
 * @date Tue Aug 9 21:43:02 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief IO hdf5 tests
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE DbArray Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <blitz/array.h>
#include <complex>
#include <string>
#include "core/logging.h" // for bob::core::tmpdir()
#include "core/cast.h"
#include "io/HDF5File.h"

struct T {
  blitz::Array<double,2> a;
  blitz::Array<double,1> c;

  T(): a(4,2), c(5) {
    a = 1, 2, 
        3, 4,
        5, 6,
        7, 8;
    c = 5, 4, 3, 2, 1;
  }

  ~T() { }

};


/**
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  boost::filesystem::path tpl = bob::core::tmpdir();
  tpl /= "bobtest_core_hdf5XXXXXX.hdf5";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),5);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), bob::core::cast<T>(b(i)) );
  }
}

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

template<typename T, typename U> 
void check_equal(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
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

BOOST_AUTO_TEST_CASE( hdf5_1d_save_read )
{
  // Put a 1D array in a HDF5File
  const std::string filename = temp_file();
  bob::io::HDF5File::mode_t flag = bob::io::HDF5File::inout;
  bob::io::HDF5File config(filename, flag);
  config.setArray("c", c);

  // Read it and compare to original
  blitz::Array<double,1> c_read;
  c_read.reference(config.readArray<double,1>("c"));
  check_equal(c, c_read);
}

BOOST_AUTO_TEST_CASE( hdf5_2d_save_read )
{
  // Put a 2D array in a HDF5File
  const std::string filename = temp_file();
  bob::io::HDF5File::mode_t flag = bob::io::HDF5File::inout;
  bob::io::HDF5File config(filename, flag);
  blitz::Array<double,2> at = a.transpose(1,0);
  config.setArray("at", at);

  // Read it and compare to original
  blitz::Array<double,2> at_read;
  at_read.reference(config.readArray<double,2>("at"));
  check_equal(at, at_read);
}

BOOST_AUTO_TEST_SUITE_END()
