/**
 * @file cxx/core/test/blitzAdapter.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Test the blitz adapter and blitz additions
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
#define BOOST_TEST_MODULE Core-Tensor Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "core/logging.h"
#include "core/cast.h"
#include "core/Tensor.h"
#include "core/BlitzAdapter.h"
#include <iostream>
#include <sstream>


struct T {
  typedef blitz::Array<double,2> BAd2;
  typedef blitz::Array<int,3> BAi3;
  typedef blitz::Array<double,3> BAd3;
  typedef blitz::Array<int8_t,2> BAi8_2;
  typedef blitz::Array<uint8_t,2> BAui8_2;
  BAd2 bl1, bl2;
  BAi3 bl3, bl4;
  BAd3 bl5;
  BAi8_2 bl6, bl7;
  BAui8_2 bl8, bl9;
  

  T(): bl1(3,5), bl3(2,2,2), bl6(2,3), bl8(3,2) {
    bl1 = 1.1, 0, 0, 1, 5,  
          1.3, 2, 3, 4, 5,  
          6.5, 7, 8, 9, 10;
    
    bl3 = 5, 4,   6, 7,
          11, -6, 37, 54;

    bl6 = 0, 1, 2, 3, 4, 5;

    bl8 = -2, -1, 0, 0, 1, 2;
  }

  ~T() {}

};

template<typename BA1, typename BA2>  void check_dimensions( BA1& t1, BA2& t2) {
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename BA1, typename BA2>  void check_equal2d( BA1& t1, BA2& t2) {
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

template<typename BA1, typename BA2>  void check_equal3d( BA1& t1, BA2& t2) {
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), t2(i,j,k));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

//this will save, load and compare two blitz arrays
BOOST_AUTO_TEST_CASE( test_write_read )
{
  // 1/ 2D double array
  // Save
  std::stringstream stream_d;
  Torch::core::BlitzAdapter<BAd2> X(bl1);
  stream_d << X;
  // Load
  Torch::core::BlitzAdapter<BAd2> Y(bl2);
  stream_d >> Y;
  // Compare
  check_equal2d( bl1, bl2);

  // 2/ 3D int array
  // Save
  std::stringstream stream_i;
  Torch::core::BlitzAdapter<BAi3> Z(bl3);
  stream_i << Z;
  // Load
  Torch::core::BlitzAdapter<BAi3> T(bl4);
  stream_i >> T;
  // Compare
  check_equal3d( bl3, bl4);
}


//this will save, load and compare two blitz arrays
//of type int8_t and uint8_t
BOOST_AUTO_TEST_CASE( test_write_read_int8_uint8 )
{
  // 2D/ int8_t array
  // Save
  std::stringstream stream_i;
  Torch::core::BlitzAdapter<BAi8_2> A(bl6);
  stream_i << A;
  // Load
  Torch::core::BlitzAdapter<BAi8_2> B(bl7);
  stream_i >> B;
  // Compare
  check_equal2d( bl6, bl7);

  // 2D/ uint8_t array
  // Save
  std::stringstream stream_u;
  Torch::core::BlitzAdapter<BAui8_2> C(bl8);
  stream_u << C;
  // Load
  Torch::core::BlitzAdapter<BAui8_2> D(bl9);
  stream_u >> D;
  // Compare
  check_equal2d( bl8, bl9);
}

//this will save, read a converted blitz array and compare it to the original
BOOST_AUTO_TEST_CASE( test_write_convertread )
{
  // 1/ 3D array: Load an int array into a double array
  // Save
  std::stringstream stream_i;
  Torch::core::BlitzAdapter<BAi3> X(bl3);
  stream_i << X;
  // Load
  Torch::core::BlitzAdapter<BAd3> Y(bl5, false);
  stream_i >> Y;
  // Compare
  check_equal3d( bl3, bl5);
}


BOOST_AUTO_TEST_SUITE_END()

