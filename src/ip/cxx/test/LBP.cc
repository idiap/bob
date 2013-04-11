/**
 * @file ip/cxx/test/LBP.cc
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Rewritten:
 * @date Wed Apr 10 17:39:21 CEST 2013
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 *
 * @brief Test the LBP functions for 2D arrays/images
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE IP-Lbp Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "bob/ip/LBP.h"

#include <iostream>

struct T {
  blitz::Array<uint8_t,2> a1, a2;
  uint16_t lbp_4_a1, lbp_4_a2, lbp_4_a1_u2, lbp_4_a2_u2, lbp_4_ri, lbp_4_ur,
           lbp_8_a1, lbp_8_a2, lbp_8_a1_u2, lbp_8_a2_u2, lbp_8_ri, lbp_8_ur,
           lbp_16_a1, lbp_16_a2, lbp_16_a1_u2, lbp_16_a2_u2, lbp_16_ri, lbp_16_ur,

           lbp_4_a1_d, lbp_4_a2_d, lbp_8_a1_d, lbp_8_a2_d, lbp_16_a1_d, lbp_16_a2_d,
           lbp_4_a1_t, lbp_4_a2_t, lbp_8_a1_t, lbp_8_a2_t, lbp_16_a1_t, lbp_16_a2_t;

  T(): a1(3,3), a2(3,3)
  {
    a1 = 0, 1, 2,
         3, 4, 5,
         6, 7, 8;

    a2 = 8, 7, 6,
         5, 4, 3,
         2, 1, 0;

    // normal LBP4
    lbp_4_a1 = 6;
    lbp_4_a2 = 9;
    lbp_4_a1_u2 = 7;
    lbp_4_a2_u2 = 9;
    lbp_4_ri = 2;
    lbp_4_ur = 3;

    // normal LBP8
    lbp_8_a1 = 30;
    lbp_8_a2 = 225;
    lbp_8_a1_u2 = 29;
    lbp_8_a2_u2 = 33;
    lbp_8_ri = 8;
    lbp_8_ur = 5;

    // normal LBP16
    lbp_16_a1 = 1020;
    lbp_16_a2 = 64515;
    lbp_16_a1_u2 = 120;
    lbp_16_a2_u2 = 128;
    lbp_16_ri = 128;
    lbp_16_ur = 9;

    // direction coded LBP
    lbp_4_a1_d = 10;
    lbp_4_a2_d = 10;
    lbp_8_a1_d = 42;
    lbp_8_a2_d = 162;
    lbp_16_a1_d = 10410;
    lbp_16_a2_d = 34858;

    // transitional LBP
    lbp_4_a1_t = 3;
    lbp_4_a2_t = 12;
    lbp_8_a1_t = 135;
    lbp_8_a2_t = 120;
    lbp_16_a1_t = 32895;
    lbp_16_a2_t = 32640;

  }

  ~T() {}
};

template<typename T, int d>
void check_dimensions( blitz::Array<T,d>& t1, blitz::Array<T,d>& t2)
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T>
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

template<typename T>
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<T,3>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), t2(i,j,k));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_lbp4_1_uint8 )
{
  // LBP 4,1
  bob::ip::LBP lbp(4);

  // rectangular LBP
  BOOST_CHECK_EQUAL( lbp_4_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 16, lbp.getMaxLabel() );

  // circular LBP
  lbp = bob::ip::LBP(4, 1., true);
  BOOST_CHECK_EQUAL( lbp_4_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 16, lbp.getMaxLabel() );

  // to average
  lbp = bob::ip::LBP(4, 1., true, true);
  BOOST_CHECK_EQUAL( lbp_4_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 16, lbp.getMaxLabel() );

  // add average bit
  lbp = bob::ip::LBP(4, 1., true, true, true);
  BOOST_CHECK_EQUAL( lbp_4_a1*2+1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2*2+1, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 32, lbp.getMaxLabel() );

  // uniform LBP
  lbp = bob::ip::LBP(4, 1., true, false, false, true);
  BOOST_CHECK_EQUAL( lbp_4_a1_u2, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2_u2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 15, lbp.getMaxLabel() );

  // rotation invariant LBP
  lbp = bob::ip::LBP(4, 1., true, false, false, false, true);
  BOOST_CHECK_EQUAL( lbp_4_ri, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_ri, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 6, lbp.getMaxLabel() );

  // rotation invariant uniform LBP
  lbp = bob::ip::LBP(4, 1., true, false, false, true, true);
  BOOST_CHECK_EQUAL( lbp_4_ur, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_ur, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 6, lbp.getMaxLabel() );
}

BOOST_AUTO_TEST_CASE( test_lbp8_1_uint8 )
{
  // LBP 8,1
  bob::ip::LBP lbp(8);

  // rectangular LBP
  BOOST_CHECK_EQUAL( lbp_8_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 256, lbp.getMaxLabel() );

  // circular LBP
  lbp = bob::ip::LBP(8, 1., true);
  BOOST_CHECK_EQUAL( lbp_8_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 256, lbp.getMaxLabel() );

  // to average
  lbp = bob::ip::LBP(8, 1., true, true);
  BOOST_CHECK_EQUAL( lbp_8_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 256, lbp.getMaxLabel() );

  // add average bit
  lbp = bob::ip::LBP(8, 1., true, true, true);
  BOOST_CHECK_EQUAL( lbp_8_a1*2+1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2*2+1, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 512, lbp.getMaxLabel() );

  // uniform LBP
  lbp = bob::ip::LBP(8, 1., true, false, false, true);
  BOOST_CHECK_EQUAL( lbp_8_a1_u2, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2_u2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 59, lbp.getMaxLabel() );

  // rotation invariant LBP
  lbp = bob::ip::LBP(8, 1., true, false, false, false, true);
  BOOST_CHECK_EQUAL( lbp_8_ri, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_ri, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 36, lbp.getMaxLabel() );

  // rotation invariant uniform LBP
  lbp = bob::ip::LBP(8, 1., true, false, false, true, true);
  BOOST_CHECK_EQUAL( lbp_8_ur, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_ur, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 10, lbp.getMaxLabel() );
}

BOOST_AUTO_TEST_CASE( test_lbp16_1_uint8 )
{
  // circular LBP 16,1
  bob::ip::LBP lbp(16, 1., true);

  BOOST_CHECK_EQUAL( lbp_16_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 65536, lbp.getMaxLabel() );

  // to average
  lbp = bob::ip::LBP(16, 1., true, true);
  BOOST_CHECK_EQUAL( lbp_16_a1, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_a2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 65536, lbp.getMaxLabel() );

  // add average bit DOESN'T WORK
//  lbp = bob::ip::LBP(16, 1., true, true, true);
//  BOOST_CHECK_EQUAL( lbp_16_a1*2, lbp(a1,1,1) );
//  BOOST_CHECK_EQUAL( lbp_16_a2*2, lbp(a2,1,1) );

  // uniform LBP
  lbp = bob::ip::LBP(16, 1., true, false, false, true);
  BOOST_CHECK_EQUAL( lbp_16_a1_u2, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_a2_u2, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 243, lbp.getMaxLabel() );

  // rotation invariant LBP
  lbp = bob::ip::LBP(16, 1., true, false, false, false, true);
  BOOST_CHECK_EQUAL( lbp_16_ri, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_ri, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 4116, lbp.getMaxLabel() );

  // rotation invariant uniform LBP
  lbp = bob::ip::LBP(16, 1., true, false, false, true, true);
  BOOST_CHECK_EQUAL( lbp_16_ur, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_ur, lbp(a2,1,1) );
  BOOST_CHECK_EQUAL( 18, lbp.getMaxLabel() );
}

BOOST_AUTO_TEST_CASE( test_lbp_image )
{
  // LBP 8,1
  bob::ip::LBP lbp(8);
  blitz::TinyVector<int,2> resolution = lbp.getLBPShape(a1);
  BOOST_CHECK_EQUAL(resolution[0], 1);
  BOOST_CHECK_EQUAL(resolution[1], 1);
  blitz::Array<uint16_t, 2> result(1,1);

  lbp(a1,result);
  BOOST_CHECK_EQUAL( lbp_8_a1, result(0,0) );
  lbp(a2,result);
  BOOST_CHECK_EQUAL( lbp_8_a2, result(0,0) );
}


BOOST_AUTO_TEST_CASE( test_lbp_other )
{
  // direction-coded LBP
  bob::ip::LBP lbp(4, 1., true, false, false, false, false, bob::ip::ELBP_DIRECTION_CODED);
  BOOST_CHECK_EQUAL( lbp_4_a1_d, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2_d, lbp(a2,1,1) );
  lbp = bob::ip::LBP(8, 1., true, false, false, false, false, bob::ip::ELBP_DIRECTION_CODED);
  BOOST_CHECK_EQUAL( lbp_8_a1_d, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2_d, lbp(a2,1,1) );
  lbp = bob::ip::LBP(16, 1., true, false, false, false, false, bob::ip::ELBP_DIRECTION_CODED);
  BOOST_CHECK_EQUAL( lbp_16_a1_d, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_a2_d, lbp(a2,1,1) );

  // transitional LBP
  lbp = bob::ip::LBP(4, 1., true, false, false, false, false, bob::ip::ELBP_TRANSITIONAL);
  BOOST_CHECK_EQUAL( lbp_4_a1_t, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_4_a2_t, lbp(a2,1,1) );
  lbp = bob::ip::LBP(8, 1., true, false, false, false, false, bob::ip::ELBP_TRANSITIONAL);
  BOOST_CHECK_EQUAL( lbp_8_a1_t, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_8_a2_t, lbp(a2,1,1) );
  lbp = bob::ip::LBP(16, 1., true, false, false, false, false, bob::ip::ELBP_TRANSITIONAL);
  BOOST_CHECK_EQUAL( lbp_16_a1_t, lbp(a1,1,1) );
  BOOST_CHECK_EQUAL( lbp_16_a2_t, lbp(a2,1,1) );
}

BOOST_AUTO_TEST_SUITE_END()
