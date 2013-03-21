/**
 * @file core/cxx/test/repmat.cc
 * @date Fri Jul 15 18:50:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the repmat function for 1D and 2D blitz arrays
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
#define BOOST_TEST_MODULE Core-repmat Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <bob/core/repmat.h>

struct T {
  blitz::Array<uint32_t,2> a, a33_s;
  blitz::Array<uint32_t,1> b, b3_s, b4_s;
  blitz::Array<uint32_t,2> b23_s, b23_s_row;
  T(): a(2,3), a33_s(6,9), b(2), b3_s(6), b4_s(8), b23_s(4,3), b23_s_row(2,6) {
    a = 1, 2, 3, 4, 5, 6;
    a33_s = 1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        4, 5, 6, 4, 5, 6, 4, 5, 6;

    b = 1, 2;
    b3_s = 1, 2, 1, 2, 1, 2;
    b4_s = 1, 1, 1, 1, 2, 2, 2, 2;

    b23_s = 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2;
    b23_s_row = 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2;
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
void checkBlitzEqual( blitz::Array<T,1>& t1, blitz::Array<T,1>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_EQUAL( t1(i), t2(i) );
}

template<typename T>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************************** ALLOCATION TESTS ******************************/
BOOST_AUTO_TEST_CASE( test_repmat_1d )
{
  blitz::Array<uint32_t,2> b23_a(2*b.extent(0), 3);
  bob::core::repmat(b, b23_a);
  checkBlitzEqual(b23_a, b23_s);

  blitz::Array<uint32_t,2> b23_b(b.extent(0), 3*b.extent(0));
  bob::core::repmat(b, b23_b, true);
  checkBlitzEqual(b23_b, b23_s_row);
}

BOOST_AUTO_TEST_CASE( test_repmat_2d )
{
  blitz::Array<uint32_t,2> a33(3*a.extent(0), 3*a.extent(1));
  bob::core::repmat(a, a33);
  checkBlitzEqual(a33, a33_s);
}

BOOST_AUTO_TEST_CASE( test_repvec )
{
  blitz::Array<uint32_t,1> b3(3*b.extent(0));
  bob::core::repvec(b, b3);
  checkBlitzEqual(b3, b3_s);
}

BOOST_AUTO_TEST_CASE( test_repelem )
{
  blitz::Array<uint32_t,1> b4(4*b.extent(0));
  bob::core::repelem(b, b4);
  checkBlitzEqual(b4, b4_s);
}

BOOST_AUTO_TEST_SUITE_END()

