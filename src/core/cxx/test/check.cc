/**
 * @file core/cxx/test/check.cc
 * @date Thu Mar 3 20:17:53 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the check function
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
#define BOOST_TEST_MODULE core-check Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <iostream>
#include <bob/core/logging.h>
#include <bob/core/check.h>
#include <bob/core/cast.h>

struct T {
  double x, y;
  float z, t, u;
  blitz::Array<double,1> a, b, c;
  blitz::Array<std::complex<double>,1> d, e, f;
  blitz::Array<uint8_t,1> g, h, i, j;
  T(): x(1e-5), y(x+1e-8),
    z(1.f), t(1.f), u(1.1f),
    a(2), b(2), c(2),
    d(1), e(1), f(1),
    g(2), h(2), i(2), j(3)
  {
    a = 0., 1.;
    b = 0.+1e-12, 1.+1e-12;
    c = 0.5, 1.5;
    d = std::complex<double>(0.,0.);
    e = std::complex<double>(1e-10,1e-10);
    f = std::complex<double>(1e-10,1.);
    g = 0, 1;
    h = 0, 1;
    i = 0, 2;
    j = 0, 1, 2;
  }
  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions( blitz::Array<T,d>& t1, blitz::Array<U,d>& t2) 
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,1>& t1, blitz::Array<U,1>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_EQUAL( t1(i), bob::core::cast<T>(t2(i)) );
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<U,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), bob::core::cast<T>(t2(i,j,k)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,4>& t1, blitz::Array<U,4>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        for( int l=0; l<t1.extent(3); ++l)
          BOOST_CHECK_EQUAL(t1(i,j,k,l), bob::core::cast<T>(t2(i,j,k,l)) );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************************** ALLOCATION TESTS ******************************/
BOOST_AUTO_TEST_CASE( test_check )
{
  // Floating point scalars
  BOOST_CHECK(  bob::core::isClose( x, y, 1e-5, 1e-8) );
  BOOST_CHECK( !bob::core::isClose( x, y, 1e-5, 1e-9) );
  BOOST_CHECK(  bob::core::isClose( z, t, 1e-5f, 1e-8f) );
  BOOST_CHECK( !bob::core::isClose( z, u, 1e-5f, 1e-8f) );

  // Floating point arrays
  BOOST_CHECK(  bob::core::array::isClose( a, b, 1e-5, 1e-8) );
  BOOST_CHECK( !bob::core::array::isClose( a, c, 1e-5, 1e-8) );
  BOOST_CHECK( !bob::core::array::isClose( b, c, 1e-5, 1e-8) );

  // Complex arrays
  BOOST_CHECK(  bob::core::array::isClose( d, e, 1e-5, 1e-8) );
  BOOST_CHECK( !bob::core::array::isClose( d, f, 1e-5, 1e-8) );
  BOOST_CHECK( !bob::core::array::isClose( e, f, 1e-5, 1e-8) );
  
  // Integer arrays
  BOOST_CHECK(  bob::core::array::isEqual( g, h) );
  BOOST_CHECK( !bob::core::array::isEqual( g, i) );
  BOOST_CHECK( !bob::core::array::isEqual( g, j) );
}

BOOST_AUTO_TEST_SUITE_END()

