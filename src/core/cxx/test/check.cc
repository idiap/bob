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
#include <bob/core/array_copy.h>
#include <bob/core/check.h>
#include <bob/core/cast.h>
#include <vector>
#include <map>

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

BOOST_AUTO_TEST_CASE( test_check_equal )
{
  // Integer arrays
  BOOST_CHECK(  bob::core::array::isEqual( g, h) );
  BOOST_CHECK( !bob::core::array::isEqual( g, i) );
  BOOST_CHECK( !bob::core::array::isEqual( g, j) );
}

BOOST_AUTO_TEST_CASE( test_check_close )
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
}
  
BOOST_AUTO_TEST_CASE( test_check_equal_vector_map )
{
  blitz::Array<uint8_t,1> x1a(4);
  x1a = 1, 2, 3, 4;
  blitz::Array<uint8_t,1> x1b(4);
  x1b = 1, 2, 3, 4;
  blitz::Array<uint8_t,1> x2a(5);
  x2a = 1, 2, 3, 4, 5;
  blitz::Array<uint8_t,1> x2b(5);
  x2b = 1, 2, 3, 4, 5;
  blitz::Array<uint8_t,1> x3a(5);
  x3a = 1, 2, 3, 4, 6;
  blitz::Array<uint8_t,1> x3b(5);
  x3b = 1, 2, 3, 4, 7;

  // Vectors
  std::vector<blitz::Array<uint8_t,1> > vec_a, vec_b;
  BOOST_CHECK( bob::core::array::isEqual( vec_a, vec_b) );
  vec_a.push_back(x1a);
  vec_b.push_back(x1b);
  BOOST_CHECK( bob::core::array::isEqual( vec_a, vec_b) );
  vec_a.push_back(x2a);
  vec_b.push_back(x2b);
  BOOST_CHECK( bob::core::array::isEqual( vec_a, vec_b) );
  vec_a.push_back(x3a);
  BOOST_CHECK( !bob::core::array::isEqual( vec_a, vec_b) );
  vec_b.push_back(x3a);
  BOOST_CHECK( bob::core::array::isEqual( vec_a, vec_b) );
  vec_a.push_back(x3a);
  vec_b.push_back(x3b);
  BOOST_CHECK( !bob::core::array::isEqual( vec_a, vec_b) );

  // Maps
  std::map<int, blitz::Array<uint8_t,1> > map_a, map_b;
  BOOST_CHECK( bob::core::array::isEqual( map_a, map_b) );
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[1].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( !bob::core::array::isEqual( map_a, map_b) );
  map_a.clear();
  map_b.clear();
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[0].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( bob::core::array::isEqual( map_a, map_b) );
  map_a[37].reference(bob::core::array::ccopy(x2a));
  map_b[37].reference(bob::core::array::ccopy(x2b));
  BOOST_CHECK( bob::core::array::isEqual( map_a, map_b) );
  map_a[73].reference(bob::core::array::ccopy(x3a));
  BOOST_CHECK( !bob::core::array::isEqual( map_a, map_b) );
  map_b[73].reference(bob::core::array::ccopy(x3a));
  BOOST_CHECK( bob::core::array::isEqual( map_a, map_b) );
  map_b[73].reference(bob::core::array::ccopy(x3b));
  BOOST_CHECK( !bob::core::array::isEqual( map_a, map_b) );
}


BOOST_AUTO_TEST_CASE( test_check_close_vector_map )
{
  blitz::Array<double,1> x1a(4);
  x1a = 1, 2, 3, 4.0000000001;
  blitz::Array<double,1> x1b(4);
  x1b = 1, 2, 3, 4;
  blitz::Array<double,1> x2a(5);
  x2a = 1, 2, 3, 4, 5.0000000001;
  blitz::Array<double,1> x2b(5);
  x2b = 1, 2, 3, 4, 5;
  blitz::Array<double,1> x3a(5);
  x3a = 1, 2, 3, 4, 6.;
  blitz::Array<double,1> x3b(5);
  x3b = 1, 2, 3, 4, 7.;

  // Vectors
  std::vector<blitz::Array<double,1> > vec_a, vec_b;
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  vec_a.push_back(x1a);
  vec_b.push_back(x1b);
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  BOOST_CHECK( !bob::core::array::isClose( vec_a, vec_b, 1e-15, 1e-15) );
  vec_a.push_back(x2a);
  vec_b.push_back(x2b);
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  vec_a.push_back(x3a);
  BOOST_CHECK( !bob::core::array::isClose( vec_a, vec_b) );
  vec_b.push_back(x3a);
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  vec_a.push_back(x3a);
  vec_b.push_back(x3b);
  BOOST_CHECK( !bob::core::array::isClose( vec_a, vec_b) );

  // Maps
  std::map<int, blitz::Array<double,1> > map_a, map_b;
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[1].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b) );
  map_a.clear();
  map_b.clear();
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[0].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b, 1e-15, 1e-15) );
  map_a[37].reference(bob::core::array::ccopy(x2a));
  map_b[37].reference(bob::core::array::ccopy(x2b));
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  map_a[73].reference(bob::core::array::ccopy(x3a));
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b) );
  map_b[73].reference(bob::core::array::ccopy(x3a));
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  map_b[73].reference(bob::core::array::ccopy(x3b));
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b) );
}

BOOST_AUTO_TEST_CASE( test_check_close_vector_map_complex )
{
  blitz::Array<std::complex<double>,1> x1a(2);
  x1a(0) = std::complex<double>(1., 0.);
  x1a(1) = std::complex<double>(0., 1.);
  blitz::Array<std::complex<double>,1> x1b(2);
  x1b(0) = std::complex<double>(1., 0.);
  x1b(1) = std::complex<double>(0., 1.0000000001);

  // Vectors
  std::vector<blitz::Array<std::complex<double>,1> > vec_a, vec_b;
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  vec_a.push_back(x1a);
  vec_b.push_back(x1b);
  BOOST_CHECK( bob::core::array::isClose( vec_a, vec_b) );
  BOOST_CHECK( !bob::core::array::isClose( vec_a, vec_b, 1e-15, 1e-15) );
 
  // Maps
  std::map<int, blitz::Array<std::complex<double>,1> > map_a, map_b;
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[1].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b) );
  map_a.clear();
  map_b.clear();
  map_a[0].reference(bob::core::array::ccopy(x1a));
  map_b[0].reference(bob::core::array::ccopy(x1b));
  BOOST_CHECK( bob::core::array::isClose( map_a, map_b) );
  BOOST_CHECK( !bob::core::array::isClose( map_a, map_b, 1e-15, 1e-15) );

  // Vectors of scalars
  std::vector<std::complex<double> > vec_aa, vec_bb;
  BOOST_CHECK( bob::core::isClose( vec_aa, vec_bb) );
  vec_aa.push_back(x1a(0));
  vec_aa.push_back(x1a(1));
  vec_bb.push_back(x1b(0));
  vec_bb.push_back(x1b(1));
  BOOST_CHECK( !bob::core::isClose( vec_aa, vec_bb, 1e-15, 1e-15) );
  BOOST_CHECK( bob::core::isClose( vec_aa, vec_bb) );

  // Maps of scalars
  std::map<int, std::complex<double> > map_aa, map_bb;
  BOOST_CHECK( bob::core::isClose( map_aa, map_bb) );
  map_aa[0] = x1a(0);
  map_aa[2] = x1a(1);
  map_bb[1] = x1b(0);
  map_bb[2] = x1b(1);
  BOOST_CHECK( !bob::core::isClose( map_aa, map_bb) );
  map_aa.clear();
  map_bb.clear();
  map_aa[0] = x1a(0);
  map_aa[2] = x1a(1);
  map_bb[0] = x1b(0);
  map_bb[2] = x1b(1);
  BOOST_CHECK( bob::core::isClose( map_aa, map_bb) );
  BOOST_CHECK( !bob::core::isClose( map_aa, map_bb, 1e-15, 1e-15) );
}

BOOST_AUTO_TEST_SUITE_END()

