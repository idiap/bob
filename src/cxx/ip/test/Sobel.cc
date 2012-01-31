/**
 * @file cxx/ip/test/Sobel.cc
 * @date Fri Apr 29 12:13:22 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Sobel operator for 2D arrays/images
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE IP-Sobel Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/cast.h"
#include "ip/Sobel.h"

struct T {
  double eps;

  blitz::Array<double,2> x;
  blitz::Array<double,3> x_sol;
  T(): eps(1e-3), x(3,3), x_sol(2,1,1)
  {
    x = 0, 1, 2, 3, 4, 5, 6, 7, 8;
    x_sol = -24, -8;
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


template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  int y_min = std::min( t1.extent(0), t2.extent(0));
  int x_min = std::min( t1.extent(1), t2.extent(1));
  BOOST_CHECK_LE( abs(t1.extent(0)-t2.extent(0)), 1);
  BOOST_CHECK_LE( abs(t1.extent(1)-t2.extent(1)), 1);
  double diff = 0.;
  for( int i=0; i<y_min; ++i)
    for( int j=0; j<x_min; ++j)
      diff += abs( t1(i,j) - t2(i,j) );
  diff = (diff/(y_min*x_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  BOOST_CHECK_SMALL( diff, eps );
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,3>& t1, blitz::Array<T,3>& t2, 
  const double eps )
{
  int p_min = std::min( t1.extent(0), t2.extent(0));
  int y_min = std::min( t1.extent(1), t2.extent(1));
  int x_min = std::min( t1.extent(2), t2.extent(2));
  BOOST_CHECK_EQUAL( t1.extent(0), t2.extent(0) );
  BOOST_CHECK_LE( abs(t1.extent(1)-t2.extent(1)), 1);
  BOOST_CHECK_LE( abs(t1.extent(2)-t2.extent(2)), 1);
  double diff = 0.;
  for( int i=0; i<p_min; ++i)
    for( int j=0; j<y_min; ++j)
      for( int k=0; k<x_min; ++k)
        diff += abs( t1(i,j,k) - t2(i,j,k) );
  diff = (diff/(y_min*x_min*p_min)) / 
    (std::numeric_limits<T>::max()-std::numeric_limits<T>::min()+1);
  BOOST_CHECK_SMALL( diff, eps );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_sobel_2d )
{
  blitz::Array<double,3> y(2,1,1); 
  bob::ip::Sobel s(false,false,bob::sp::Convolution::Valid);
  s(x,y);
  checkBlitzClose(y, x_sol, eps);
}

BOOST_AUTO_TEST_SUITE_END()
