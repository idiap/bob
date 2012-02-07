/**
 * @file cxx/ip/test/shear.cc
 * @date Wed Mar 9 19:09:08 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the shear function for 2D and 3D arrays/images
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
#define BOOST_TEST_MODULE IP-Shear Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <random/uniform.h>
#include <stdint.h>
#include "core/cast.h"
#include "ip/shear.h"

struct T {
  blitz::Array<uint32_t,2> a2, a2sX_p27, a2sX_m27, a2sY_p27, a2sY_m27;
  blitz::Array<bool,2> b2sX_m27;

  T(): a2(8,8), a2sX_p27(8,10), a2sX_m27(8,10), a2sY_p27(10,8), a2sY_m27(10,8),
       b2sX_m27(8,10)
  {
    a2 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63;

    a2sX_p27 = 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
              0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
              0, 16, 17, 18, 19, 20, 21, 22, 23, 0,
              0, 24, 25, 26, 27, 28, 29, 30, 31, 0,
              0, 32, 33, 34, 35, 36, 37, 38, 39, 0,
              0, 40, 41, 42, 43, 44, 45, 46, 47, 0,
              48, 49, 50, 51, 52, 53, 54, 55, 0, 0,
              56, 57, 58, 59, 60, 61, 62, 63, 0, 0;

    a2sX_m27 = 0, 1, 2, 3, 4, 5, 6, 7, 0, 0,
              8, 9, 10, 11, 12, 13, 14, 15, 0, 0,
              0, 16, 17, 18, 19, 20, 21, 22, 23, 0,
              0, 24, 25, 26, 27, 28, 29, 30, 31, 0,
              0, 32, 33, 34, 35, 36, 37, 38, 39, 0,
              0, 40, 41, 42, 43, 44, 45, 46, 47, 0,
              0, 0, 48, 49, 50, 51, 52, 53, 54, 55,
              0, 0, 56, 57, 58, 59, 60, 61, 62, 63;

    a2sY_p27 = 0, 0, 0, 0, 0, 0, 6, 7,
               0, 0, 2, 3, 4, 5, 14, 15,
               0, 1, 10, 11, 12, 13, 22, 23,
               8, 9, 18, 19, 20, 21, 30, 31,
               16, 17, 26, 27, 28, 29, 38, 39,
               24, 25, 34, 35, 36, 37, 46, 47,
               32, 33, 42, 43, 44, 45, 54, 55,
               40, 41, 50, 51, 52, 53, 62, 63,
               48, 49, 58, 59, 60, 61, 0, 0,
               56, 57, 0, 0, 0, 0, 0, 0;

    a2sY_m27 = 0, 1, 0, 0, 0, 0, 0, 0, 
               8, 9, 2, 3, 4, 5, 0, 0,
               16, 17, 10, 11, 12, 13, 6, 7,
               24, 25, 18, 19, 20, 21, 14, 15,
               32, 33, 26, 27, 28, 29, 22, 23,
               40, 41, 34, 35, 36, 37, 30, 31,
               48, 49, 42, 43, 44, 45, 38, 39,
               56, 57, 50, 51, 52, 53, 46, 47,
               0, 0, 58, 59, 60, 61, 54, 55,
               0, 0, 0, 0, 0, 0, 62, 63;

    b2sX_m27 = false, false, true, true, true, true, true, true, true, true,
               false, false, true, true, true, true, true, true, true, true,
               false, true, true, true, true, true, true, true, true, false,
               false, true, true, true, true, true, true, true, true, false,
               false, true, true, true, true, true, true, true, true, false,
               false, true, true, true, true, true, true, true, true, false,
               true, true, true, true, true, true, true, true, false, false,
               true, true, true, true, true, true, true, true, false, false;
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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_shearX_2d_uint32 )
{
  blitz::Array<double,2> b2;
  // X-axis Shear +2px
  b2.resize(bob::ip::getShearXShape(a2, 2./7.));
  bob::ip::shearX( a2, b2, 2./7., false);
  checkBlitzEqual( a2sX_p27, b2);
  // X-axis Shear -2px
  b2.resize(bob::ip::getShearXShape(a2, -2./7.));
  bob::ip::shearX( a2, b2, -2./7., false);
  checkBlitzEqual( a2sX_m27, b2);
}

BOOST_AUTO_TEST_CASE( test_shearY_2d_uint32 )
{
  blitz::Array<double,2> b2;  
  // Y-axis Shear +2px
  b2.resize(bob::ip::getShearYShape(a2, 2./7.));
  bob::ip::shearY( a2, b2, 2./7., false);
  checkBlitzEqual( a2sY_p27, b2);
  // Y-axis Shear -2px
  b2.resize(bob::ip::getShearYShape(a2, -2./7.));
  bob::ip::shearY( a2, b2, -2./7., false);
  checkBlitzEqual( a2sY_m27, b2);
}

BOOST_AUTO_TEST_CASE( test_shearX_2d_uint32_mask )
{
  blitz::Array<double,2> b2;  
  blitz::Array<bool,2> a2_mask(8,8), b2_mask(8,10);
  a2_mask = true;
  // X-axis Shear +2px
  b2.resize(bob::ip::getShearXShape(a2, 2./7.));
  bob::ip::shearX( a2, a2_mask, b2, b2_mask, 2./7., false);
  checkBlitzEqual( a2sX_p27, b2);
  checkBlitzEqual( b2sX_m27, b2_mask);
}

BOOST_AUTO_TEST_CASE( test_shearXY_2d_double_random )
{
  // Generate a random array
  int m = 2;
  int Nx = 137;
  int Ny = 223;
  blitz::Array<double,2> in2(Ny,Nx);
  ranlib::Uniform<double> gen;
  for(int i=0; i<Ny; ++i)
    for(int j=0; j<Nx; ++j)
      in2(i,j) = gen.random();

  // Declare arrays
  blitz::Array<double,2> tmp2, out2, out2_crop;  

  // X-axis Shear +m px
  tmp2.resize(bob::ip::getShearXShape(in2, (double)m/(Ny-1)));
  bob::ip::shearX( in2, tmp2, (double)m/(Ny-1), false);
  // X-axis Shear -m px
  out2.resize(bob::ip::getShearXShape(tmp2, -(double)m/(Ny-1)));
  bob::ip::shearX( tmp2, out2, -(double)m/(Ny-1), false);
  // Crop and compare with input
  out2_crop.reference( out2( blitz::Range::all(), blitz::Range(m,m+Nx-1) ) );
  checkBlitzEqual( in2, out2_crop);
  
  // Y-axis Shear +m px
  tmp2.resize(bob::ip::getShearYShape(in2, (double)m/(Nx-1)));
  bob::ip::shearY( in2, tmp2, (double)m/(Nx-1), false);
  // X-axis Shear -m px
  out2.resize(bob::ip::getShearYShape(tmp2, -(double)m/(Nx-1)));
  bob::ip::shearY( tmp2, out2, -(double)m/(Nx-1), false);
  // Crop and compare with input
  out2_crop.reference( out2( blitz::Range(m,m+Ny-1), blitz::Range::all() ) );
  checkBlitzEqual( in2, out2_crop);  
}

BOOST_AUTO_TEST_SUITE_END()
