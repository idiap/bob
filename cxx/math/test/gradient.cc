/**
 * @file cxx/math/test/gradient.cc
 * @date Sun Apr 15 13:40:23 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the gradient computation for 1D/2D arrays
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
#define BOOST_TEST_MODULE math-gradient Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include "core/cast.h"
#include "math/gradient.h"

struct T {

  double eps;

	blitz::Array<double,1> src_a;
	blitz::Array<double,2> src_b;
	blitz::Array<double,3> src_c;

	blitz::Array<double,1> dst_a1;
	blitz::Array<double,1> dst_a2;
	blitz::Array<double,2> dst_by1;
	blitz::Array<double,2> dst_bx1;
	blitz::Array<double,2> dst_by2;
	blitz::Array<double,2> dst_bx2;
	blitz::Array<double,3> dst_cz1;
	blitz::Array<double,3> dst_cy1;
	blitz::Array<double,3> dst_cx1;

	T(): eps(1e-10), src_a(6), src_b(2,3), src_c(2,3,2),
       dst_a1(6), dst_a2(6), 
       dst_by1(2,3), dst_bx1(2,3), dst_by2(2,3), dst_bx2(2,3),
       dst_cz1(2,3,2), dst_cy1(2,3,2), dst_cx1(2,3,2)
	{
		src_a = 1., 2., 4., 7., 11., 16.;
    src_b = 1., 2., 6., 3., 4., 5.;
    src_c = 3., 5., 1., 2., 3., 7., 7., 8., 1., 3., 6., 9.;

    dst_a1 = 1., 1.5, 2.5, 3.5, 4.5, 5.;
    dst_a2 = 0.5, 0.75, 1.25, 1.75, 2.25, 2.5;
    
    dst_by1 = 2., 2., -1., 2., 2., -1.;
    dst_bx1 = 1., 2.5, 4., 1., 1., 1.;
    dst_by2 = 4., 4., -2, 4., 4., -2.;
    dst_bx2 = 0.5, 1.25, 2., 0.5, 0.5, 0.5;

    dst_cz1 = 4., 3., 0., 1., 3., 2., 4., 3., 0., 1., 3., 2.;
    dst_cy1 = -2., -3., 0., 1., 2., 5., -6., -5., -0.5, 0.5, 5., 6.;
    dst_cx1 = 2., 2., 1., 1., 4., 4., 1., 1., 2., 2., 3., 3.;
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

template<typename T>  
void checkBlitzClose( blitz::Array<T,1>& t1, blitz::Array<T,1>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs( t2(i)-t1(i) ), eps);
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,2>& t1, blitz::Array<T,2>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs( t2(i,j)-t1(i,j) ), eps);
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,3>& t1, blitz::Array<T,3>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_SMALL( fabs( t2(i,j,k)-t1(i,j,k) ), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_gradient_1d )
{
	blitz::Array<double,1> dst(6);

  // Process src_a 
	bob::math::gradient_(src_a, dst);
	checkBlitzClose(dst, dst_a1, eps);

	bob::math::gradient_(src_a, dst, 2.);
	checkBlitzClose(dst, dst_a2, eps);
}

BOOST_AUTO_TEST_CASE( test_gradient_2d )
{
	blitz::Array<double,2> dst_x(2,3), dst_y(2,3);

  // Process src_b 
	bob::math::gradient_(src_b, dst_y, dst_x);
	checkBlitzClose(dst_y, dst_by1, eps);
	checkBlitzClose(dst_x, dst_bx1, eps);

	bob::math::gradient_(src_b, dst_y, dst_x, 0.5, 2.);
	checkBlitzClose(dst_y, dst_by2, eps);
	checkBlitzClose(dst_x, dst_bx2, eps);
}

BOOST_AUTO_TEST_CASE( test_gradient_3d )
{
	blitz::Array<double,3> dst_z(2,3,2), dst_y(2,3,2), dst_x(2,3,2);

  // Process src_a 
	bob::math::gradient_(src_c, dst_z, dst_y, dst_x);
	checkBlitzClose(dst_z, dst_cz1, eps);
	checkBlitzClose(dst_y, dst_cy1, eps);
	checkBlitzClose(dst_x, dst_cx1, eps);
}

BOOST_AUTO_TEST_SUITE_END()
