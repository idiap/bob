/**
 * @file ip/cxx/test/GaussianScaleSpace.cc
 * @date Mon Sep 3 11:32:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Gaussian Scale Space
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
#define BOOST_TEST_MODULE IP-GaussianScaleSpace Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <bob/ip/GaussianScaleSpace.h>

struct T {
  blitz::Array<double,2> src;
  blitz::Array<double,2> dst;

  double eps;

  T(): src(3,4), dst(6,8), eps(1e-4)
  {
    src = 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12;
    dst = 1, 1.5, 2, 2.5, 3, 3.5, 4, 4,
          3, 3.5, 4, 4.5, 5, 5.5, 6, 6,
          5, 5.5, 6, 6.5, 7, 7.5, 8, 8,
          7, 7.5, 8, 8.5, 9, 9.5, 10, 10,
          9, 9.5, 10, 10.5, 11, 11.5, 12, 12,
          9, 9.5, 10, 10.5, 11, 11.5, 12, 12;
  }
  
  ~T() {}
};

template<typename T>  
void checkBlitzClose( const blitz::Array<T,1>& t1, const blitz::Array<T,1>& t2,
  const double eps )
{
  BOOST_CHECK_EQUAL( t1.extent(0), t2.extent(0) );
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(t1(i)-t2(i)), eps);
}

template<typename T>  
void checkBlitzClose( const blitz::Array<T,2>& t1, const blitz::Array<T,2>& t2,
  const double eps )
{
  BOOST_CHECK_EQUAL( t1.extent(0), t2.extent(0) );
  BOOST_CHECK_EQUAL( t1.extent(1), t2.extent(1) );
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_SMALL( fabs(t1(i,j)-t2(i,j)), eps);
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_upsample_downsample )
{
  blitz::Array<double,2> d(6,8);
  bob::ip::detail::upsample(src, d);
  checkBlitzClose( dst, d, eps);

  blitz::Array<double,2> dsrc(3,4);
  bob::ip::detail::downsample(d, dsrc, 1);
  checkBlitzClose( dsrc, src, eps);
}
 
BOOST_AUTO_TEST_SUITE_END()
