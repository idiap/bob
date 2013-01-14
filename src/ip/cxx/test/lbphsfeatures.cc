/**
 * @file ip/cxx/test/lbphsfeatures.cc
 * @date Wed Apr 27 20:40:12 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the LBP feature extractor for 2D arrays/images
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
#define BOOST_TEST_MODULE IP-LBPHSFeatures Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <vector>

#include "bob/core/cast.h"
#include "bob/ip/LBPHSFeatures.h"

struct T {
  blitz::Array<uint32_t,2> src;
  blitz::Array<uint64_t,1> dst1, dst2, dst3, dst4;
  std::vector<blitz::Array<uint64_t,1> > dst_mat;
  double eps;

  T(): src(10,10), dst1(16), dst2(16), dst3(16), dst4(16), dst_mat(0), eps(1e-3)
  {
    src =  0, 10,  9, 10,  0,  5,  6,  7,  8,  9,
          10,  9, 10,  9, 10, 10, 11, 12, 13, 14,
           9, 10,  9, 10,  9, 15, 16, 17, 18, 19, 
          10,  9, 10,  9, 10, 20, 21, 22, 23, 24,
           0, 10,  9, 10,  0, 25, 26, 27, 28, 29,
          20, 20, 20, 20, 20,  5, 22, 23, 22, 54, 
          20, 20, 20, 20, 20, 33, 51,  6, 19, 94, 
          20, 20, 20, 20, 20, 19,  7, 81, 53, 14, 
          20, 20, 20, 20, 20, 56, 11, 18,  3, 64, 
          20, 20, 20, 20, 20, 20,  5, 17,  9, 34; 

    dst1 = 0;
    dst1(0) = 4;
    dst1(15) = 5;
    dst2 = 0;
    dst2(6) = 9;
    dst3 = 0;
    dst3(15) = 9;
    dst4 = 0;
    dst4(0) = 2;
    dst4(1) = 1;
    dst4(5) = 1;
    dst4(8) = 1;
    dst4(14) = 1;
    dst4(15) = 3;

    dst_mat.push_back(dst1);
    dst_mat.push_back(dst2);
    dst_mat.push_back(dst3);
    dst_mat.push_back(dst4);
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

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_lbphs_feature_extract )
{
  std::vector<blitz::Array<uint64_t,1> > dst;
  bob::ip::LBPHSFeatures lbphsfeatures( 5, 5, 0, 0, 1., 4, false, false, 
    false, false, false);

  lbphsfeatures( src, dst);
  // Iterate over the blocks and compare the vector of LBP coefficients with 
  // the one obtained by hand
  int i=0;
  for( std::vector<blitz::Array<uint64_t,1> >::const_iterator it = dst.begin();
    it != dst.end(); ++it)
  {
    checkBlitzClose( *it, dst_mat[i], eps);
    ++i;
  }
}
  
BOOST_AUTO_TEST_SUITE_END()
