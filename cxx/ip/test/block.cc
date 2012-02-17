/**
 * @file cxx/ip/test/block.cc
 * @date Tue Apr 5 12:38:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the block decomposition function for 2D arrays/images
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
#define BOOST_TEST_MODULE IP-Block Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <vector>
#include "core/logging.h"
#include "ip/block.h"


struct T {
  blitz::Array<uint32_t,2> a2, a2_a1, a2_a2, a2_a3, a2_a4,
                            a2_b1, a2_b2, a2_b3, a2_b4, a2_b5, a2_b6;
  std::vector<blitz::Array<uint32_t,2> > dst_a, dst_b;

  T(): a2(6,8), a2_a1(3,3), a2_a2(3,3), a2_a3(3,3), a2_a4(3,3),
        a2_b1(3,3), a2_b2(3,3), a2_b3(3,3), a2_b4(3,3), a2_b5(3,3), a2_b6(3,3)
  {
    a2 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47;

    a2_a1 = 0, 1, 2, 8, 9, 10, 16, 17, 18;
    dst_a.push_back( a2_a1);
    a2_a2 = 3, 4, 5, 11, 12, 13, 19, 20, 21;
    dst_a.push_back( a2_a2);
    a2_a3 = 24, 25, 26, 32, 33, 34, 40, 41, 42;
    dst_a.push_back( a2_a3);
    a2_a4 = 27, 28, 29, 35, 36, 37, 43, 44, 45;
    dst_a.push_back( a2_a4);

    a2_b1 = 0, 1, 2, 8, 9, 10, 16, 17, 18;
    dst_b.push_back( a2_b1);
    a2_b2 = 2, 3, 4, 10, 11, 12, 18, 19, 20;
    dst_b.push_back( a2_b2);
    a2_b3 = 4, 5, 6, 12, 13, 14, 20, 21, 22;
    dst_b.push_back( a2_b3);
    a2_b4 = 16, 17, 18, 24, 25, 26, 32, 33, 34;
    dst_b.push_back( a2_b4);
    a2_b5 = 18, 19, 20, 26, 27, 28, 34, 35, 36;
    dst_b.push_back( a2_b5);
    a2_b6 = 20, 21, 22, 28, 29, 30, 36, 37, 38;
    dst_b.push_back( a2_b6);

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
void checkVectorBlitzEqual( std::vector<blitz::Array<T,2> > t1, 
  std::vector<blitz::Array<T,2> > t2)
{
  BOOST_REQUIRE_EQUAL( t1.size(), t2.size() );
  for( unsigned int b=0; b<t1.size(); ++b)
  {
    check_dimensions( t1[b], t2[b]);
    for( int i=0; i<t1[b].extent(0); ++i)
      for( int j=0; j<t1[b].extent(1); ++j)
        BOOST_CHECK_EQUAL(t1[b](i,j), t2[b](i,j));
  }
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_block_2d_nooverlap )
{
  std::vector<blitz::Array<uint32_t,2> > dst; 
  // 2x2 block decomposition without overlap
  bob::ip::blockReference(a2, dst, 3, 3, 0, 0);

  // Compare to reference
  checkVectorBlitzEqual(dst, dst_a); 
}
  
BOOST_AUTO_TEST_CASE( test_block_2d_overlap )
{
  std::vector<blitz::Array<uint32_t,2> > dst; 
  // 2x2 block decomposition without overlap
  bob::ip::blockReference(a2, dst, 3, 3, 1, 1);

  // Compare to reference
  checkVectorBlitzEqual(dst, dst_b); 
}
  
BOOST_AUTO_TEST_SUITE_END()
