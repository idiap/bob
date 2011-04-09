/**
 * @file src/cxx/ip/test/shift.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the shift function for 2D and 3D arrays/images
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-Shift Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/cast.h"
#include "ip/shift.h"

struct T {
  blitz::Array<uint32_t,2> a2, a2s_1, a2s_2;
  blitz::Array<uint32_t,3> a3, a3s_1, a3s_2;

  T(): a2(4,4), a2s_1(4,4), a2s_2(4,4), a3(3,4,4), a3s_1(3,4,4), a3s_2(3,4,4) 
  {
    a2 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15;

    a2s_1 = 9, 10, 11, 0, 13, 14, 15, 0,
        0, 0, 0, 0, 0, 0, 0, 0;

    a2s_2 = 9, 10, 11, 11, 13, 14, 15, 15,
        13, 14, 15, 15, 13, 14, 15, 15;

    a3 = 0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47;

    a3s_1 = 9, 10, 11, 0, 13, 14, 15, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        25, 26, 27, 0, 29, 30, 31, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        41, 42, 43, 0, 45, 46, 47, 0,
        0, 0, 0, 0, 0, 0, 0, 0;

    a3s_2 = 9, 10, 11, 11, 13, 14, 15, 15,
        13, 14, 15, 15, 13, 14, 15, 15,
        25, 26, 27, 27, 29, 30, 31, 31,
        29, 30, 31, 31, 29, 30, 31, 31,
        41, 42, 43, 43, 45, 46, 47, 47,
        45, 46, 47, 47, 45, 46, 47, 47;
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
      BOOST_CHECK_EQUAL(t1(i,j), Torch::core::cast<T>(t2(i,j)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), Torch::core::cast<T>(t2(i,j,k)));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_shift_2d_uint8 )
{
  blitz::Array<uint32_t,2> b2;
  // "No" shift +0 +0
  Torch::ip::shift(a2, b2, 0, 0);
  checkBlitzEqual(a2, b2); 

  // Shift fully out and check exception
  BOOST_CHECK_THROW( Torch::ip::shift(a2, b2, 4, 0),
    Torch::ip::ParamOutOfBoundaryError );

  // Shift +2y +1x (fill with zero)
  Torch::ip::shift(a2, b2, 2, 1, false, true);
  checkBlitzEqual(a2s_1, b2); 

  // Shift +2y +1x (fill with closest neighbour)
  Torch::ip::shift(a2, b2, 2, 1);
  checkBlitzEqual(a2s_2, b2); 
}
  
BOOST_AUTO_TEST_CASE( test_shift_3d_uint8 )
{
  blitz::Array<uint32_t,3> b3;
  // "No" shift +0 +0 
  Torch::ip::shift(a3, b3, 0, 0);
  checkBlitzEqual(a3, b3); 

  // Shift fully out and check exception
  BOOST_CHECK_THROW( Torch::ip::shift(a3, b3, 4, 0),
    Torch::ip::ParamOutOfBoundaryError );

  // Shift +2y +1x (fill with zero)
  Torch::ip::shift(a3, b3, 2, 1, false, true);
  checkBlitzEqual(a3s_1, b3);

  // Shift +2y +1x (fill with closest neighbour)
  Torch::ip::shift(a3, b3, 2, 1);
  checkBlitzEqual(a3s_2, b3); 
}

BOOST_AUTO_TEST_SUITE_END()
