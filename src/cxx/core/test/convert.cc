/**
 * @file src/cxx/core/test/convert.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the convert function for blitz arrays of different types
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Core-BlitzArray Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <iostream>
#include "core/logging.h"
#include "core/convert.h"
#include "core/cast.h"

struct T {
  blitz::Array<uint8_t,1> a8;
  blitz::Array<uint16_t,1> a16;
  T(): a8(3), a16(3) {
    a8 = 0, 127, 255;
    a16 = 0, 32639, 65535;
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
    BOOST_CHECK_EQUAL( t1(i), Torch::core::cast<T>(t2(i)) );
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

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,4>& t1, blitz::Array<U,4>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        for( int l=0; l<t1.extent(3); ++l)
          BOOST_CHECK_EQUAL(t1(i,j,k,l), Torch::core::cast<T>(t2(i,j,k,l)) );
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************************** ALLOCATION TESTS ******************************/
BOOST_AUTO_TEST_CASE( test_convert_uint8_to_uint16 )
{
  blitz::Array<uint16_t,1> b = Torch::core::convert<uint16_t,uint8_t>(a8);
  checkBlitzEqual( b, a16);

  blitz::Array<uint16_t,1> c = Torch::core::convert<uint16_t,uint8_t>(a8,0,255,0,255);
  checkBlitzEqual( c, a8);

  blitz::Array<uint16_t,1> d = Torch::core::convertFromRange<uint16_t,uint8_t>(a8,0,255);
  checkBlitzEqual( d, a16);

  blitz::Array<uint16_t,1> e = Torch::core::convertToRange<uint16_t,uint8_t>(a8,0,255);
  checkBlitzEqual( e, a8);
}

BOOST_AUTO_TEST_SUITE_END()

