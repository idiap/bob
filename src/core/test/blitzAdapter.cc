/**
 * @file blitzAdapter.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the blitz adapter 
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Core-Tensor Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "core/Tensor.h"
#include "core/BlitzAdapter.h"
#include <iostream>
#include <fstream>


struct T {
  typedef blitz::Array<double,2> BAd2;
  typedef blitz::Array<int,3> BAi3;
  typedef blitz::Array<double,3> BAd3;
  BAd2 bl1, bl2;
  BAi3 bl3, bl4;
  BAd3 bl5;

  T(): bl1(3,5), bl3(2,2,2) {
    bl1 = 1.1, 0, 0, 1, 5,  
          1.3, 2, 3, 4, 5,  
          6.5, 7, 8, 9, 10;
    
    bl3 = 5, 4,   6, 7,
          11, -6, 37, 54;
  }

  ~T() {}

};

template<typename BA1, typename BA2>  void check_dimensions( BA1& t1, BA2& t2) {
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for( int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename BA1, typename BA2>  void check_equal2d( BA1& t1, BA2& t2) {
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), t2(i,j));
}

template<typename BA1, typename BA2>  void check_equal3d( BA1& t1, BA2& t2) {
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), t2(i,j,k));
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

//this will save, load and compare two blitz arrays
BOOST_AUTO_TEST_CASE( test_write_read )
{
  // 1/ 2D double array
  // Save
  std::ofstream out_d("test_adapter_BAd2.blitz");
  Torch::core::BlitzAdapter<BAd2> X(bl1);
  out_d << X;
  out_d.close();
  // Load
  std::ifstream in_d("test_adapter_BAd2.blitz");
  Torch::core::BlitzAdapter<BAd2> Y(bl2);
  in_d >> Y;
  in_d.close();
  // Compare
  check_equal2d( bl1, bl2);

  // 2/ 3D int array
  // Save
  std::ofstream out_i("test_adapter_BAi3.blitz");
  Torch::core::BlitzAdapter<BAi3> Z(bl3);
  out_i << Z;
  out_i.close();
  // Load
  std::ifstream in_i("test_adapter_BAi3.blitz");
  Torch::core::BlitzAdapter<BAi3> T(bl4);
  in_i >> T;
  in_i.close();
  // Compare
  check_equal3d( bl3, bl4);

}

//this will save, read a converted blitz array and compare it to the original
BOOST_AUTO_TEST_CASE( test_write_convertread )
{
  // 1/ 3D array: Load an int array into a double array
  // Save
  std::ofstream out_i("test_adapter_BAi3.blitz");
  Torch::core::BlitzAdapter<BAi3> X(bl3);
  out_i << X;
  out_i.close();
  // Load
  std::ifstream in_d("test_adapter_BAi3.blitz");
  Torch::core::BlitzAdapter<BAd3> Y(bl5, false);
  in_d >> Y;
  in_d.close();
  // Compare
  check_equal3d( bl3, bl5);
}


BOOST_AUTO_TEST_SUITE_END()

