/**
 * @file src/cxx/sp/test/convolution.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Test the blitz-based convolution product
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE sp-convolution Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "core/Tensor.h"
#include "sp/spDCT.h"
#include "sp/spDFT.h"

#include "sp/convolve.h"
#include <blitz/array.h>

struct T {
  blitz::Array<double,1> A1_10;
  blitz::Array<double,1> b1_3;
  blitz::Array<double,1> b1_4;
  blitz::Array<double,1> b1_5;

  blitz::Array<double,1> res_A1_10_b1_3_full;
  blitz::Array<double,1> res_A1_10_b1_3_same;
  blitz::Array<double,1> res_A1_10_b1_3_valid;

  blitz::Array<double,1> res_A1_10_b1_4_full;
  blitz::Array<double,1> res_A1_10_b1_4_same;
  blitz::Array<double,1> res_A1_10_b1_4_valid;

  blitz::Array<double,1> res_A1_10_b1_5_full;
  blitz::Array<double,1> res_A1_10_b1_5_same;
  blitz::Array<double,1> res_A1_10_b1_5_valid;

  double eps_d;
  T(): eps_d(1e-3) {
    // The following arrays have been generated randomly.
    A1_10.resize(10);
    A1_10 = 0.7094, 0.7547, 0.2760, 0.6797, 0.6551, 0.1626, 0.1190, 0.4984, 
      0.9597, 0.3404;
    
    b1_3.resize(3);
    b1_3 = 0.5853, 0.2238, 0.7513;
    b1_4.resize(4);
    b1_4 = 0.2551, 0.5060, 0.6991, 0.8909;
    b1_5.resize(5);
    b1_5 = 0.9593, 0.5472, 0.1386, 0.1493, 0.2575;

    // Reference results have been produced using matlab conv and conv2 
    // functions.
    res_A1_10_b1_3_full.resize(12);
    res_A1_10_b1_3_full = 0.4152, 0.6005, 0.8634, 1.0266, 0.7429, 0.7524, 
      0.5982, 0.4405, 0.7626, 0.7884, 0.7972, 0.2557;

    res_A1_10_b1_3_same.resize(10);
    res_A1_10_b1_3_same = 0.6005, 0.8634, 1.0266, 0.7429, 0.7524, 0.5982, 
      0.4405, 0.7626, 0.7884, 0.7972;

    res_A1_10_b1_3_valid.resize(8);
    res_A1_10_b1_3_valid = 0.8634, 1.0266, 0.7429, 0.7524, 0.5982, 0.4405, 
      0.7626, 0.7884;

    res_A1_10_b1_4_full.resize(13);
    res_A1_10_b1_4_full = 0.1810, 0.5514, 0.9482, 1.4726, 1.3763, 1.0940,
      1.1761, 0.8846, 0.7250, 1.0268, 1.2872, 1.0930, 0.3033;

    res_A1_10_b1_4_same.resize(10);
    res_A1_10_b1_4_same = 0.9482, 1.4726, 1.3763, 1.0940, 1.1761, 0.8846,
      0.7250, 1.0268, 1.2872, 1.0930;

    res_A1_10_b1_4_valid.resize(7);
    res_A1_10_b1_4_valid = 1.4726, 1.3763, 1.0940, 1.1761, 0.8846, 0.7250,
      1.0268;

    res_A1_10_b1_5_full.resize(14);
    res_A1_10_b1_5_full = 0.6805, 1.1122, 0.7761, 1.0136, 1.3340, 0.8442, 
      0.4665, 0.8386, 1.4028, 0.9804, 0.4244, 0.3188, 0.2980, 0.0877;

    res_A1_10_b1_5_same.resize(10);
    res_A1_10_b1_5_same = 0.7761, 1.0136, 1.3340, 0.8442, 0.4665, 0.8386,
      1.4028, 0.9804, 0.4244, 0.3188;

    res_A1_10_b1_5_valid.resize(6);
    res_A1_10_b1_5_valid = 1.3340, 0.8442, 0.4665, 0.8386, 1.4028, 0.9804;
  }
  ~T() {}
};


template <typename T> 
void test_convolve_1D( T eps, const blitz::Array<T,1>& a1, 
  const blitz::Array<T,1>& a2, const blitz::Array<T,1>& mat, size_t option = 0)
{
  blitz::Array<T,1> res = Torch::sp::convolve( a1, a2, option);
  for(int i=0; i<res.extent(0); ++i)
    BOOST_CHECK_SMALL(res(i) - mat(i), eps);
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )
// The following tests compare results from Torch and Matlab.

// 1D convolution between a 1D vector of length 10 and 3 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_full )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_full);
}

// 1D convolution between a 1D vector of length 10 and 3 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_same )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_same, 1);
}

// 1D convolution between a 1D vector of length 10 and 3 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_valid, 2);
}

// 1D convolution between a 1D vector of length 10 and 4 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_full )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_full);
}

// 1D convolution between a 1D vector of length 10 and 4 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_same )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_same, 1);
}

// 1D convolution between a 1D vector of length 10 and 4 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_valid, 2);
}

// 1D convolution between a 1D vector of length 10 and 5 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_full )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_full);
}

// 1D convolution between a 1D vector of length 10 and 5 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_same )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_same, 1);
}

// 1D convolution between a 1D vector of length 10 and 5 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_valid, 2);
}

BOOST_AUTO_TEST_SUITE_END()
