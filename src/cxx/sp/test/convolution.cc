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

#include "sp/convolution.h"

namespace conv = Torch::sp::Convolution;

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

 
  blitz::Array<double,2> A2_5;
  blitz::Array<double,2> b2_2;
  blitz::Array<double,2> b2_3;

  blitz::Array<double,2> res_A2_5_b2_2_full;
  blitz::Array<double,2> res_A2_5_b2_2_same;
  blitz::Array<double,2> res_A2_5_b2_2_valid;

  blitz::Array<double,2> res_A2_5_b2_3_full;
  blitz::Array<double,2> res_A2_5_b2_3_same;
  blitz::Array<double,2> res_A2_5_b2_3_valid;


  blitz::Array<double,1> A1b_5;
  blitz::Array<double,1> b1b_3;

  blitz::Array<double,1> res_A1b_5_b1b_3_full_zero;
  blitz::Array<double,1> res_A1b_5_b1b_3_full_near;
  blitz::Array<double,1> res_A1b_5_b1b_3_full_circ;
  blitz::Array<double,1> res_A1b_5_b1b_3_full_mirr;
  blitz::Array<double,1> res_A1b_5_b1b_3_same_zero;
  blitz::Array<double,1> res_A1b_5_b1b_3_same_near;
  blitz::Array<double,1> res_A1b_5_b1b_3_same_circ;
  blitz::Array<double,1> res_A1b_5_b1b_3_same_mirr;
  blitz::Array<double,1> res_A1b_5_b1b_3_valid;

  blitz::Array<double,2> A2b_3x4;
  blitz::Array<double,2> b2b_2x2;

  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_full_zero;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_full_near;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_full_circ;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_full_mirr;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_same_zero;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_same_near;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_same_circ;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_same_mirr;
  blitz::Array<double,2> res_A2b_3x4_b2b_2x2_valid;

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

    A2_5.resize(5,5);
    A2_5 = 0.8407, 0.3500, 0.3517, 0.2858, 0.0759,
      0.2543, 0.1966, 0.8308, 0.7572, 0.0540,
      0.8143, 0.2511, 0.5853, 0.7537, 0.5308,
      0.2435, 0.6160, 0.5497, 0.3804, 0.7792,
      0.9293, 0.4733, 0.9172, 0.5678, 0.9340;

    b2_2.resize(2,2);
    b2_2 = 0.1299, 0.4694,
      0.5688, 0.0119;

    b2_3.resize(3,3);
    b2_3 = 0.3371, 0.3112, 0.6020,
      0.1622, 0.5285, 0.2630,
      0.7943, 0.1656, 0.6541;

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

    res_A2_5_b2_2_full.resize(6,6);
    res_A2_5_b2_2_full = 0.1092, 0.4401, 0.2100, 0.2022, 0.1440, 0.0356,
      0.5113, 0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
      0.2504, 0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
      0.4948, 0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
      0.2592, 0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
      0.5286, 0.2803, 0.5274, 0.3339, 0.5380, 0.0111;

    res_A2_5_b2_2_same.resize(5,5);
    res_A2_5_b2_2_same = 0.3540, 0.4044, 0.6551, 0.4090, 0.0262,
      0.5297, 0.6688, 0.8132, 0.4624, 0.2498,
      0.3469, 0.6965, 0.7432, 0.5907, 0.3721,
      0.8510, 0.6613, 0.7272, 0.8356, 0.4477,
      0.2803, 0.5274, 0.3339, 0.5380, 0.0111;

    res_A2_5_b2_2_valid.resize(4,4);
    res_A2_5_b2_2_valid = 0.3540, 0.4044, 0.6551, 0.4090,
      0.5297, 0.6688, 0.8132, 0.4624,
      0.3469, 0.6965, 0.7432, 0.5907,
      0.8510, 0.6613, 0.7272, 0.8356;

    res_A2_5_b2_3_full.resize(7,7);
    res_A2_5_b2_3_full = 
      0.2834, 0.3796, 0.7336, 0.4165, 0.3262, 0.1957, 0.0457,
      0.2221, 0.6465, 0.9574, 0.9564, 1.0098, 0.5879, 0.0524,
      0.9835, 0.9216, 1.9583, 1.7152, 1.7309, 1.0461, 0.3833,
      0.4161, 0.9528, 1.8242, 2.0354, 2.0621, 1.4545, 0.6439,
      0.9995, 1.0117, 2.5338, 2.1359, 2.4450, 1.7253, 1.1143,
      0.3441, 1.0976, 1.3412, 1.4975, 1.7343, 1.0209, 0.7553,
      0.7381, 0.5299, 1.4147, 0.9125, 1.4358, 0.5261, 0.6109;

    res_A2_5_b2_3_same.resize(5,5);
    res_A2_5_b2_3_same = 0.6465, 0.9574, 0.9564, 1.0098, 0.5879,
      0.9216, 1.9583, 1.7152, 1.7309, 1.0461,
      0.9528, 1.8242, 2.0354, 2.0621, 1.4545,
      1.0117, 2.5338, 2.1359, 2.4450, 1.7253,
      1.0976, 1.3412, 1.4975, 1.7343, 1.0209;

    res_A2_5_b2_3_valid.resize(3,3);
    res_A2_5_b2_3_valid = 1.9583, 1.7152, 1.7309,
      1.8242, 2.0354, 2.0621,
      2.5338, 2.1359, 2.4450;

    A1b_5.resize(5);
    A1b_5 = 0, 1, 2, 3, 4;
    b1b_3.resize(3);
    b1b_3 = 3, 1, 2;

    res_A1b_5_b1b_3_full_zero.resize(7);
    res_A1b_5_b1b_3_full_zero = 0, 3, 7, 13, 19, 10, 8;
    res_A1b_5_b1b_3_full_near.resize(7);
    res_A1b_5_b1b_3_full_near = 0, 3, 7, 13, 19, 22, 24;
    res_A1b_5_b1b_3_full_circ.resize(7);
    res_A1b_5_b1b_3_full_circ = 10, 11, 7, 13, 19, 10, 11;
    res_A1b_5_b1b_3_full_mirr.resize(7);
    res_A1b_5_b1b_3_full_mirr = 2, 3, 7, 13, 19, 22, 21;
    res_A1b_5_b1b_3_same_zero.resize(5);
    res_A1b_5_b1b_3_same_zero = 3, 7, 13, 19, 10;
    res_A1b_5_b1b_3_same_near.resize(5);
    res_A1b_5_b1b_3_same_near = 3, 7, 13, 19, 22;
    res_A1b_5_b1b_3_same_circ.resize(5);
    res_A1b_5_b1b_3_same_circ = 11, 7, 13, 19, 10;
    res_A1b_5_b1b_3_same_mirr.resize(5);
    res_A1b_5_b1b_3_same_mirr = 3, 7, 13, 19, 22;
    res_A1b_5_b1b_3_valid.resize(3);
    res_A1b_5_b1b_3_valid = 7, 13, 19;
    
    A2b_3x4.resize(3,4);
    A2b_3x4 = 0, 1, 2, 3, 
      4, 5, 6, 7, 
      8, 9, 10, 11;
    b2b_2x2.resize(2,2);
    b2b_2x2 = 4, 1, 3, 2;

    res_A2b_3x4_b2b_2x2_full_zero.resize(4,5);
    res_A2b_3x4_b2b_2x2_full_zero = 0, 4, 9, 14, 3,
      16, 27, 37, 47, 13,
      44, 67, 77, 87, 25,
      24, 43, 48, 53, 22;
    res_A2b_3x4_b2b_2x2_full_near.resize(4,5);
    res_A2b_3x4_b2b_2x2_full_near = 0, 7, 17, 27, 30,
      20, 27, 37, 47, 50,
      60, 67, 77, 87, 90,
      80, 87, 97, 107, 110;
    res_A2b_3x4_b2b_2x2_full_circ.resize(4,5);
    res_A2b_3x4_b2b_2x2_full_circ = 49, 47, 57, 67, 49,
      29, 27, 37, 47, 29,
      69, 67, 77, 87, 69,
      49, 47, 57, 67, 49;
    res_A2b_3x4_b2b_2x2_full_mirr.resize(4,5);
    res_A2b_3x4_b2b_2x2_full_mirr = 0, 7, 17, 27, 30,
      20, 27, 37, 47, 50,
      60, 67, 77, 87, 90,
      80, 87, 97, 107, 110;
    res_A2b_3x4_b2b_2x2_same_zero.resize(3,4);
    res_A2b_3x4_b2b_2x2_same_zero = 27, 37, 47, 13,
      67, 77, 87, 25,
      43, 48, 53, 22;
    res_A2b_3x4_b2b_2x2_same_near.resize(3,4);
    res_A2b_3x4_b2b_2x2_same_near = 27, 37, 47, 50,
      67, 77, 87, 90,
      87, 97, 107, 110;
    res_A2b_3x4_b2b_2x2_same_circ.resize(3,4);
    res_A2b_3x4_b2b_2x2_same_circ = 27, 37, 47, 29,
      67, 77, 87, 69,
      47, 57, 67, 49;
    res_A2b_3x4_b2b_2x2_same_mirr.resize(3,4);
    res_A2b_3x4_b2b_2x2_same_mirr = 27, 37, 47, 50,
      67, 77, 87, 90,
      87, 97, 107, 110;
    res_A2b_3x4_b2b_2x2_valid.resize(2,3);
    res_A2b_3x4_b2b_2x2_valid = 27, 37, 47,
      67, 77, 87;
    
  }
  ~T() {}
};


template <typename T> 
void test_convolve_1D_nopt( T eps, const blitz::Array<T,1>& a1, 
  const blitz::Array<T,1>& a2, const blitz::Array<T,1>& mat)
{
  blitz::Array<T,1> res;
  Torch::sp::convolve( a1, a2, res);
  for(int i=0; i<res.extent(0); ++i)
    BOOST_CHECK_SMALL(res(i) - mat(i), eps);
}

template <typename T> 
void test_convolve_2D_nopt( T eps, const blitz::Array<T,2>& a1, 
  const blitz::Array<T,2>& a2, const blitz::Array<T,2>& mat)
{
  blitz::Array<T,2> res;
  Torch::sp::convolve( a1, a2, res);
  for(int i=0; i<res.extent(0); ++i)
    for(int j=0; j<res.extent(1); ++j)
      BOOST_CHECK_SMALL(res(i,j) - mat(i,j), eps);
}

template <typename T> 
void test_convolve_1D( T eps, const blitz::Array<T,1>& a1, 
  const blitz::Array<T,1>& a2, const blitz::Array<T,1>& mat, 
  const enum conv::SizeOption opt1 = conv::Full,
  const enum conv::BorderOption opt2 = conv::Zero)
{
  blitz::Array<T,1> res;
  Torch::sp::convolve( a1, a2, res, opt1, opt2);
  for(int i=0; i<res.extent(0); ++i)
    BOOST_CHECK_SMALL(res(i) - mat(i), eps);
}

template <typename T> 
void test_convolve_2D( T eps, const blitz::Array<T,2>& a1, 
  const blitz::Array<T,2>& a2, const blitz::Array<T,2>& mat,
  const enum conv::SizeOption opt1 = conv::Full,
  const enum conv::BorderOption opt2 = conv::Zero)
{
  blitz::Array<T,2> res;
  Torch::sp::convolve( a1, a2, res, opt1, opt2);
  for(int i=0; i<res.extent(0); ++i)
    for(int j=0; j<res.extent(1); ++j)
      BOOST_CHECK_SMALL(res(i,j) - mat(i,j), eps);
}


BOOST_FIXTURE_TEST_SUITE( test_setup, T )
// The following tests compare results from Torch and Matlab.

// 1D convolution between a 1D vector of length 10 and 3 (no option)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_nopt )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_full);
}

// 1D convolution between a 1D vector of length 10 and 3 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_full )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_full, conv::Full);
}

// 1D convolution between a 1D vector of length 10 and 3 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_same )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_same, conv::Same);
}

// 1D convolution between a 1D vector of length 10 and 3 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_3_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_3, res_A1_10_b1_3_valid, conv::Valid);
}

// 1D convolution between a 1D vector of length 10 and 4 (no option)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_nopt )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_full);
}

// 1D convolution between a 1D vector of length 10 and 4 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_full )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_full, conv::Full);
}

// 1D convolution between a 1D vector of length 10 and 4 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_same )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_same, conv::Same);
}

// 1D convolution between a 1D vector of length 10 and 4 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_4_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_4, res_A1_10_b1_4_valid, conv::Valid);
}

// 1D convolution between a 1D vector of length 10 and 5 (no option)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_nopt )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_full);
}

// 1D convolution between a 1D vector of length 10 and 5 (full)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_full )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_full, conv::Full);
}

// 1D convolution between a 1D vector of length 10 and 5 (same)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_same )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_same, conv::Same);
}

// 1D convolution between a 1D vector of length 10 and 5 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_1D_10_5_valid )
{
  test_convolve_1D( eps_d, A1_10, b1_5, res_A1_10_b1_5_valid, conv::Valid);
}

// 2D convolution between a 2D vector of length 5x5 and 2x2 (no option)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_2_nopt )
{
  test_convolve_2D( eps_d, A2_5, b2_2, res_A2_5_b2_2_full);
}

// 2D convolution between a 2D vector of length 5x5 and 2x2 (full)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_2_full )
{
  test_convolve_2D( eps_d, A2_5, b2_2, res_A2_5_b2_2_full, conv::Full);
}

// 2D convolution between a 2D vector of length 5x5 and 2x2 (same)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_2_same )
{
  test_convolve_2D( eps_d, A2_5, b2_2, res_A2_5_b2_2_same, conv::Same);
}

// 2D convolution between a 2D vector of length 5x5 and 2x2 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_2_valid )
{
  test_convolve_2D( eps_d, A2_5, b2_2, res_A2_5_b2_2_valid, conv::Valid);
}

// 2D convolution between a 2D vector of length 5x5 and 3x3 (no option)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_3_nopt )
{
  test_convolve_2D( eps_d, A2_5, b2_3, res_A2_5_b2_3_full);
}

// 2D convolution between a 2D vector of length 5x5 and 3x3 (full)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_3_full )
{
  test_convolve_2D( eps_d, A2_5, b2_3, res_A2_5_b2_3_full, conv::Full);
}

// 2D convolution between a 2D vector of length 5x5 and 3x3 (same)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_3_same )
{
  test_convolve_2D( eps_d, A2_5, b2_3, res_A2_5_b2_3_same, conv::Same);
}

// 2D convolution between a 2D vector of length 5x5 and 3x3 (valid)
BOOST_AUTO_TEST_CASE( test_convolve_2D_5_3_valid )
{
  test_convolve_2D( eps_d, A2_5, b2_3, res_A2_5_b2_3_valid, conv::Valid);
}

// 1D convolution between a 1D vector of length 5 and 3
BOOST_AUTO_TEST_CASE( test_convolve_1D_5_3 )
{
  // Full size
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_full_zero, 
    conv::Full, conv::Zero);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_full_near, 
    conv::Full, conv::NearestNeighbour);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_full_circ, 
    conv::Full, conv::Circular);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_full_mirr, 
    conv::Full, conv::Mirror);

  // Same size
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_same_zero, 
    conv::Same, conv::Zero);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_same_near, 
    conv::Same, conv::NearestNeighbour);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_same_circ, 
    conv::Same, conv::Circular);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_same_mirr, 
    conv::Same, conv::Mirror);

  // Valid size
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_valid, 
    conv::Valid, conv::Zero);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_valid, 
    conv::Valid, conv::NearestNeighbour);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_valid, 
    conv::Valid, conv::Circular);
  test_convolve_1D( eps_d, A1b_5, b1b_3, res_A1b_5_b1b_3_valid, 
    conv::Valid, conv::Mirror);
}

// 2D convolution between a 2D vector of length 3x4 and 2x2
BOOST_AUTO_TEST_CASE( test_convolve_2D_3x4_2x2 )
{
  // Full size
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_full_zero, 
    conv::Full, conv::Zero);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_full_near, 
    conv::Full, conv::NearestNeighbour);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_full_circ, 
    conv::Full, conv::Circular);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_full_mirr, 
    conv::Full, conv::Mirror);

  // Same size
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_same_zero, 
    conv::Same, conv::Zero);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_same_near, 
    conv::Same, conv::NearestNeighbour);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_same_mirr, 
    conv::Same, conv::Mirror);

  // Valid size
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_valid, 
    conv::Valid, conv::Zero);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_valid, 
    conv::Valid, conv::NearestNeighbour);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_valid, 
    conv::Valid, conv::Circular);
  test_convolve_2D( eps_d, A2b_3x4, b2b_2x2, res_A2b_3x4_b2b_2x2_valid, 
    conv::Valid, conv::Mirror);
}

BOOST_AUTO_TEST_SUITE_END()
