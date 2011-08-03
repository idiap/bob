/**
 * @file src/cxx/trainer/test/jfa.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the jfa trainer
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trainer-jfa Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "core/cast.h"
#include "trainer/JFATrainer.h"
#include "machine/JFAMachine.h"


struct T {
  double eps;

  T(): eps(1e-4) { }
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



BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_estimateXandU )
{
  blitz::Array<double,2> F(4,6);
  F = 0.3833, 0.6173, 0.5755, 0.5301, 0.2751, 0.2486,
      0.4516, 0.2277, 0.8044, 0.9861, 0.0300, 0.5357,
      0.0871, 0.8021, 0.9891, 0.0669, 0.9394, 0.0182,
      0.6838, 0.7837, 0.5341, 0.8854, 0.8990, 0.6259;
  blitz::Array<double,2> N(4,2);
  N = 0.1379, 0.2178, 0.1821, 0.0418, 0.1069, 0.6164, 0.9397, 0.3545;
  blitz::Array<double,1> m(6);
  m = 0.1806, 0.0451, 0.7232, 0.3474, 0.6606, 0.3839;
  blitz::Array<double,1> E(6);
  E = 0.6273, 0.0216, 0.9106, 0.8006, 0.7458, 0.8131;
  blitz::Array<double,1> d(6);
  d = 0.4106, 0.9843, 0.9456, 0.6766, 0.9883, 0.7668;
  blitz::Array<double,2> v(2,6);
  v = 0.3367, 0.6624, 0.2442, 0.2955, 0.6802, 0.5278,
      0.4116, 0.6026, 0.7505, 0.5835, 0.5518, 0.5836;
  blitz::Array<double,2> u(2,6);
  u = 0.5118, 0.0826, 0.7196, 0.9962, 0.3545, 0.9713,
      0.3464, 0.8865, 0.4547, 0.4134, 0.2177, 0.1257;
  blitz::Array<double,2> z(2,6);
  z = 0.3089, 0.7261, 0.7829, 0.6938, 0.0098, 0.8432,
      0.9223, 0.7710, 0.0427, 0.3782, 0.7043, 0.7295;
  blitz::Array<double,2> y(2,2);
  y = 0.2243, 0.2691, 0.6730, 0.4775;
  blitz::Array<double,2> x(4,2);
  blitz::Array<uint32_t,1> spk_ids(4);
  spk_ids = 0, 0, 1, 1;

  // estimateXandU
  Torch::trainer::jfa::estimateXandU(F,N,m,E,d,v,u,z,y,x,spk_ids);

  // JFA cookbook reference
  blitz::Array<double,2> x_ref(4,2);
  x_ref = 0.2143, 3.1979,
      1.8275, 0.1227,
      -1.3861, 5.3326,
      0.2359,  -0.7914;

  checkBlitzClose(x, x_ref, eps);
}

BOOST_AUTO_TEST_SUITE_END()
