/**
 * @file src/cxx/math/test/interiorpoint.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the interior point methods to solve LP
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE math-interiorpoint Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/logging.h"
#include "math/interiorpoint.h"

#include <iostream>

struct T {
  T() {}

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
    BOOST_CHECK_EQUAL(t1(i), Torch::core::cast<T>(t2(i)));
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


BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_solve_3x3 )
{
  blitz::Array<double,2> A(1,3);
  A = 1, 1, 1;
  blitz::Array<double,1> b(1);
  b = 1;
  blitz::Array<double,1> c(3);
  c = 1, 2, 3;

  blitz::Array<double,1> x(3);
  x = 0.6, 0.2, 0.2;
  blitz::Array<double,1> lambda(1);
  lambda = 0;
  blitz::Array<double,1> mu(3);
  mu = 1, 2, 3;

  Torch::math::interiorpointShortstep(A, b, c, 0.4, x, lambda, mu, 1e-6);
  std::cout << x << std::endl;

//  Torch::math::interiorpointPredictorCorrector(A, b, c, 0.5, 0.25, x, lambda, mu, 1e-6);
//  std::cout << x << std::endl;

  Torch::math::interiorpointLongstep(A, b, c, 1e-3, 0.1, x, lambda, mu, 1e-6);
  std::cout << x << std::endl;
}
  
BOOST_AUTO_TEST_SUITE_END()

