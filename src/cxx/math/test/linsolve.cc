/**
 * @file src/cxx/math/test/linsolve.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Test the solver A*x=b
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE math-linsolve Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include "core/cast.h"
#include "math/linsolve.h"


struct T {
  blitz::Array<double,2> A33_1, A33_2;
  blitz::Array<double,1> b3_1, b3_2, s3_1, s3_2;
  double eps;

  T(): A33_1(3,3), A33_2(3,3), b3_1(3), b3_2(3), s3_1(3), s3_2(3), eps(1e-6)
  {
    A33_1 = 1., 0., 0., 0., 1., 0., 0., 0., 1.;
    b3_1 = 7., 5., 3.;
    s3_1 = b3_1;
    A33_2 = 1., 3., 5., 7., 9., 1., 3., 5., 7.;
    b3_2 = 2., 4., 6.;
    s3_2 = 3., -2., 1.;
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
  blitz::Array<double,1> x;

  Torch::math::linsolve(A33_1, x, b3_1);
  checkBlitzClose(s3_1, x, eps); 


  Torch::math::linsolve(A33_2, x, b3_2);
  checkBlitzClose(s3_2, x, eps); 
}
  
BOOST_AUTO_TEST_SUITE_END()

