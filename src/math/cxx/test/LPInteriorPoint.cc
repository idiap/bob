/**
 * @file math/cxx/test/LPInteriorPoint.cc
 * @date Thu Mar 31 14:32:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the interior point methods to solve LP
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
#define BOOST_TEST_MODULE math-interiorpoint Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <bob/core/cast.h>
#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <bob/core/array_type.h>
#include <bob/math/linear.h>
#include <bob/math/LPInteriorPoint.h>

struct T {
  double eps;

  T():eps(1e-4) {}

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
    BOOST_CHECK_EQUAL(t1(i), bob::core::cast<T>(t2(i)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,2>& t1, blitz::Array<U,2>& t2)
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      BOOST_CHECK_EQUAL(t1(i,j), bob::core::cast<T>(t2(i,j)));
}

template<typename T, typename U>  
void checkBlitzEqual( blitz::Array<T,3>& t1, blitz::Array<U,3>& t2) 
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    for( int j=0; j<t1.extent(1); ++j)
      for( int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_EQUAL(t1(i,j,k), bob::core::cast<T>(t2(i,j,k)));
}

template<typename T>  
void checkBlitzClose( blitz::Array<T,1>& t1, blitz::Array<T,1>& t2, 
  const double eps )
{
  check_dimensions( t1, t2);
  for( int i=0; i<t1.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs( t2(i)-t1(i) ), eps);
}

void generateProblem( const int n, blitz::Array<double,2>& A,
  blitz::Array<double,1>& b, blitz::Array<double,1>& c, 
  blitz::Array<double,1>& x0)
{
  A.resize(n, 2*n);
  b.resize(n);
  c.resize(2*n);
  x0.resize(2*n);
  A = 0.;
  c = 0.;
  for( int i=0; i<n; ++i) {
    A(i,i) = 1.;
    A(i,n+i) = 1.;
    for( int j=i+1; j<n; ++j)
      A(j,i) = pow(2., 1+j);
    b(i) = pow(5.,i+1);
    c(i) = -pow(2., n-1-i);
    x0(i) = 1.;
  }
  blitz::Array<double,2> A1 = A(blitz::Range::all(), blitz::Range( 0, n-1));
  blitz::Array<double,1> ones(n);
  ones = 1.;
  blitz::Array<double,1> A1_1(n);
  bob::math::prod(A1, ones, A1_1);
  for( int i=0; i<n; ++i) {
    x0(n+i) = b(i) - A1_1(i);
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_solve )
{
  blitz::Array<double,2> A;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;
  blitz::Array<double,1> x0;
  blitz::Array<double,1> sol;

  // Check problem 1 for dimension 1 to 10
  for (int n=1; n<=10; ++n)
  {
    generateProblem(n, A, b, c, x0);
    sol.resize(n);
    sol = 0.;
    sol(n-1) = pow(5., n); // Solution to problem 1 is [0 ... 0 5^n] 

    // Short step
    blitz::Array<double,1> x2(bob::core::array::ccopy(x0));
    bob::math::LPInteriorPointShortstep solver2(n, 2*n, 0.4, 1e-6);
    solver2.solve(A, b, c, x2);
    for( int i=0; i<n; ++i)
      BOOST_CHECK_SMALL( fabs(x2(i)-sol(i)), eps);

    // Predictor corrector
    blitz::Array<double,1> x3(bob::core::array::ccopy(x0));
    bob::math::LPInteriorPointPredictorCorrector solver3(n, 2*n, 0.5, 0.25, 1e-6);
    solver3.solve(A, b, c, x3);
    for( int i=0; i<n; ++i)
      BOOST_CHECK_SMALL( fabs( x3(i)-sol(i)), eps);

    // Long step
    blitz::Array<double,1> x4(bob::core::array::ccopy(x0));
    bob::math::LPInteriorPointLongstep solver4(n, 2*n, 1e-3, 0.1, 1e-6);
    solver4.solve(A, b, c, x4);
    for( int i=0; i<n; ++i)
      BOOST_CHECK_SMALL( fabs( x4(i)-sol(i)), eps);
  }
}
 


BOOST_AUTO_TEST_CASE( test_detail_neighborhood )
{
  // Test math::detail::isFeasible
  blitz::Array<double,2> A;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;
  blitz::Array<double,1> x0;
  const int n=2;
  generateProblem(n, A, b, c, x0);

  blitz::Array<double,1> x(4), lambda(2), mu1(4), mu2(4);
  x = 1.0532, 1.4341, 3.9468, 19.3532;
  lambda = -1.5172, -3.2922;
  mu1 = 12.6859, 2.2922, 1.5172, 3.2922;
  mu2 = 12.6859, 2.2922, 1.5172, 7.2922;
  bob::math::LPInteriorPointShortstep op(A.extent(0), A.extent(1), 0.5, eps);
  BOOST_CHECK_EQUAL( true, 
    op.isFeasible(A, b, c, x, lambda, mu1) );
  BOOST_CHECK_EQUAL( false, 
    op.isFeasible(A, b, c, x, lambda, mu2) );

  // Test math::detail::isInVinf
  const double gamma1=1e-3, gamma2=1;
  bob::math::LPInteriorPointLongstep op2(A.extent(0), A.extent(1), gamma1, gamma2, eps);
  BOOST_CHECK_EQUAL( true,
    op2.isInV(x, mu1, gamma1) );
  BOOST_CHECK_EQUAL( false,
    op2.isInV(x, mu1, gamma2) );

  // Test math::detail::isInV2
  const double theta1=0.5, theta2=0.03;
  blitz::Array<double,1> x3(4), mu3(4);
  x3 = 0.5562, 21.5427, 4.4438, 1.2327;
  mu3 = 2.4616, 0.0506, 0.2590, 1.0506;
  BOOST_CHECK_EQUAL( true,
    op.isInV(x3, mu3, theta1) );
  BOOST_CHECK_EQUAL( false,
    op.isInV(x3, mu3, theta2) );
}

BOOST_AUTO_TEST_CASE( test_dual_variables_init )
{
  blitz::Array<double,2> A(2,4);
  A =  1., 0., 1., 0., 4., 1., 0., 1.;
  blitz::Array<double,1> c(4);
  c = -2., -1., 0., 0.;

  // Initialize lambda and mu
  bob::math::LPInteriorPointShortstep op(A.extent(0), A.extent(1), 0.5, 1e-6);
  op.initializeDualLambdaMu(A, c);
  blitz::Array<double,1> lambda = op.getLambda();
  blitz::Array<double,1> mu = op.getMu();
  // Check that the point found fulfill the constraints:
  //  mu>=0 and transpose(A)*lambda+mu=c
  for (int i=0; i<mu.extent(0); ++i)
    BOOST_CHECK_EQUAL( true, mu(i) >= 0);
  //  transpose(A)*lambda+mu=c
  blitz::Array<double,2> A_t = A.transpose(1,0);
  blitz::Array<double,1> left_vec(A_t.extent(0));
  bob::math::prod( A_t, lambda, left_vec);
  left_vec += mu;
  for (int i=0; i<c.extent(0); ++i)
    BOOST_CHECK_SMALL( fabs(left_vec(i)-c(i)), eps);
}

BOOST_AUTO_TEST_SUITE_END()

