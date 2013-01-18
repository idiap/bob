/**
 * @file math/cxx/test/norm.cc
 * @date Thu Jan 17 17:50:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the Frobenius norm
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
#define BOOST_TEST_MODULE math-norm Tests
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "bob/math/norm.h"

struct T {
  blitz::Array<double,2> A24, B33;
  blitz::Array<int,2> C32;
  double sol_a, sol_b, sol_c;
  double tolerance;

  T(): A24(2,4), B33(3,3), C32(3,2),
    sol_a(14.282856857085701), sol_b(16.881943016134130), 
    sol_c(9.539392014169458),
    tolerance(1e-8) // Tolerance in percentage
  { 
    A24 =  1,  2,  3,  4,  5,  6,  7,  8;
    B33 =  1,  2,  3,  4,  5,  6,  7,  8,  9;
    C32 = -1,  2, -3,  4, -5,  6;
  }

  ~T() {}
};

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_norm )
{
  // Computes and compares again Matlab reference values
  BOOST_CHECK_CLOSE( sol_a, bob::math::frobenius(A24), tolerance);
  BOOST_CHECK_CLOSE( sol_b, bob::math::frobenius(B33), tolerance);
  BOOST_CHECK_CLOSE( sol_c, bob::math::frobenius(C32), tolerance);
}

BOOST_AUTO_TEST_SUITE_END()
