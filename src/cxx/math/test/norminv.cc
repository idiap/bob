/**
 * @file cxx/math/test/norminv.cc
 * @date Tue Apr 12 21:33:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the inverse normal cumulative distribution
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE math-norminv Tests
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include "math/norminv.h"

namespace math = Torch::math;

struct T {
  double sols_d05, sols_d25, sols_d50, sols_d75, sols_d95;
  double sol_m2s4_d37, sol_m2s4_d48, sol_m2s4_d79;
  double eps;

  T(): sols_d05(-1.64485362695), sols_d25(-0.674489750196), sols_d50(0.), 
    sols_d75(0.674489750196), sols_d95(1.64485362695), 
    sol_m2s4_d37(0.672586614252), sol_m2s4_d48(1.799385666141), 
    sol_m2s4_d79(5.225684988072), eps(1e-5)
  { }

  ~T() {}
};

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_normsinv )
{
  // Compute and compare again OpenOffice! reference values
  BOOST_CHECK_SMALL( fabs( sols_d05 - math::normsinv(0.05)), eps);
  BOOST_CHECK_SMALL( fabs( sols_d25 - math::normsinv(0.25)), eps);
  BOOST_CHECK_SMALL( fabs( sols_d50 - math::normsinv(0.5)), eps);
  BOOST_CHECK_SMALL( fabs( sols_d75 - math::normsinv(0.75)), eps);
  BOOST_CHECK_SMALL( fabs( sols_d95 - math::normsinv(0.95)), eps);
}

BOOST_AUTO_TEST_CASE( test_norminv )
{
  // Compute and compare again OpenOffice! reference values
  BOOST_CHECK_SMALL( fabs( sol_m2s4_d37 - math::norminv(0.37, 2., 4.)), eps);
  BOOST_CHECK_SMALL( fabs( sol_m2s4_d48 - math::norminv(0.48, 2., 4.)), eps);
  BOOST_CHECK_SMALL( fabs( sol_m2s4_d79 - math::norminv(0.79, 2., 4.)), eps);
}

BOOST_AUTO_TEST_SUITE_END()
