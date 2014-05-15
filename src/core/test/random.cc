/**
 * @date Thu Mar 3 20:17:53 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the random functions
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE core-random Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <bob/core/check.h>
#include <bob/core/array_random.h>

struct T {
};

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_blitz_array_random )
{
  blitz::Array<double,2> x(1000,1000);
  boost::mt19937 rng;
  const double mean_ref = 1.;
  const double std_ref = 2.;
  bob::core::array::randn(rng, x, mean_ref, std_ref);
  double mean = blitz::mean(x);
  double std = sqrt(blitz::mean(blitz::pow2(x - mean)));
  BOOST_CHECK( bob::core::isClose( mean, mean_ref, 1e-2, 1e-2) );
  BOOST_CHECK( bob::core::isClose( std, std_ref, 1e-2, 1e-2) );
}

BOOST_AUTO_TEST_SUITE_END()
