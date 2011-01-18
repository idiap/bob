/**
 * @file src/core/test/blitzArray.cc
 * @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Extensive Blitz Array tests 
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Core-BlitzArray Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>


struct T {
  T() {}
  ~T() {}
};

void checkBlitzAllocation( const int n_megabytes ) {
  // Dimensions of the blitz::Array
  int n_elems_first = n_megabytes*1024;
  int n_elems_second = 1024;

  // Create the blitz::Array
  blitz::Array<int8_t,2> X(n_elems_first,n_elems_second);

  // Check X.numElements equals n_elems_first * n_elems_second 
  // careful: use a 64 bit type to store the result)
  int64_t n_e = (int64_t)n_elems_first*(int64_t)n_elems_second;
#if !(defined(__LP64__) || defined(__APPLE__))
  BOOST_CHECK_(n_e != (int64_t)X.numElements() );
#else
  BOOST_CHECK_EQUAL(n_e, (int64_t)X.numElements() );
#endif

#ifdef blitzArrayFullTest
  // Check X.extent(blitz::firstDim) equals n_elems_first
  BOOST_CHECK_EQUAL(n_elems_first, X.extent(blitz::firstDim));
  // Check X.extent(blitz::secondDim) equals n_elems_second
  BOOST_CHECK_EQUAL(n_elems_second, X.extent(blitz::secondDim));

  for(int i=0; i<n_elems_first; ++i)
    for(int j=0; j<n_elems_second; ++j) {
      int8_t tmp = j % 37 % 37;
      // Make sure no exceptions are thrown
      BOOST_CHECK_NO_THROW(X(i,j) = tmp);
      // Check the validity of the value stored in the array
      BOOST_CHECK_EQUAL(X(i,j), tmp);
    }
#endif
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_blitz_array_512MB )
{
  checkBlitzAllocation( 512 );
}

BOOST_AUTO_TEST_CASE( test_blitz_array_1024MB )
{
  checkBlitzAllocation( 1024 );
}

BOOST_AUTO_TEST_CASE( test_blitz_array_1536MB )
{
  checkBlitzAllocation( 1536 );
}

BOOST_AUTO_TEST_CASE( test_blitz_array_2048MB )
{
  checkBlitzAllocation( 2048 );
}

BOOST_AUTO_TEST_CASE( test_blitz_array_2560MB )
{
  checkBlitzAllocation( 2560 );
}

BOOST_AUTO_TEST_CASE( test_blitz_array_3072MB )
{
  checkBlitzAllocation( 3072 );
}

BOOST_AUTO_TEST_SUITE_END()

