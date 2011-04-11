/**
 * @file src/cxx/core/test/blitzArray.cc
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
#include "core/cast.h"
#include "core/array_check.h"
#include "core/logging.h"

#ifdef __APPLE__
# include <sys/types.h>
# include <sys/sysctl.h>
#else
# include <unistd.h>
#endif

struct T {
  double eps;
  T():eps(1e-6) {}
  ~T() {}
};

/**
 * This method will work in UN*X based platforms
 */
size_t maxRAMInMegabytes () {
#ifdef __APPLE__
  int64_t memsize;
  size_t len = sizeof(memsize);
  int mib[] = { CTL_HW, HW_MEMSIZE };
  if (sysctl(mib, 2, &memsize, &len, 0, 0) != 0) return 1024; //returns 1G
  return memsize / (1024 * 1024);
#else
  return sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE) / (1024 * 1024);
#endif
}

void checkBlitzAllocation( const unsigned int n_megabytes ) {
  if (n_megabytes > 0.75*maxRAMInMegabytes()) {
    std::cout << "Warning: Skipping allocation test for " << n_megabytes << " MB because this machine only has " << maxRAMInMegabytes() << " MB of RAM (and we use a max-75% rule)" << std::endl;
    return;
  }

  // Dimensions of the blitz::Array
  int n_elems_first = n_megabytes*1024;
  int n_elems_second = 1024;

  int64_t n_e = (int64_t)n_elems_first*(int64_t)n_elems_second;

  // Create the blitz::Array
  blitz::Array<int8_t,2> X;

// If we can't allocate more than 2GB adresses, throw an exception
#if !((defined(__LP64__) || defined(__APPLE__)) \
  && defined(HAVE_BLITZ_DIFFTYPE))
  static const int64_t TWO_GB = ((int64_t)2*1024)*((int64_t)1024*1024);
  if( n_e < TWO_GB ) {
    // Resize the blitz::Array and check that no exception is thrown
    BOOST_REQUIRE_NO_THROW( X.resize(n_elems_first,n_elems_second) );

    // Check X.numElements equals n_elems_first * n_elems_second 
    // careful: use a 64 bit type to store the result)
    BOOST_CHECK_EQUAL(n_e, (int64_t)X.numElements() );
  }
#else
  // Resize the blitz::Array and check that no exception is thrown
  BOOST_REQUIRE_NO_THROW( X.resize(n_elems_first,n_elems_second) );

  // Check X.numElements equals n_elems_first * n_elems_second 
  // careful: use a 64 bit type to store the result)
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

template <typename T>
void checkBlitzEqual( const blitz::Array<T,1> a, const blitz::Array<T,1> b ) {
  BOOST_REQUIRE_EQUAL( a.extent(0), b.extent(0) );
  for(int i=0; i<a.extent(0); ++i)
    BOOST_CHECK_EQUAL( a(i+a.lbound(0)), b(i+b.lbound(0)) );
}

template <typename T>
void checkBlitzEqual( const blitz::Array<T,2> a, const blitz::Array<T,2> b ) {
  BOOST_REQUIRE_EQUAL( a.extent(0), b.extent(0) );
  BOOST_REQUIRE_EQUAL( a.extent(1), b.extent(1) );
  for(int i=0; i<a.extent(0); ++i)
    for(int j=0; j<a.extent(1); ++j)
      BOOST_CHECK_EQUAL( a(i+a.lbound(0),j+a.lbound(1)), 
        b(i+b.lbound(0),j+b.lbound(1)) );
}

template <typename T>
void checkBlitzEqual( const blitz::Array<T,3> a, const blitz::Array<T,3> b ) {
  BOOST_REQUIRE_EQUAL( a.extent(0), b.extent(0) );
  BOOST_REQUIRE_EQUAL( a.extent(1), b.extent(1) );
  BOOST_REQUIRE_EQUAL( a.extent(2), b.extent(2) );
  for(int i=0; i<a.extent(0); ++i)
    for(int j=0; j<a.extent(1); ++j)
      for(int k=0; k<a.extent(2); ++k)
        BOOST_CHECK_EQUAL( a(i+a.lbound(0),j+a.lbound(1),k+a.lbound(2)), 
          b(i+b.lbound(0),j+b.lbound(1),k+b.lbound(2)) );
}

template <typename T>
void checkBlitzEqual( const blitz::Array<T,4> a, const blitz::Array<T,4> b ) {
  BOOST_REQUIRE_EQUAL( a.extent(0), b.extent(0) );
  BOOST_REQUIRE_EQUAL( a.extent(1), b.extent(1) );
  BOOST_REQUIRE_EQUAL( a.extent(2), b.extent(2) );
  BOOST_REQUIRE_EQUAL( a.extent(3), b.extent(3) );
  for(int i=0; i<a.extent(0); ++i)
    for(int j=0; j<a.extent(1); ++j)
      for(int k=0; k<a.extent(2); ++k)
        for(int l=0; l<a.extent(3); ++l)
          BOOST_CHECK_EQUAL( 
            a(i+a.lbound(0),j+a.lbound(1),k+a.lbound(2),l+a.lbound(3)), 
            b(i+b.lbound(0),j+b.lbound(1),k+b.lbound(2),l+b.lbound(3)) );
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

/*************************** ALLOCATION TESTS ******************************/
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


/************************** COMPLEX_CAST TESTS *****************************/
BOOST_AUTO_TEST_CASE( test_blitz_array_cast_simple_1D )
{
  blitz::Array<int32_t,1> X(4);
  for(int i=0; i<4; ++i)
    X(i) = i+1;

  blitz::Array<double,1> Y(4), ref(4); 
  Y = Torch::core::cast<double>(X);

  for(int i=0; i<4; ++i)
    ref(i) = static_cast<double>(i+1);

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_tocomplex_1D )
{
  blitz::Array<int32_t,1> X(4);
  for(int i=0; i<4; ++i)
    X(i) = i+1;

  blitz::Array<std::complex<double>,1> Y(4), ref(4); 
  Y = Torch::core::cast<std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    ref(i) = std::complex<double>(i+1,0);

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_tocomplex_2D )
{
  blitz::Array<int32_t,2> X(4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      X(i,j) = i+j+1;

  blitz::Array<std::complex<double>,2> Y(4,4), ref(4,4); 
  Y = Torch::core::cast<std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      ref(i,j) = std::complex<double>(i+j+1,0);

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_tocomplex_3D )
{
  blitz::Array<int32_t,3> X(4,4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        X(i,j,k) = i+j+k+1;

  blitz::Array<std::complex<double>,3> Y(4,4,4), ref(4,4,4); 
  Y = Torch::core::cast<std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        ref(i,j,k) = std::complex<double>(i+j+k+1,0);

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_tocomplex_4D )
{
  blitz::Array<int32_t,4> X(4,4,4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        for(int l=0; l<4; ++l)
          X(i,j,k,l) = i+j+k+l+1;

  blitz::Array<std::complex<double>,4> Y(4,4,4,4), ref(4,4,4,4); 
  Y = Torch::core::cast<std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        for(int l=0; l<4; ++l)
          ref(i,j,k,l) = std::complex<double>(i+j+k+l+1,0);

  checkBlitzEqual( Y, ref);
}


BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromcomplex_1D )
{
  blitz::Array<std::complex<double>,1> X(4);
  for(int i=0; i<4; ++i)
    X(i) = std::complex<double>(i+1,i);

  blitz::Array<int32_t,1> Y(4), ref(4); 
  Y = Torch::core::cast<int32_t >(X);

  for(int i=0; i<4; ++i)
    ref(i) = i+1;

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromtocomplex_1D )
{
  blitz::Array<std::complex<double>,1> X(4);
  for(int i=0; i<4; ++i)
    X(i) = std::complex<double>(i+1,i);

  blitz::Array<std::complex<float>,1> Y(4), ref(4); 
  Y = Torch::core::cast<std::complex<float>,std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    ref(i) = std::complex<float>(i+1,i);

  checkBlitzEqual( Y, ref);

  blitz::Array<std::complex<double>,1> Z(4); 
  Z = Torch::core::cast<std::complex<double>,std::complex<double> >(X);

  checkBlitzEqual( Z, X);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromcomplex_2D )
{
  blitz::Array<std::complex<double>,2> X(4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      X(i,j) = std::complex<double>(i+j+1,i*j);

  blitz::Array<int32_t,2> Y(4,4), ref(4,4); 
  Y = Torch::core::cast<int32_t >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      ref(i,j) = i+j+1;

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromcomplex_3D )
{
  blitz::Array<std::complex<double>,3> X(4,4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        X(i,j,k) = std::complex<double>(i+j+k+1,i*j*k);

  blitz::Array<int32_t,3> Y(4,4,4), ref(4,4,4); 
  Y = Torch::core::cast<int32_t >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        ref(i,j,k) = i+j+k+1;

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromcomplex_4D )
{
  blitz::Array<std::complex<double>,4> X(4,4,4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        for(int l=0; l<4; ++l)
          X(i,j,k,l) = std::complex<double>(i+j+k+l+1,i*j*k*l);

  blitz::Array<int32_t,4> Y(4,4,4,4), ref(4,4,4,4); 
  Y = Torch::core::cast<int32_t >(X);

  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      for(int k=0; k<4; ++k)
        for(int l=0; l<4; ++l)
          ref(i,j,k,l) = i+j+k+l+1;

  checkBlitzEqual( Y, ref);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_check_C_fortran)
{
  blitz::Array<uint8_t,2> a(4,7);
  a = 0;
  // Check contiguous C-style array
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(a), true);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(a), true);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(a), true);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(a), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(a), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(a), false);

  blitz::Array<uint8_t,2> b = a.transpose(1,0);
  // Check non-contiguous C-style array
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(b), true);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(b), false);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(b), false);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(b), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(b), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(b), false);

  blitz::Array<uint8_t,2> c(blitz::Range(1,4),blitz::Range(1,2));
  c = 0;
  // Check contiguous C-style array (non-zero base indices)
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(c), false);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(c), true);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(c), false);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(c), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(c), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(c), false);

  blitz::Array<uint8_t,2> d(5,2,blitz::FortranArray<2>());
  d = 0;
  // Check contiguous Fortran array
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(d), false);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(d), false);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(d), false);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(d), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(d), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(d), true);

  blitz::Array<uint8_t,2> e = d.transpose(1,0);
  // Check non-contiguous Fortran array
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(e), false);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(e), true);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(e), false);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(e), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(e), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(e), false);

  blitz::Array<uint8_t,2> f(blitz::Range(0,4), blitz::Range(0,2),blitz::FortranArray<2>());
  f = 0;
  // Check contiguous C-style array (non-zero base indices)
  BOOST_CHECK_EQUAL( Torch::core::isZeroBase(f), true);
  BOOST_CHECK_EQUAL( Torch::core::isCContiguous(f), false);
  BOOST_CHECK_EQUAL( Torch::core::isCZeroBaseContiguous(f), false);
  BOOST_CHECK_EQUAL( Torch::core::isOneBase(f), false);
  BOOST_CHECK_EQUAL( Torch::core::isFortranContiguous(f), true);
  BOOST_CHECK_EQUAL( Torch::core::isFortranOneBaseContiguous(f), false);
}

BOOST_AUTO_TEST_SUITE_END()
