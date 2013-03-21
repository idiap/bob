/**
 * @file core/cxx/test/blitzArray.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Extensive Blitz Array tests
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
#define BOOST_TEST_MODULE Core-BlitzArray Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>
#include <bob/core/cast.h>
#include <bob/core/check.h>
#include <bob/core/array_copy.h>
#include <map>
#include <vector>

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
  && defined(HAVE_BLITZ_SPECIAL_TYPES))
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
  Y = bob::core::cast<double>(X);

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
  Y = bob::core::cast<std::complex<double> >(X);

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
  Y = bob::core::cast<std::complex<double> >(X);

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
  Y = bob::core::cast<std::complex<double> >(X);

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
  Y = bob::core::cast<std::complex<double> >(X);

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
  Y = bob::core::cast<int32_t >(X);

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
  Y = bob::core::cast<std::complex<float>,std::complex<double> >(X);

  for(int i=0; i<4; ++i)
    ref(i) = std::complex<float>(i+1,i);

  checkBlitzEqual( Y, ref);

  blitz::Array<std::complex<double>,1> Z(4); 
  Z = bob::core::cast<std::complex<double>,std::complex<double> >(X);

  checkBlitzEqual( Z, X);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_cast_fromcomplex_2D )
{
  blitz::Array<std::complex<double>,2> X(4,4);
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j)
      X(i,j) = std::complex<double>(i+j+1,i*j);

  blitz::Array<int32_t,2> Y(4,4), ref(4,4); 
  Y = bob::core::cast<int32_t >(X);

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
  Y = bob::core::cast<int32_t >(X);

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
  Y = bob::core::cast<int32_t >(X);

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
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(a), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(a), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(a), true);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(a), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(a), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(a), false);

  blitz::Array<uint8_t,2> b = a.transpose(1,0);
  // Check non-contiguous C-style array
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(b), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(b), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(b), false);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(b), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(b), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(b), false);

  blitz::Array<uint8_t,2> c(blitz::Range(1,4),blitz::Range(1,2));
  c = 0;
  // Check contiguous C-style array (non-zero base indices)
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(c), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(c), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(c), false);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(c), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(c), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(c), false);

  blitz::Array<uint8_t,2> d(5,2,blitz::FortranArray<2>());
  d = 0;
  // Check contiguous Fortran array
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(d), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(d), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(d), false);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(d), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(d), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(d), true);

  blitz::Array<uint8_t,2> e = d.transpose(1,0);
  // Check non-contiguous Fortran array
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(e), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(e), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(e), false);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(e), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(e), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(e), false);

  blitz::Array<uint8_t,2> f(blitz::Range(0,4), blitz::Range(0,2),blitz::FortranArray<2>());
  f = 0;
  // Check contiguous C-style array (non-zero base indices)
  BOOST_CHECK_EQUAL( bob::core::array::isZeroBase(f), true);
  BOOST_CHECK_EQUAL( bob::core::array::isCContiguous(f), false);
  BOOST_CHECK_EQUAL( bob::core::array::isCZeroBaseContiguous(f), false);
  BOOST_CHECK_EQUAL( bob::core::array::isOneBase(f), false);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranContiguous(f), true);
  BOOST_CHECK_EQUAL( bob::core::array::isFortranOneBaseContiguous(f), false);
}

BOOST_AUTO_TEST_CASE( test_blitz_array_vector_map_ccopy )
{
  blitz::Array<uint8_t,1> x1(4), x2(4);
  x1 = 1, 2, 3, 4;
  x2 = 5, 6, 7, 8;

  // 1/ vector
  std::vector<blitz::Array<uint8_t,1> > v1, v2;
  v1.push_back(x1);
  v1.push_back(x2);
  // Copies the vector
  bob::core::array::ccopy(v1, v2);
  // Checks that the vectors are equal
  BOOST_CHECK_EQUAL( v1.size(), v2.size());
  checkBlitzEqual(v1[0], v2[0]);
  checkBlitzEqual(v1[1], v2[1]);

  // 2/ map
  std::map<size_t,blitz::Array<uint8_t,1> > m1, m2;
  m1[1].resize(x1.shape());
  m1[1] = x1;
  m1[4].resize(x2.shape());
  m1[4] = x2;
  // Copies the map
  bob::core::array::ccopy(m1, m2);
  // Checks that the vectors are equal
  BOOST_CHECK_EQUAL( m1.size(), m2.size());
  checkBlitzEqual(m1[1], m2[1]);
  checkBlitzEqual(m1[4], m2[4]);
}


BOOST_AUTO_TEST_SUITE_END()
