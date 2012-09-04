/**
 * @file cxx/io/test/arrayset.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief IO Arrayset tests
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
#define BOOST_TEST_MODULE DbArrayset Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <vector>

#include <blitz/array.h>
#include <string>
#include "core/logging.h" // for bob::core::tmpdir()
#include "core/cast.h"
#include "io/BinFile.h"
#include "io/Arrayset.h"

struct T {
  blitz::Array<double,1> a;
  blitz::Array<double,1> b;
  blitz::Array<double,1> c;

  blitz::Array<float,2> d;
  blitz::Array<float,2> e;
  blitz::Array<float,2> f;

  blitz::Array<double,4> g;
  blitz::Array<double,4> h;

  T() {
    a.resize(4);
    a = 1, 2, 3, 4;
    c.resize(4);
    c = 1, 2, 3, 4;

    d.resize(2,2);
    d = 1, 2, 3, 4;
    e.resize(2,2);
    e = 5, 6, 7, 8;

    g.resize(2,3,4,5);
    g = 37.;
  }

  ~T() { }

};


/**
 * Use this static variable to count the check_equal() calls so you can know
 * which of the tests failed. Uncomment the first line in each check_equal()
 * implementation to printout the compared arrays and an instance number.
 */
//static size_t counter = 0;

template<typename T, typename U> 
void check_equal(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), bob::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), bob::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  //std::cout << "[" << counter++ << "] " << a << " against " << b << std::endl;
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  BOOST_REQUIRE_EQUAL(a.extent(3), b.extent(3));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          BOOST_CHECK_EQUAL(a(i,j,k,l), bob::core::cast<T>(b(i,j,k,l)));
        }
      }
    }
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( dbArrayset_construction_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  bob::io::Array db_a(a);
  bob::io::Array db_b(b);
  bob::io::Array db_c(c);

  // Put these io Arrays in a STL vector
  std::vector<bob::io::Array> vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  bob::io::Arrayset db_Ar(vec);

  // Set and get attributes
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), bob::core::array::t_float64 );
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1 );
  const size_t* shape = db_Ar.getShape();
  BOOST_CHECK_EQUAL(shape[0], 4 );
  BOOST_CHECK_EQUAL(db_Ar.size(), 3 );

  // Get the content of the io Arrays in blitz Arrays
  blitz::Array<double,1> a_get = db_Ar[0].get<double,1>();
  blitz::Array<double,1> b_get = db_Ar[1].get<double,1>();
  blitz::Array<double,1> c_get = db_Ar[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);

  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), std::invalid_argument );


  // Copy constructor
  bob::io::Arrayset db_Ar2(db_Ar);
  
  // Check attributes 
  BOOST_CHECK_EQUAL( db_Ar2.getElementType(), bob::core::array::t_float64 );
  BOOST_CHECK_EQUAL( db_Ar2.getNDim(), 1 );
  shape = db_Ar2.getShape();
  BOOST_CHECK_EQUAL( shape[0], 4 );
  BOOST_CHECK_EQUAL( db_Ar2.size(), 3 );

  // Get the content of the io Arrays in blitz Arrays
  a_get = db_Ar2[0].get<double,1>();
  b_get = db_Ar2[1].get<double,1>();
  c_get = db_Ar2[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);
 
  // Assignment
  db_Ar2 = db_Ar;
  
  // Check attributes 
  BOOST_CHECK_EQUAL( db_Ar2.getElementType(), bob::core::array::t_float64 );
  BOOST_CHECK_EQUAL( db_Ar2.getNDim(), 1 );
  shape = db_Ar2.getShape();
  BOOST_CHECK_EQUAL( shape[0], 4 );
  BOOST_CHECK_EQUAL( db_Ar2.size(), 3 );

  // Get the content of the io Arrays in blitz Arrays
  a_get = db_Ar2[0].get<double,1>();
  b_get = db_Ar2[1].get<double,1>();
  c_get = db_Ar2[2].get<double,1>();
 
  // Check that the content remains unchanged
  check_equal(a, a_get);
  check_equal(b, b_get);
  check_equal(c, c_get);
}

BOOST_AUTO_TEST_CASE( dbArrayset_loadsave_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  bob::io::Array db_a(a);
  bob::io::Array db_b(b);
  bob::io::Array db_c(c);

  // Put these io Arrays in a STL vector
  std::vector<bob::io::Array > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  BOOST_REQUIRE_NO_THROW(bob::io::Arrayset db_Ar(vec));
  bob::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), bob::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);

  // Save the Arrayset to a file
  std::string tmp_file = bob::core::tmpfile();
  BOOST_REQUIRE_NO_THROW(db_Ar.save( tmp_file ));
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), std::invalid_argument );
  
  // Create an Arrayset from a file and check its properties
  bob::io::Arrayset db_Ar_read(tmp_file);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), db_Ar_read.getElementType());
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), db_Ar_read.getNDim());
  BOOST_CHECK_EQUAL(db_Ar.size(), db_Ar_read.size());
  for( size_t i=0; i<db_Ar.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_Ar.getShape()[i], db_Ar_read.getShape()[i]);
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar_read.add(g), std::invalid_argument );

  db_Ar_read.load();

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_CASE( dbArrayset_cast_remove_inline )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  bob::io::Array db_a(a);
  bob::io::Array db_b(b);
  bob::io::Array db_c(c);

  // Put these io Arrays in a STL vector
  std::vector<bob::io::Array > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  bob::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), bob::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);

  // Check the content
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(b, db_Ar[1].get<double,1>());
  check_equal(c, db_Ar[2].get<double,1>());

  // Check that an exception is thrown when accessing a non-existent array
  BOOST_CHECK_THROW(db_Ar[137], std::out_of_range);

  // Check the content when using the cast function
  check_equal(a, db_Ar[0].cast<uint32_t,1>());

  // Remove the second array and check
  db_Ar.remove(1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 2); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());

  // Add a blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());

  // Add a io array and check
  bob::io::Array db_b2(b);
  db_Ar.add(db_b2);
  BOOST_CHECK_EQUAL(db_Ar.size(), 4); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());

  // Add a blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 5); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());
  check_equal(b, db_Ar[4].get<double,1>());

  // Add a io array and check
  bob::io::Array db_b3(b);
  db_Ar.add(db_b3);
  BOOST_CHECK_EQUAL(db_Ar.size(), 6); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());
  check_equal(b, db_Ar[3].get<double,1>());
  check_equal(b, db_Ar[4].get<double,1>());
  check_equal(b, db_Ar[5].get<double,1>());
}

BOOST_AUTO_TEST_CASE( dbArrayset_remove_external )
{
  // Initialize some blitz arrays
  b.resize(4);
  b = 33.;
  b(0) = 2.;
  c.resize(4);
  c = 23.;
  c(0) = 3.;

  // Create io Arrays from blitz::arrays
  bob::io::Array db_a(a);
  bob::io::Array db_b(b);
  bob::io::Array db_c(c);

  // Put these io Arrays in a STL vector
  std::vector<bob::io::Array > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  BOOST_REQUIRE_NO_THROW(bob::io::Arrayset db_Ar(vec));
  bob::io::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), bob::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  
  // Check the content
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( b, db_Ar[1].get<double,1>() );
  check_equal( c, db_Ar[2].get<double,1>() );

  // Save the Arrayset to a file
  std::string tmp_file = bob::core::tmpfile();
  BOOST_REQUIRE_NO_THROW(db_Ar.save( tmp_file ));
  // Check data
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( b, db_Ar[1].get<double,1>() );
  check_equal( c, db_Ar[2].get<double,1>() );

  // Check the content when using the cast function
  check_equal( a, db_Ar[0].cast<uint32_t,1>() );

  // Remove the second array and check
  db_Ar.remove(1);
  BOOST_CHECK_EQUAL(db_Ar.size(), 2); 
  check_equal( a, db_Ar[0].get<double,1>() );
  check_equal( c, db_Ar[1].get<double,1>() );

  // Add blitz array and check
  db_Ar.add(b);
  BOOST_CHECK_EQUAL(db_Ar.size(), 3); 
  check_equal(a, db_Ar[0].get<double,1>());
  check_equal(c, db_Ar[1].get<double,1>());
  check_equal(b, db_Ar[2].get<double,1>());

  // Clean-up
  boost::filesystem::remove(tmp_file);
}

BOOST_AUTO_TEST_SUITE_END()
