/**
 * @file src/cxx/database/test/arrayset.cc
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Database Arrayset tests
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DbArrayset Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <vector>

#include <blitz/array.h>
#include "core/cast.h"
#include "database/BinFile.h"
#include "database/Arrayset.h"

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
 * @brief Generates a unique temporary filename, and returns the file
 * descriptor
 */
std::string temp_file() {
  boost::filesystem::path tpl = Torch::core::tmpdir();
  tpl /= "torchtest_core_binformatXXXXXX.bin";
  boost::shared_array<char> char_tpl(new char[tpl.file_string().size()+1]);
  strcpy(char_tpl.get(), tpl.file_string().c_str());
  int fd = mkstemps(char_tpl.get(),4);
  close(fd);
  boost::filesystem::remove(char_tpl.get());
  std::string res = char_tpl.get();
  return res;
//  return char_tpl.get();
}

template<typename T, typename U> 
void check_equal_1d(const blitz::Array<T,1>& a, const blitz::Array<U,1>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  for (int i=0; i<a.extent(0); ++i) {
    BOOST_CHECK_EQUAL(a(i), Torch::core::cast<T>(b(i)) );
  }
}

template<typename T, typename U> 
void check_equal_2d(const blitz::Array<T,2>& a, const blitz::Array<U,2>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      BOOST_CHECK_EQUAL(a(i,j), Torch::core::cast<T>(b(i,j)));
    }
  }
}

template<typename T, typename U> 
void check_equal_4d(const blitz::Array<T,4>& a, const blitz::Array<U,4>& b) 
{
  BOOST_REQUIRE_EQUAL(a.extent(0), b.extent(0));
  BOOST_REQUIRE_EQUAL(a.extent(1), b.extent(1));
  BOOST_REQUIRE_EQUAL(a.extent(2), b.extent(2));
  BOOST_REQUIRE_EQUAL(a.extent(3), b.extent(3));
  for (int i=0; i<a.extent(0); ++i) {
    for (int j=0; j<a.extent(1); ++j) {
      for (int k=0; k<a.extent(2); ++k) {
        for (int l=0; l<a.extent(3); ++l) {
          BOOST_CHECK_EQUAL(a(i,j,k,l), Torch::core::cast<T>(b(i,j,k,l)));
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

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a(new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_b(new Torch::database::Array(b));
  boost::shared_ptr<Torch::database::Array> db_c(new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar(vec);

  // Get the ids of the Arrays
  std::vector<size_t> ids;
  db_Ar.index( ids);
  std::vector<size_t>::const_iterator it=ids.begin();

  // Get the content of the database Arrays in blitz Arrays
  blitz::Array<double,1> a_get = db_Ar[*it].get<double,1>();
  ++it;
  blitz::Array<double,1> b_get = db_Ar[*it].get<double,1>();
  ++it;
  blitz::Array<double,1> c_get = db_Ar[*it].get<double,1>();
  ++it;
 
  // Check that the content remains unchanged
  check_equal_1d( a, a_get);
  check_equal_1d( b, b_get);
  check_equal_1d( c, c_get);

  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), Torch::database::DimensionError );
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

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a(new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_b(new Torch::database::Array(b));
  boost::shared_ptr<Torch::database::Array> db_c(new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.getNSamples(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);

  // Save the Arrayset to a file
  std::string tmp_file = temp_file();
  db_Ar.save( tmp_file );
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare("torch.arrayset.binary"), 0);
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar.add(g), Torch::database::DimensionError );
  
  // Create an Arrayset from a file and check its properties
  Torch::database::Arrayset db_Ar_read(tmp_file);
  BOOST_CHECK_EQUAL(db_Ar.getId(), db_Ar_read.getId());
  BOOST_CHECK_EQUAL(db_Ar.getRole().compare( db_Ar_read.getRole()), 0);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), db_Ar_read.isLoaded());
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), db_Ar_read.getElementType());
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), db_Ar_read.getNDim());
  BOOST_CHECK_EQUAL(db_Ar.getNSamples(), db_Ar_read.getNSamples());
  for( size_t i=0; i<db_Ar.getNDim(); ++i)
    BOOST_CHECK_EQUAL(db_Ar.getShape()[i], db_Ar_read.getShape()[i]);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(db_Ar_read.getFilename()), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare(db_Ar_read.getCodec()->name()), 0); 
  // Check that adding a blitz arrays with different dimensions will raise
  // an exception
  BOOST_CHECK_THROW( db_Ar_read.add(g), Torch::database::DimensionError );

  db_Ar_read.load();
  BOOST_CHECK_EQUAL(db_Ar_read.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar_read.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar_read.getCodec().use_count(), 0);  
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

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a(new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_b(new Torch::database::Array(b));
  boost::shared_ptr<Torch::database::Array> db_c(new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.getNSamples(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);

  
  // Get the ids of the Arrays
  std::vector<size_t> ids;
  db_Ar.index( ids);
  // Check the content
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( b, db_Ar[ids[1]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[2]].get<double,1>() );

  // Check that an exception is thrown when accessing a non-existent array
  BOOST_CHECK_THROW( db_Ar[137], Torch::database::IndexError );

  // Check the content when using the cast function
  check_equal_1d( a, db_Ar[ids[0]].cast<uint32_t,1>() );

  // Remove the second array and check
  db_Ar.remove( ids[1] );
  ids.clear();
  db_Ar.index( ids);
  BOOST_CHECK_EQUAL( ids.size(), 2); 
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[1]].get<double,1>() );

  // Add array and check
  db_Ar.add( db_b );
  ids.clear();
  db_Ar.index( ids);
  BOOST_CHECK_EQUAL( ids.size(), 3); 
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[1]].get<double,1>() );
  check_equal_1d( b, db_Ar[ids[2]].get<double,1>() );
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

  // Create database Arrays from blitz::arrays
  boost::shared_ptr<Torch::database::Array> db_a(new Torch::database::Array(a));
  boost::shared_ptr<Torch::database::Array> db_b(new Torch::database::Array(b));
  boost::shared_ptr<Torch::database::Array> db_c(new Torch::database::Array(c));

  // Put these database Arrays in a STL vector
  std::vector<boost::shared_ptr<Torch::database::Array> > vec;
  vec.push_back(db_a);
  vec.push_back(db_b);
  vec.push_back(db_c);

  // Create an Arrayset from the STL vector
  Torch::database::Arrayset db_Ar(vec);
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), true);
  BOOST_CHECK_EQUAL(db_Ar.getElementType(), Torch::core::array::t_float64);
  BOOST_CHECK_EQUAL(db_Ar.getNDim(), 1);
  BOOST_CHECK_EQUAL(db_Ar.getNSamples(), 3);
  BOOST_CHECK_EQUAL(db_Ar.getShape()[0], 4);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(""), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec().use_count(), 0);
  
  // Get the ids of the Arrays
  std::vector<size_t> ids;
  db_Ar.index( ids);
  // Check the content
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( b, db_Ar[ids[1]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[2]].get<double,1>() );

  // Save the Arrayset to a file
  std::string tmp_file = temp_file();
  db_Ar.save( tmp_file );
  BOOST_CHECK_EQUAL(db_Ar.isLoaded(), false);
  BOOST_CHECK_EQUAL(db_Ar.getFilename().compare(tmp_file), 0);
  BOOST_CHECK_EQUAL(db_Ar.getCodec()->name().compare("torch.arrayset.binary"), 0);
  // Check data
  ids.clear();
  db_Ar.index( ids);
  BOOST_CHECK_EQUAL( ids.size(), 3); 
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( b, db_Ar[ids[1]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[2]].get<double,1>() );

  // Check the content when using the cast function
  check_equal_1d( a, db_Ar[ids[0]].cast<uint32_t,1>() );

  // Remove the second array and check
  db_Ar.remove( 2);
  ids.clear();
  db_Ar.index( ids);
  BOOST_CHECK_EQUAL( ids.size(), 2); 
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[1]].get<double,1>() );

  // Add array and check
  db_Ar.add( db_b );
  ids.clear();
  db_Ar.index( ids);
  BOOST_CHECK_EQUAL( ids.size(), 3); 
  check_equal_1d( a, db_Ar[ids[0]].get<double,1>() );
  check_equal_1d( c, db_Ar[ids[1]].get<double,1>() );
  check_equal_1d( b, db_Ar[ids[2]].get<double,1>() );
}

BOOST_AUTO_TEST_SUITE_END()

